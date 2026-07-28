# Fully nested meta-learning across every cached canonical-fold OOF source.
#
# Two genuinely learned stackers are evaluated:
#   1. a ridge-selected conditional-logit/logarithmic opinion pool; and
#   2. a deliberately shallow multiclass xgboost meta-model.
#
# Hyperparameters are selected inside each held-out meta fold using only the
# other four canonical folds. The cached-OOF convention matches the project's
# earlier stacking analysis; no base-model in-sample predictions are used.

options(stringsAsFactors = FALSE)

suppressPackageStartupMessages(library(xgboost))
source("R/codex_modeling_common.R")

output_dir <- "data_processed/codex_deep_stack"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
stack_stage <- Sys.getenv("CODEX_STACK_STAGE", "all")
stopifnot(stack_stage %in% c("all", "logpool_refine"))
output_stem <- if (stack_stage == "all") {
  "full_stacking"
} else {
  "logpool_refine"
}

train <- read.csv("csv files/train.csv")
truth <- as.matrix(train[, paste0("Ch", 1:4)])
fold_map <- canonical_fold_map()
row_fold <- unname(fold_map[as.character(train$Case)])
stopifnot(!anyNA(row_fold), all(row_fold %in% 1:5))

base <- readRDS("data_processed/oof_ensemble_v10.rds")
rank_prediction <-
  readRDS("data_processed/codex/rank_oof.rds")[[1]]$pred
retuned_prediction <-
  readRDS("data_processed/codex/xgb_retune_oof.rds")[[1]]$pred
cox_prediction <-
  readRDS("data_processed/codex/cox_oof.rds")$oof_min
shallow_prediction <- readRDS(
  "data_processed/codex_behavioral_round/mlp_oof.rds"
)$oof[["h08_d0.100"]]
triple_prediction <- readRDS(
  "data_processed/codex_triples/triple_oof.rds"
)$oof[["triple_price_income_miles"]]
deep_prediction <- readRDS(
  file.path(output_dir, "torch_deep_oof.rds")
)$deep_oof

all_components <- list(
  mlogit = base$oof_mlogit,
  original_xgb = base$oof_xgb,
  rank_ndcg = rank_prediction,
  retuned_xgb = retuned_prediction,
  cox = cox_prediction,
  shallow_mlp = shallow_prediction,
  triple_mlogit = triple_prediction,
  deep_mlp = deep_prediction
)
stopifnot(all(vapply(all_components, function(prediction) {
  is.matrix(prediction) &&
    identical(dim(prediction), dim(truth)) &&
    !anyNA(prediction) &&
    all(is.finite(prediction)) &&
    max(abs(rowSums(prediction) - 1)) < 1e-6
}, logical(1))))

known_losses <- c(
  mlogit = 1.1470212110518,
  original_xgb = 1.178668,
  rank_ndcg = 1.163610,
  retuned_xgb = 1.176029,
  cox = 1.164331,
  shallow_mlp = 1.19054334979533,
  triple_mlogit = 1.146782
)
actual_losses <- vapply(
  all_components[names(known_losses)],
  function(x) log_loss_matrix(truth, x),
  numeric(1)
)
stopifnot(
  abs(actual_losses[["mlogit"]] -
    known_losses[["mlogit"]]) < 1e-8,
  abs(actual_losses[["original_xgb"]] -
    known_losses[["original_xgb"]]) < 1e-5,
  abs(actual_losses[["rank_ndcg"]] -
    known_losses[["rank_ndcg"]]) < 2e-3,
  abs(actual_losses[["retuned_xgb"]] -
    known_losses[["retuned_xgb"]]) < 2e-3,
  abs(actual_losses[["cox"]] -
    known_losses[["cox"]]) < 2e-3,
  abs(actual_losses[["shallow_mlp"]] -
    known_losses[["shallow_mlp"]]) < 1e-8,
  abs(actual_losses[["triple_mlogit"]] -
    known_losses[["triple_mlogit"]]) < 2e-6
)

current <- readRDS(
  "data_processed/codex_behavioral_round/mlp_precision.rds"
)$crossfit_prediction
v11 <- 0.8 * all_components$mlogit +
  0.2 * all_components$original_xgb
stopifnot(
  abs(log_loss_matrix(truth, v11) -
    1.14509421298673) < 1e-8,
  abs(log_loss_matrix(truth, current) -
    1.14378944178118) < 1e-8
)

pools <- list(
  requested7 = c(
    "mlogit", "original_xgb", "rank_ndcg", "retuned_xgb",
    "cox", "shallow_mlp", "triple_mlogit"
  ),
  augmented8 = c(
    "mlogit", "original_xgb", "rank_ndcg", "retuned_xgb",
    "cox", "shallow_mlp", "triple_mlogit", "deep_mlp"
  )
)

log_component_array <- function(components) {
  array_value <- array(
    NA_real_,
    dim = c(nrow(truth), 4L, length(components)),
    dimnames = list(NULL, paste0("alt", 1:4), names(components))
  )
  for (index in seq_along(components)) {
    array_value[, , index] <-
      log(pmax(components[[index]], 1e-9))
  }
  array_value
}

predict_log_pool <- function(beta, design, rows) {
  utility <- matrix(0, length(rows), 4L)
  for (index in seq_along(beta)) {
    utility <- utility +
      beta[[index]] * design[rows, , index]
  }
  utility <- utility - apply(utility, 1L, max)
  probability <- exp(utility)
  probability / rowSums(probability)
}

fit_log_pool <- function(design, rows, lambda) {
  n_components <- dim(design)[[3]]
  evaluate <- function(beta) {
    utility <- matrix(0, length(rows), 4L)
    for (index in seq_len(n_components)) {
      utility <- utility +
        beta[[index]] * design[rows, , index]
    }
    utility <- utility - apply(utility, 1L, max)
    probability <- exp(utility)
    probability <- probability / rowSums(probability)
    value <- log_loss_matrix(
      truth[rows, , drop = FALSE], probability
    ) + lambda * sum(beta^2)
    gradient <- numeric(n_components)
    residual <- truth[rows, , drop = FALSE] - probability
    for (index in seq_len(n_components)) {
      gradient[[index]] <- -mean(rowSums(
        residual * design[rows, , index]
      )) + 2 * lambda * beta[[index]]
    }
    list(value = value, gradient = gradient)
  }
  initial <- rep(0, n_components)
  initial[[1]] <- 1
  fitted <- optim(
    initial,
    function(beta) evaluate(beta)$value,
    gr = function(beta) evaluate(beta)$gradient,
    method = "BFGS",
    control = list(maxit = 1000, reltol = 1e-10)
  )
  stopifnot(fitted$convergence == 0L)
  list(beta = fitted$par, objective = fitted$value)
}

log_pool_lambdas <- if (stack_stage == "logpool_refine") {
  # The first nested run already established that 0.01 beats every smaller
  # value in all five outer folds. The refinement only needs to test whether
  # stronger shrinkage improves further.
  c(1e-2, 3e-2, 1e-1, 3e-1)
} else {
  c(0, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1)
}

nested_log_pool <- function(component_names) {
  components <- all_components[component_names]
  design <- log_component_array(components)
  prediction <- matrix(NA_real_, nrow(truth), 4L)
  beta_rows <- list()
  selection_rows <- list()

  for (outer_fold in 1:5) {
    inner_folds <- setdiff(1:5, outer_fold)
    inner_losses <- numeric(length(log_pool_lambdas))
    for (lambda_index in seq_along(log_pool_lambdas)) {
      lambda <- log_pool_lambdas[[lambda_index]]
      inner_prediction <- matrix(NA_real_, nrow(truth), 4L)
      inner_rows_used <- row_fold != outer_fold
      for (inner_fold in inner_folds) {
        fit_rows <- which(
          row_fold != outer_fold & row_fold != inner_fold
        )
        validation_rows <- which(row_fold == inner_fold)
        fitted <- fit_log_pool(design, fit_rows, lambda)
        inner_prediction[validation_rows, ] <-
          predict_log_pool(
            fitted$beta, design, validation_rows
          )
      }
      inner_losses[[lambda_index]] <- log_loss_matrix(
        truth[inner_rows_used, , drop = FALSE],
        inner_prediction[inner_rows_used, , drop = FALSE]
      )
    }
    best_index <- which.min(inner_losses)
    selected_lambda <- log_pool_lambdas[[best_index]]
    outer_fit_rows <- which(row_fold != outer_fold)
    outer_validation_rows <- which(row_fold == outer_fold)
    fitted <- fit_log_pool(
      design, outer_fit_rows, selected_lambda
    )
    prediction[outer_validation_rows, ] <-
      predict_log_pool(
        fitted$beta, design, outer_validation_rows
      )
    beta_rows[[outer_fold]] <- data.frame(
      outer_fold = outer_fold,
      component = component_names,
      beta = fitted$beta,
      selected_lambda = selected_lambda
    )
    selection_rows[[outer_fold]] <- data.frame(
      outer_fold = outer_fold,
      lambda = log_pool_lambdas,
      inner_logloss = inner_losses,
      selected = seq_along(log_pool_lambdas) == best_index
    )
    cat(sprintf(
      "log-pool outer fold %d: lambda %.5g, loss %.9f\n",
      outer_fold, selected_lambda,
      log_loss_matrix(
        truth[outer_validation_rows, , drop = FALSE],
        prediction[outer_validation_rows, , drop = FALSE]
      )
    ))
    flush.console()
  }
  stopifnot(!anyNA(prediction))
  list(
    prediction = prediction,
    logloss = log_loss_matrix(truth, prediction),
    beta = do.call(rbind, beta_rows),
    selection = do.call(rbind, selection_rows)
  )
}

meta_xgb_configs <- list(
  stump100 = list(
    max_depth = 1L, eta = 0.03, nrounds = 100L,
    min_child_weight = 20, reg_lambda = 10
  ),
  depth2_100 = list(
    max_depth = 2L, eta = 0.03, nrounds = 100L,
    min_child_weight = 20, reg_lambda = 10
  ),
  depth2_200 = list(
    max_depth = 2L, eta = 0.015, nrounds = 200L,
    min_child_weight = 30, reg_lambda = 20
  )
)

meta_matrix <- function(components) {
  value <- do.call(cbind, lapply(names(components), function(name) {
    prediction <- log(pmax(components[[name]], 1e-9))
    colnames(prediction) <- paste0(name, "_alt", 1:4)
    prediction
  }))
  storage.mode(value) <- "double"
  value
}

fit_meta_xgb <- function(x, rows, config, seed) {
  xgb.train(
    params = list(
      objective = "multi:softprob",
      num_class = 4L,
      eval_metric = "mlogloss",
      max_depth = config$max_depth,
      eta = config$eta,
      min_child_weight = config$min_child_weight,
      subsample = 0.85,
      colsample_bytree = 0.85,
      reg_alpha = 0,
      reg_lambda = config$reg_lambda,
      seed = seed,
      nthread = 1L
    ),
    data = xgb.DMatrix(
      x[rows, , drop = FALSE],
      label = max.col(truth[rows, , drop = FALSE]) - 1L
    ),
    nrounds = config$nrounds,
    verbose = 0
  )
}

predict_meta_xgb <- function(model, x, rows) {
  prediction <- predict(
    model,
    xgb.DMatrix(x[rows, , drop = FALSE])
  )
  prediction <- as.matrix(prediction)
  prediction / rowSums(prediction)
}

nested_meta_xgb <- function(component_names) {
  components <- all_components[component_names]
  x <- meta_matrix(components)
  prediction <- matrix(NA_real_, nrow(truth), 4L)
  selection_rows <- list()

  for (outer_fold in 1:5) {
    inner_folds <- setdiff(1:5, outer_fold)
    inner_losses <- numeric(length(meta_xgb_configs))
    for (config_index in seq_along(meta_xgb_configs)) {
      config <- meta_xgb_configs[[config_index]]
      inner_prediction <- matrix(NA_real_, nrow(truth), 4L)
      inner_rows_used <- row_fold != outer_fold
      for (inner_fold in inner_folds) {
        fit_rows <- which(
          row_fold != outer_fold & row_fold != inner_fold
        )
        validation_rows <- which(row_fold == inner_fold)
        fitted <- fit_meta_xgb(
          x, fit_rows, config,
          seed = 4821L + outer_fold * 100L +
            inner_fold * 10L + config_index
        )
        inner_prediction[validation_rows, ] <-
          predict_meta_xgb(fitted, x, validation_rows)
      }
      inner_losses[[config_index]] <- log_loss_matrix(
        truth[inner_rows_used, , drop = FALSE],
        inner_prediction[inner_rows_used, , drop = FALSE]
      )
    }
    best_index <- which.min(inner_losses)
    selected_name <- names(meta_xgb_configs)[[best_index]]
    selected_config <- meta_xgb_configs[[best_index]]
    outer_fit_rows <- which(row_fold != outer_fold)
    outer_validation_rows <- which(row_fold == outer_fold)
    fitted <- fit_meta_xgb(
      x, outer_fit_rows, selected_config,
      seed = 5821L + outer_fold
    )
    prediction[outer_validation_rows, ] <-
      predict_meta_xgb(fitted, x, outer_validation_rows)
    selection_rows[[outer_fold]] <- data.frame(
      outer_fold = outer_fold,
      config = names(meta_xgb_configs),
      inner_logloss = inner_losses,
      selected = seq_along(meta_xgb_configs) == best_index
    )
    cat(sprintf(
      "meta-xgb outer fold %d: %s, loss %.9f\n",
      outer_fold, selected_name,
      log_loss_matrix(
        truth[outer_validation_rows, , drop = FALSE],
        prediction[outer_validation_rows, , drop = FALSE]
      )
    ))
    flush.console()
  }
  stopifnot(!anyNA(prediction))
  list(
    prediction = prediction,
    logloss = log_loss_matrix(truth, prediction),
    selection = do.call(rbind, selection_rows)
  )
}

softmax_weights <- function(theta, n_components) {
  value <- c(theta, 0)
  value <- value - max(value)
  exp(value) / sum(exp(value))
}

fit_arithmetic <- function(components, rows) {
  n_components <- length(components)
  chosen <- do.call(cbind, lapply(components, function(prediction) {
    rowSums(
      truth[rows, , drop = FALSE] *
        prediction[rows, , drop = FALSE]
    )
  }))
  evaluate <- function(theta) {
    weights <- softmax_weights(theta, n_components)
    mixed <- as.numeric(chosen %*% weights)
    value <- -mean(log(pmax(mixed, 1e-15)))
    gradient_weight <- -colMeans(chosen / mixed)
    weighted_gradient <- sum(weights * gradient_weight)
    gradient <- weights[seq_len(n_components - 1L)] *
      (
        gradient_weight[seq_len(n_components - 1L)] -
          weighted_gradient
      )
    list(value = value, gradient = gradient)
  }
  fitted <- optim(
    rep(0, n_components - 1L),
    function(theta) evaluate(theta)$value,
    gr = function(theta) evaluate(theta)$gradient,
    method = "BFGS",
    control = list(maxit = 1000, reltol = 1e-10)
  )
  stopifnot(fitted$convergence == 0L)
  softmax_weights(fitted$par, n_components)
}

crossfit_arithmetic <- function(component_names) {
  components <- all_components[component_names]
  prediction <- matrix(NA_real_, nrow(truth), 4L)
  weight_rows <- list()
  for (fold in 1:5) {
    fit_rows <- which(row_fold != fold)
    validation_rows <- row_fold == fold
    weights <- fit_arithmetic(components, fit_rows)
    prediction[validation_rows, ] <- Reduce(
      `+`,
      Map(
        function(x, weight) {
          x[validation_rows, , drop = FALSE] * weight
        },
        components, weights
      )
    )
    weight_rows[[fold]] <- data.frame(
      outer_fold = fold,
      component = component_names,
      weight = weights
    )
  }
  list(
    prediction = prediction,
    logloss = log_loss_matrix(truth, prediction),
    weights = do.call(rbind, weight_rows)
  )
}

results <- list()
for (pool_name in names(pools)) {
  cat(sprintf("\n=== pool %s ===\n", pool_name))
  components <- pools[[pool_name]]
  if (stack_stage == "all") {
    results[[paste0(pool_name, "_arithmetic")]] <-
      crossfit_arithmetic(components)
  }
  results[[paste0(pool_name, "_logpool")]] <-
    nested_log_pool(components)
  if (stack_stage == "all") {
    results[[paste0(pool_name, "_xgbmeta")]] <-
      nested_meta_xgb(components)
  }
}

summary <- do.call(rbind, lapply(names(results), function(name) {
  method <- sub("^.*_", "", name)
  pool_name <- sub(paste0("_", method, "$"), "", name)
  data.frame(
    candidate = name,
    pool = pool_name,
    method = method,
    n_components = length(pools[[pool_name]]),
    logloss = results[[name]]$logloss,
    gain_vs_current =
      log_loss_matrix(truth, current) -
      results[[name]]$logloss
  )
}))
summary <- summary[order(summary$logloss), , drop = FALSE]

learned_names <- summary$candidate[
  summary$method %in% c("logpool", "xgbmeta")
]
row_loss <- function(prediction) {
  -rowSums(
    truth * log(pmax(prediction / rowSums(prediction), 1e-15))
  )
}
case_gain <- do.call(cbind, lapply(learned_names, function(name) {
  unname(tapply(
    row_loss(current) - row_loss(results[[name]]$prediction),
    train$Case,
    mean
  ))
}))
colnames(case_gain) <- learned_names
set.seed(4821L)
n_boot <- 100000L
n_case <- nrow(case_gain)
bootstrap <- matrix(
  NA_real_, n_boot, ncol(case_gain),
  dimnames = list(NULL, colnames(case_gain))
)
for (start in seq.int(1L, n_boot, by = 1000L)) {
  stop_at <- min(n_boot, start + 999L)
  n_this <- stop_at - start + 1L
  sampled <- matrix(
    sample.int(n_case, n_case * n_this, replace = TRUE),
    nrow = n_case
  )
  for (index in seq_len(ncol(case_gain))) {
    bootstrap[start:stop_at, index] <- colMeans(matrix(
      case_gain[sampled, index],
      nrow = n_case
    ))
  }
}
family_size <- length(learned_names)
family_alpha <- 0.05 / family_size
bootstrap_summary <- do.call(rbind, lapply(
  seq_along(learned_names),
  function(index) {
    value <- bootstrap[, index]
    data.frame(
      candidate = learned_names[[index]],
      point_gain = mean(case_gain[, index]),
      bootstrap_sd = sd(value),
      lower_95 = unname(quantile(value, 0.025)),
      upper_95 = unname(quantile(value, 0.975)),
      lower_99 = unname(quantile(value, 0.005)),
      upper_99 = unname(quantile(value, 0.995)),
      lower_bonferroni = unname(
        quantile(value, family_alpha / 2)
      ),
      upper_bonferroni = unname(
        quantile(value, 1 - family_alpha / 2)
      ),
      win_rate = mean(value > 0),
      n_boot = n_boot,
      family_size = family_size
    )
  }
))

write.csv(
  summary,
  file.path(output_dir, paste0(output_stem, "_summary.csv")),
  row.names = FALSE
)
write.csv(
  bootstrap_summary,
  file.path(output_dir, paste0(output_stem, "_bootstrap.csv")),
  row.names = FALSE
)
write.csv(
  do.call(rbind, lapply(names(results), function(name) {
    result <- results[[name]]
    if (is.null(result$selection)) return(NULL)
    selection <- result$selection
    hyperparameter <- if ("lambda" %in% names(selection)) {
      paste0("lambda=", selection$lambda)
    } else {
      paste0("config=", selection$config)
    }
    data.frame(
      candidate = name,
      outer_fold = selection$outer_fold,
      hyperparameter = hyperparameter,
      inner_logloss = selection$inner_logloss,
      selected = selection$selected
    )
  })),
  file.path(output_dir, paste0(output_stem, "_selection.csv")),
  row.names = FALSE
)
write.csv(
  do.call(rbind, lapply(names(results), function(name) {
    result <- results[[name]]
    if (!is.null(result$beta)) {
      return(data.frame(
        candidate = name,
        outer_fold = result$beta$outer_fold,
        component = result$beta$component,
        coefficient_type = "log_pool_beta",
        value = result$beta$beta,
        selected_lambda = result$beta$selected_lambda
      ))
    }
    if (!is.null(result$weights)) {
      return(data.frame(
        candidate = name,
        outer_fold = result$weights$outer_fold,
        component = result$weights$component,
        coefficient_type = "arithmetic_weight",
        value = result$weights$weight,
        selected_lambda = NA_real_
      ))
    }
    NULL
  })),
  file.path(output_dir, paste0(output_stem, "_coefficients.csv")),
  row.names = FALSE
)
saveRDS(
  list(
    pools = pools,
    component_losses = vapply(
      all_components,
      function(x) log_loss_matrix(truth, x),
      numeric(1)
    ),
    current = current,
    results = results,
    summary = summary,
    bootstrap_summary = bootstrap_summary,
    bootstrap = bootstrap,
    case_gain = case_gain
  ),
  file.path(output_dir, paste0(output_stem, ".rds"))
)

print(summary, digits = 9)
print(bootstrap_summary, digits = 9)
