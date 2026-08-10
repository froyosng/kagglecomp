# Diagnostic: allow the CV-confirmed MLP to compete for weight alongside the
# existing OOF component families, rather than only replacing 15% of the fixed
# ensemble_v11 prediction. This is post-hoc ensemble analysis; the simpler
# fixed-v11 blend remains the primary pre-specified comparison.

source("R/codex_modeling_common.R")

output_dir <- "data_processed/codex_behavioral_round"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

train <- read.csv("csv files/train.csv")
truth <- as.matrix(train[, paste0("Ch", 1:4)])
fold_map <- canonical_fold_map()
row_fold <- unname(fold_map[as.character(train$Case)])

base <- readRDS("data_processed/oof_ensemble_v10.rds")
rank <- readRDS("data_processed/codex/rank_oof.rds")[[1]]$pred
tuned <- readRDS(
  "data_processed/codex/xgb_retune_oof.rds"
)[[1]]$pred
cox <- readRDS("data_processed/codex/cox_oof.rds")$oof_min
mlp <- readRDS(file.path(output_dir, "mlp_oof.rds"))$oof[[1]]

components <- list(
  mlogit = base$oof_mlogit,
  original_xgb = base$oof_xgb,
  rank_ndcg = rank,
  retuned_xgb = tuned,
  cox_min = cox,
  mlp = mlp
)
stopifnot(
  all(vapply(
    components,
    function(pred) {
      is.matrix(pred) &&
        identical(dim(pred), dim(truth)) &&
        !anyNA(pred) &&
        max(abs(rowSums(pred) - 1)) < 1e-6
    },
    logical(1)
  ))
)

softmax_weights <- function(theta, k) {
  z <- c(theta, 0)
  z <- z - max(z)
  exp(z) / sum(exp(z))
}

blend_components <- function(component_list, weights) {
  Reduce(
    `+`,
    Map(function(pred, weight) pred * weight,
        component_list, weights)
  )
}

fit_weights <- function(component_list, rows) {
  k <- length(component_list)
  chosen_probability <- do.call(cbind, lapply(
    component_list,
    function(pred) {
      rowSums(
        truth[rows, , drop = FALSE] *
          pred[rows, , drop = FALSE]
      )
    }
  ))
  evaluate <- function(theta) {
    weights <- softmax_weights(theta, k)
    mixed_probability <- as.numeric(
      chosen_probability %*% weights
    )
    value <- -mean(log(pmax(mixed_probability, 1e-15)))
    gradient_weight <- -colMeans(
      chosen_probability / mixed_probability
    )
    weighted_gradient <- sum(weights * gradient_weight)
    gradient_theta <- weights[seq_len(k - 1L)] *
      (
        gradient_weight[seq_len(k - 1L)] -
          weighted_gradient
      )
    list(value = value, gradient = gradient_theta)
  }
  objective <- function(theta) evaluate(theta)$value
  gradient <- function(theta) evaluate(theta)$gradient
  start <- c(log(4), rep(0, max(0, k - 2L)))
  fitted <- optim(
    start,
    objective,
    gr = gradient,
    method = "BFGS",
    control = list(maxit = 1000, reltol = 1e-10)
  )
  stopifnot(fitted$convergence == 0L)
  list(
    weights = softmax_weights(fitted$par, k),
    logloss = fitted$value
  )
}

evaluate_pool <- function(pool_names) {
  pool <- components[pool_names]
  global <- fit_weights(pool, seq_len(nrow(train)))
  global_pred <- blend_components(pool, global$weights)
  crossfit_pred <- matrix(NA_real_, nrow(train), 4L)
  fold_weights <- matrix(
    NA_real_,
    nrow = 5L,
    ncol = length(pool_names),
    dimnames = list(paste0("fold", 1:5), pool_names)
  )
  for (fold in 1:5) {
    fit_rows <- row_fold != fold
    validation_rows <- row_fold == fold
    fitted <- fit_weights(pool, which(fit_rows))
    fold_weights[fold, ] <- fitted$weights
    crossfit_pred[validation_rows, ] <- blend_components(
      lapply(
        pool,
        function(x) x[validation_rows, , drop = FALSE]
      ),
      fitted$weights
    )
  }
  stopifnot(!anyNA(crossfit_pred))
  list(
    pool = paste(pool_names, collapse = "+"),
    global_weights = global$weights,
    global_logloss = log_loss_matrix(truth, global_pred),
    crossfit_logloss = log_loss_matrix(truth, crossfit_pred),
    global_pred = global_pred,
    crossfit_pred = crossfit_pred,
    fold_weights = fold_weights
  )
}

pools <- list(
  c("mlogit", "original_xgb", "mlp"),
  c("mlogit", "retuned_xgb", "cox_min", "mlp"),
  c("mlogit", "rank_ndcg", "retuned_xgb", "mlp"),
  c("mlogit", "rank_ndcg", "retuned_xgb", "cox_min", "mlp"),
  c(
    "mlogit", "original_xgb", "rank_ndcg",
    "retuned_xgb", "cox_min", "mlp"
  )
)
evaluated <- lapply(pools, evaluate_pool)
summary <- do.call(rbind, lapply(evaluated, function(x) {
  data.frame(
    pool = x$pool,
    global_logloss = x$global_logloss,
    crossfit_logloss = x$crossfit_logloss,
    global_weights =
      paste(sprintf("%.5f", x$global_weights), collapse = "/")
  )
}))
summary <- summary[order(summary$crossfit_logloss), ]

best <- evaluated[[match(
  summary$pool[[1]],
  vapply(evaluated, `[[`, character(1), "pool")
)]]
v11 <- 0.8 * components$mlogit + 0.2 * components$original_xgb
fixed_mlp <- readRDS(
  file.path(output_dir, "mlp_precision.rds")
)$crossfit_prediction

row_loss <- function(pred) {
  pred <- pmin(pmax(pred, 1e-15), 1 - 1e-15)
  -rowSums(truth * log(pred))
}
case_gains <- cbind(
  versus_v11 = unname(tapply(
    row_loss(v11) - row_loss(best$crossfit_pred),
    train$Case, mean
  )),
  versus_fixed_mlp = unname(tapply(
    row_loss(fixed_mlp) - row_loss(best$crossfit_pred),
    train$Case, mean
  ))
)

set.seed(4821)
n_boot <- 10000L
n_case <- nrow(case_gains)
boot <- matrix(NA_real_, n_boot, ncol(case_gains))
colnames(boot) <- colnames(case_gains)
for (index in seq_len(n_boot)) {
  sampled <- sample.int(n_case, n_case, replace = TRUE)
  boot[index, ] <- colMeans(case_gains[sampled, , drop = FALSE])
}
bootstrap <- data.frame(
  comparison = colnames(boot),
  point_gain = colMeans(case_gains),
  bootstrap_mean = colMeans(boot),
  bootstrap_sd = apply(boot, 2, sd),
  lower_95 = apply(boot, 2, quantile, 0.025),
  upper_95 = apply(boot, 2, quantile, 0.975),
  win_rate = colMeans(boot > 0)
)

fold_weight_rows <- do.call(rbind, lapply(evaluated, function(x) {
  do.call(rbind, lapply(1:5, function(fold) {
    data.frame(
      pool = x$pool,
      fold = fold,
      component = colnames(x$fold_weights),
      weight = as.numeric(x$fold_weights[fold, ])
    )
  }))
}))

write_result_csv(
  summary,
  file.path(output_dir, "mlp_extended_ensemble.csv")
)
write_result_csv(
  fold_weight_rows,
  file.path(output_dir, "mlp_extended_ensemble_weights.csv")
)
write_result_csv(
  bootstrap,
  file.path(output_dir, "mlp_extended_ensemble_bootstrap.csv")
)
saveRDS(
  list(
    summary = summary,
    evaluated = evaluated,
    best = best,
    bootstrap = bootstrap
  ),
  file.path(output_dir, "mlp_extended_ensemble.rds")
)

print(summary, digits = 9)
print(bootstrap, digits = 9)
print(best$fold_weights, digits = 6)
