# Shared, frozen machinery for repeated respondent-grouped CV.

options(stringsAsFactors = FALSE)

old_codex_stage <- Sys.getenv("CODEX_STAGE", unset = NA_character_)
old_deep_stage <- Sys.getenv("CODEX_DEEP_STAGE", unset = NA_character_)
old_history_stage <- Sys.getenv(
  "CODEX_HISTORY_PRIOR_STAGE", unset = NA_character_
)
Sys.setenv(
  CODEX_STAGE = "define",
  CODEX_DEEP_STAGE = "define",
  CODEX_HISTORY_PRIOR_STAGE = "define"
)
source("R/codex_history_prior_smoothing.R")
source("R/codex_triple_common.R")
source("R/codex_rank_xgb.R")
source("R/codex_xgb_retune.R")
source("R/codex_glmnet_cox_ensemble.R")
source("R/codex_torch_deep_mlp.R")
if (is.na(old_codex_stage)) {
  Sys.unsetenv("CODEX_STAGE")
} else {
  Sys.setenv(CODEX_STAGE = old_codex_stage)
}
if (is.na(old_deep_stage)) {
  Sys.unsetenv("CODEX_DEEP_STAGE")
} else {
  Sys.setenv(CODEX_DEEP_STAGE = old_deep_stage)
}
if (is.na(old_history_stage)) {
  Sys.unsetenv("CODEX_HISTORY_PRIOR_STAGE")
} else {
  Sys.setenv(CODEX_HISTORY_PRIOR_STAGE = old_history_stage)
}

repeat_output_dir <- "data_processed/codex_repeat_cv"
repeat_checkpoint_dir <- file.path(
  repeat_output_dir, "checkpoints"
)
dir.create(
  repeat_checkpoint_dir, recursive = TRUE, showWarnings = FALSE
)

repeat_additional_seeds <- c(
  1907L, 2719L, 6151L, 8293L, 104729L
)
repeat_all_seeds <- c(4821L, repeat_additional_seeds)
repeat_component_names <- c(
  "mlogit", "original_xgb", "rank_ndcg", "retuned_xgb",
  "cox", "shallow_mlp", "triple_mlogit", "deep_mlp"
)

repeat_rank_spec <- readRDS(
  "data_processed/codex/rank_oof.rds"
)[[1]]$spec
repeat_retuned_spec <- readRDS(
  "data_processed/codex/xgb_retune_oof.rds"
)[[1]]$spec
repeat_deep_spec <- readRDS(
  "data_processed/codex_deep_stack/torch_deep_oof.rds"
)$specification

repeat_validate_prediction <- function(prediction, n_rows = NULL) {
  prediction <- as.matrix(prediction)
  if (!is.null(n_rows)) {
    stopifnot(identical(dim(prediction), c(n_rows, 4L)))
  } else {
    stopifnot(ncol(prediction) == 4L)
  }
  stopifnot(
    !anyNA(prediction),
    all(is.finite(prediction)),
    all(prediction > 0),
    max(abs(rowSums(prediction) - 1)) < 1e-6
  )
  prediction / rowSums(prediction)
}

repeat_fold_map <- function(cases, seed) {
  cases <- as.integer(cases)
  set.seed(seed)
  fold <- sample(rep(1:5, length.out = length(cases)))
  names(fold) <- as.character(cases)
  stopifnot(
    length(fold) == length(cases),
    all(table(fold) %in% c(227L))
  )
  fold
}

repeat_row_loss <- function(truth, prediction) {
  prediction <- repeat_validate_prediction(
    prediction, nrow(truth)
  )
  -rowSums(
    truth * log(pmax(prediction, 1e-15))
  )
}

repeat_fit_original_xgb <- function(train_wide) {
  y <- max.col(
    train_wide[, paste0("Ch", 1:4), drop = FALSE]
  ) - 1L
  xgb.train(
    params = list(
      objective = "multi:softprob",
      num_class = 4L,
      eval_metric = "mlogloss",
      eta = 0.1,
      max_depth = 4L,
      subsample = 0.8,
      colsample_bytree = 0.8,
      tree_method = "hist",
      seed = 4821L,
      nthread = 1L
    ),
    data = xgb.DMatrix(
      wide_feature_matrix(train_wide), label = y
    ),
    nrounds = 73L,
    verbose = 0
  )
}

repeat_predict_original_xgb <- function(model, target_wide) {
  repeat_validate_prediction(predict(
    model,
    xgb.DMatrix(wide_feature_matrix(target_wide))
  ), nrow(target_wide))
}

repeat_fit_rank <- function(train_long) {
  params <- rank_params(
    repeat_rank_spec$objective,
    repeat_rank_spec$eta,
    repeat_rank_spec$depth
  )
  params$seed <- 4821L
  params$nthread <- 1L
  fit_ranker(
    train_long, params,
    nrounds = as.integer(repeat_rank_spec$nrounds)
  )
}

repeat_rank_margin_matrix <- function(model, target_long) {
  target_long <- sort_long_tasks(target_long)
  margin <- predict_ranker_margin(model, target_long)
  stopifnot(length(margin) == nrow(target_long))
  matrix(margin, ncol = 4L, byrow = TRUE)
}

repeat_calibrate_rank <- function(margin_oof, truth, row_fold) {
  stopifnot(
    identical(dim(margin_oof), dim(truth)),
    !anyNA(margin_oof)
  )
  prediction <- matrix(NA_real_, nrow(truth), 4L)
  scale <- numeric(5L)
  for (fold in 1:5) {
    fit_rows <- row_fold != fold
    validation_rows <- row_fold == fold
    scale[[fold]] <- best_margin_scale(
      as.vector(t(
        margin_oof[fit_rows, , drop = FALSE]
      )),
      truth[fit_rows, , drop = FALSE]
    )$scale
    prediction[validation_rows, ] <- softmax_margins(
      as.vector(t(
        margin_oof[validation_rows, , drop = FALSE]
      )),
      scale[[fold]]
    )
  }
  list(
    prediction = repeat_validate_prediction(
      prediction, nrow(truth)
    ),
    scales = scale
  )
}

repeat_fit_retuned_xgb <- function(train_wide) {
  spec <- repeat_retuned_spec
  spec$seed <- NULL
  fit_multiclass(train_wide, spec)
}

repeat_fit_arithmetic <- function(truth, components, rows) {
  n_components <- length(components)
  stopifnot(n_components >= 2L)
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
    control = list(maxit = 3000, reltol = 1e-10)
  )
  if (fitted$convergence != 0L) {
    stop(
      "arithmetic optimizer failed: ",
      fitted$convergence, " ", fitted$message
    )
  }
  weights <- softmax_weights(fitted$par, n_components)
  names(weights) <- names(components)
  weights
}

repeat_crossfit_arithmetic <- function(truth, components, row_fold) {
  stopifnot(identical(names(components), repeat_component_names))
  prediction <- matrix(NA_real_, nrow(truth), 4L)
  weights <- matrix(
    NA_real_, 5L, length(components),
    dimnames = list(
      paste0("fold", 1:5), names(components)
    )
  )
  for (fold in 1:5) {
    fit_rows <- which(row_fold != fold)
    validation_rows <- row_fold == fold
    weights[fold, ] <- repeat_fit_arithmetic(
      truth, components, fit_rows
    )
    prediction[validation_rows, ] <- Reduce(
      `+`,
      Map(
        function(component, weight) {
          component[validation_rows, , drop = FALSE] *
            weight
        },
        components, weights[fold, ]
      )
    )
  }
  list(
    prediction = repeat_validate_prediction(
      prediction, nrow(truth)
    ),
    weights = weights,
    logloss = log_loss_matrix(truth, prediction)
  )
}

repeat_crossfit_current <- function(
    truth, mlogit, original_xgb, shallow_mlp, row_fold) {
  v11 <- 0.8 * mlogit + 0.2 * original_xgb
  weight_grid <- seq(0, 0.30, by = 0.01)
  prediction <- matrix(NA_real_, nrow(truth), 4L)
  weights <- numeric(5L)
  for (fold in 1:5) {
    fit_rows <- row_fold != fold
    validation_rows <- row_fold == fold
    loss <- vapply(weight_grid, function(weight) {
      log_loss_matrix(
        truth[fit_rows, , drop = FALSE],
        (1 - weight) *
          v11[fit_rows, , drop = FALSE] +
          weight *
          shallow_mlp[fit_rows, , drop = FALSE]
      )
    }, numeric(1))
    weights[[fold]] <- weight_grid[[which.min(loss)]]
    prediction[validation_rows, ] <-
      (1 - weights[[fold]]) *
        v11[validation_rows, , drop = FALSE] +
      weights[[fold]] *
        shallow_mlp[validation_rows, , drop = FALSE]
  }
  list(
    prediction = repeat_validate_prediction(
      prediction, nrow(truth)
    ),
    weights = weights,
    logloss = log_loss_matrix(truth, prediction)
  )
}

repeat_fixed_history_predictions <- function(
    mlogit, history_mlogit, original_xgb, shallow_mlp) {
  baseline <- 0.85 * (
    0.8 * mlogit + 0.2 * original_xgb
  ) + 0.15 * shallow_mlp
  candidate <- 0.85 * (
    0.8 * history_mlogit + 0.2 * original_xgb
  ) + 0.15 * shallow_mlp
  list(
    baseline = repeat_validate_prediction(
      baseline, nrow(baseline)
    ),
    candidate = repeat_validate_prediction(
      candidate, nrow(candidate)
    )
  )
}

repeat_case_gain <- function(
    truth, baseline, candidate, case) {
  row_gain <- repeat_row_loss(truth, baseline) -
    repeat_row_loss(truth, candidate)
  unname(tapply(row_gain, case, mean))
}

repeat_bootstrap_average_gain <- function(
    case_gain_matrix, family_size,
    n_boot = 100000L, seed = 4821L) {
  stopifnot(
    nrow(case_gain_matrix) == 1135L,
    ncol(case_gain_matrix) == 6L,
    !anyNA(case_gain_matrix)
  )
  case_gain <- rowMeans(case_gain_matrix)
  set.seed(seed)
  bootstrap <- numeric(n_boot)
  n_case <- length(case_gain)
  for (start in seq.int(1L, n_boot, by = 1000L)) {
    stop_at <- min(n_boot, start + 999L)
    n_this <- stop_at - start + 1L
    sampled <- matrix(
      sample.int(
        n_case, n_case * n_this, replace = TRUE
      ),
      nrow = n_case
    )
    bootstrap[start:stop_at] <- colMeans(matrix(
      case_gain[sampled], nrow = n_case
    ))
  }
  family_alpha <- 0.05 / family_size
  list(
    summary = data.frame(
      point_gain = mean(case_gain),
      bootstrap_mean = mean(bootstrap),
      bootstrap_sd = sd(bootstrap),
      lower_95 = unname(quantile(bootstrap, 0.025)),
      upper_95 = unname(quantile(bootstrap, 0.975)),
      lower_99 = unname(quantile(bootstrap, 0.005)),
      upper_99 = unname(quantile(bootstrap, 0.995)),
      lower_family = unname(quantile(
        bootstrap, family_alpha / 2
      )),
      upper_family = unname(quantile(
        bootstrap, 1 - family_alpha / 2
      )),
      win_rate = mean(bootstrap > 0),
      n_boot = n_boot,
      family_size = family_size
    ),
    case_gain = case_gain,
    bootstrap = bootstrap
  )
}

joint_history_triple_formula <- function() {
  base <- m8trpg_formula("none")
  base_rhs <- sub("\\| 0$", "", as.character(base)[3])
  out <- as.formula(paste(
    "chosen ~", base_rhs,
    "+ P_income_miles",
    "+ hist_prior_price_gap_k3",
    "+ hist_prior_attr_familiarity_k3",
    "| 0"
  ))
  environment(out) <- environment()
  out
}

fit_predict_joint_history_triple <- function(
    train_long, valid_long) {
  scaler <- choice_scaler(train_long)
  tr_feat <- make_m8trpg_features(
    train_long, scaler$ctr, scaler$scl, "none"
  )
  va_feat <- make_m8trpg_features(
    valid_long, scaler$ctr, scaler$scl, "none"
  )
  tr_feat <- add_candidate_features(
    tr_feat, scaler$ctr, scaler$scl,
    "triple_price_income_miles"
  )
  va_feat <- add_candidate_features(
    va_feat, scaler$ctr, scaler$scl,
    "triple_price_income_miles"
  )
  formula <- joint_history_triple_formula()
  model <- mlogit(
    formula,
    data = tr_feat,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen"
  )
  mdat_valid <- dfidx(
    va_feat,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen"
  )
  class(mdat_valid) <- c(
    "dfidx_mlogit", class(mdat_valid)
  )
  mf_valid <- model.frame(
    mdat_valid, formula, balanced = TRUE
  )
  x_valid <- model.matrix(mf_valid, rhs = 1:3)
  beta <- coef(model)
  stopifnot(all(names(beta) %in% colnames(x_valid)))
  eta <- as.numeric(
    x_valid[, names(beta), drop = FALSE] %*% beta
  )
  valid_map <- unique(
    va_feat[, c("chid", "No"), drop = FALSE]
  )
  valid_map <- valid_map[
    order(valid_map$No), , drop = FALSE
  ]
  eta_chid <- as.character(dfidx::idx(mf_valid, 1))
  eta_alt <- as.integer(
    as.character(dfidx::idx(mf_valid, 2))
  )
  eta_order <- order(
    match(eta_chid, valid_map$chid), eta_alt
  )
  stopifnot(
    identical(
      eta_chid[eta_order],
      rep(valid_map$chid, each = 4)
    ),
    identical(
      eta_alt[eta_order],
      rep(1:4, times = nrow(valid_map))
    )
  )
  prediction <- softmax_margins(eta[eta_order])
  rownames(prediction) <- valid_map$chid
  extra_names <- c(
    "P_income_miles",
    "hist_prior_price_gap_k3",
    "hist_prior_attr_familiarity_k3"
  )
  stopifnot(all(extra_names %in% names(beta)))
  list(
    pred = repeat_validate_prediction(
      prediction, nrow(valid_map)
    ),
    no = valid_map$No,
    model = model,
    extra_coefficient = beta[extra_names]
  )
}
