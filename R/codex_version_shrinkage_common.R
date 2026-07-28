## Shared helpers for questionnaire-version residual calibration.
##
## The version IDs are loaded from the already-computed fingerprint artifact.
## This script deliberately does not reconstruct fingerprints from the designs.

suppressPackageStartupMessages({
  library(nnet)
  library(mlogit)
  library(dfidx)
  library(xgboost)
})

source("R/codex_shared_utility_common.R")

old_codex_stage <- Sys.getenv("CODEX_STAGE", unset = NA_character_)
Sys.setenv(CODEX_STAGE = "define")
source("R/codex_mlp_ensemble.R")
if (is.na(old_codex_stage)) {
  Sys.unsetenv("CODEX_STAGE")
} else {
  Sys.setenv(CODEX_STAGE = old_codex_stage)
}

version_output_dir <- "data_processed/codex_version_shrinkage"
dir.create(version_output_dir, recursive = TRUE, showWarnings = FALSE)

## Keep the literal mean-residual shrinkage grid requested for the primary
## experiment alongside the separate Newton/curvature grid.
version_alpha_grid <- c(20, 40, 60, 80, 100, Inf)
version_lambda_grid <- c(5, 10, 20, 40, 80, 160, 320, Inf)

truth_wide <- function(wide) {
  wide <- wide[order(wide$No), , drop = FALSE]
  truth <- as.matrix(wide[, paste0("Ch", 1:4), drop = FALSE])
  storage.mode(truth) <- "double"
  stopifnot(all(rowSums(truth) == 1))
  truth
}

normalize_probability <- function(prediction) {
  prediction <- as.matrix(prediction)
  prediction <- prediction / rowSums(prediction)
  stopifnot(
    ncol(prediction) == 4L,
    all(is.finite(prediction)),
    all(prediction > 0)
  )
  prediction
}

load_questionnaire_versions <- function() {
  artifact <- readRDS("data_processed/questionnaire_fingerprints.rds")
  stopifnot(
    is.list(artifact),
    identical(as.integer(artifact$n_unique), 299L),
    length(artifact$case_ids) == length(artifact$fingerprints),
    !anyDuplicated(artifact$case_ids)
  )
  version_id <- match(
    artifact$fingerprints,
    unique(artifact$fingerprints)
  )
  mapping <- data.frame(
    Case = as.integer(artifact$case_ids),
    version_id = as.integer(version_id),
    stringsAsFactors = FALSE
  )
  stopifnot(
    nrow(mapping) == 1398L,
    length(unique(mapping$version_id)) == 299L
  )
  mapping
}

case_version <- function(case, version_map) {
  out <- version_map$version_id[
    match(as.integer(case), version_map$Case)
  ]
  stopifnot(!anyNA(out))
  out
}

current_fixed_oof <- function(train_wide) {
  train_wide <- train_wide[order(train_wide$No), , drop = FALSE]
  ensemble <- readRDS("data_processed/oof_ensemble_v10.rds")
  mlp <- readRDS(
    "data_processed/codex_behavioral_round/mlp_oof.rds"
  )
  mlp_prediction <- mlp$oof[["h08_d0.100"]]
  stopifnot(
    identical(dim(ensemble$oof_mlogit), c(nrow(train_wide), 4L)),
    identical(dim(mlp_prediction), c(nrow(train_wide), 4L)),
    max(abs(ensemble$oof_truth - truth_wide(train_wide))) < 1e-12
  )
  v11 <- 0.8 * ensemble$oof_mlogit + 0.2 * ensemble$oof_xgb
  current <- 0.85 * v11 + 0.15 * mlp_prediction
  current <- normalize_probability(current)
  stopifnot(
    abs(
      log_loss_matrix(truth_wide(train_wide), current) -
        1.143686618134879
    ) < 1e-10
  )
  list(
    pred = current,
    fold_of_case = ensemble$fold_of_case,
    no = as.integer(train_wide$No)
  )
}

fit_submitted_architecture <- function(source_wide, target_wide,
                                       reference,
                                       seed_offset = 0L) {
  source_wide <- source_wide[order(source_wide$No), , drop = FALSE]
  target_wide <- target_wide[order(target_wide$No), , drop = FALSE]
  stopifnot(
    nrow(source_wide) > 0L,
    nrow(target_wide) > 0L,
    !any(source_wide$Case %in% target_wide$Case)
  )
  started <- proc.time()[["elapsed"]]

  source_long <- reshape_choice_long(source_wide)
  target_long <- reshape_choice_long(target_wide)
  mlogit_fit <- fit_m8trpg_model(source_long)
  source_mlogit_result <- predict_m8trpg_margin(
    mlogit_fit, source_long
  )
  target_mlogit_result <- predict_m8trpg_margin(
    mlogit_fit, target_long
  )
  source_mlogit <- aligned_prediction(
    source_mlogit_result, source_wide$No
  )
  target_mlogit <- aligned_prediction(
    target_mlogit_result, target_wide$No
  )

  source_label <- max.col(
    source_wide[, paste0("Ch", 1:4), drop = FALSE]
  ) - 1L
  xgb_fit <- xgb.train(
    params = list(
      objective = "multi:softprob",
      num_class = 4L,
      eval_metric = "mlogloss",
      eta = 0.1,
      max_depth = 4L,
      subsample = 0.8,
      colsample_bytree = 0.8,
      seed = as.integer(4821L + seed_offset),
      nthread = 1L
    ),
    data = xgb.DMatrix(
      wide_feature_matrix(source_wide),
      label = source_label,
      nthread = 1L
    ),
    nrounds = 73L,
    verbose = 0
  )
  source_xgb <- as.matrix(predict(
    xgb_fit,
    xgb.DMatrix(wide_feature_matrix(source_wide), nthread = 1L)
  ))
  target_xgb <- as.matrix(predict(
    xgb_fit,
    xgb.DMatrix(wide_feature_matrix(target_wide), nthread = 1L)
  ))

  scaler <- continuous_scaler(source_wide)
  source_matrix <- build_mlp_matrix(
    source_wide, reference, scaler
  )
  target_matrix <- build_mlp_matrix(
    target_wide, reference, scaler,
    keep_columns = source_matrix$keep_columns
  )
  target_start <- nrow(source_wide) + 1L
  mlp_fit <- fit_mlp_average(
    source_matrix$x,
    truth_wide(source_wide),
    rbind(source_matrix$x, target_matrix$x),
    size = 8L,
    decay = 0.1,
    seeds = 4821L + seed_offset + 0:4,
    max_iterations = 200L
  )
  source_mlp <- mlp_fit$pred[seq_len(nrow(source_wide)), , drop = FALSE]
  target_mlp <- mlp_fit$pred[
    target_start:(target_start + nrow(target_wide) - 1L),
    ,
    drop = FALSE
  ]

  source_v11 <- 0.8 * source_mlogit + 0.2 * source_xgb
  target_v11 <- 0.8 * target_mlogit + 0.2 * target_xgb
  source_current <- normalize_probability(
    0.85 * source_v11 + 0.15 * source_mlp
  )
  target_current <- normalize_probability(
    0.85 * target_v11 + 0.15 * target_mlp
  )

  elapsed <- proc.time()[["elapsed"]] - started
  list(
    source_pred = source_current,
    target_pred = target_current,
    source_no = as.integer(source_wide$No),
    target_no = as.integer(target_wide$No),
    source_case = as.integer(source_wide$Case),
    target_case = as.integer(target_wide$Case),
    source_loss = log_loss_matrix(
      truth_wide(source_wide), source_current
    ),
    target_loss = log_loss_matrix(
      truth_wide(target_wide), target_current
    ),
    mlp_fits = mlp_fit$fits,
    elapsed_seconds = elapsed,
    seed_offset = as.integer(seed_offset)
  )
}

newton_components <- function(wide, prediction, version_map) {
  wide <- wide[order(wide$No), , drop = FALSE]
  prediction <- normalize_probability(prediction)
  stopifnot(nrow(wide) == nrow(prediction))
  data.frame(
    Case = as.integer(wide$Case),
    No = as.integer(wide$No),
    Task = as.integer(wide$Task),
    version_id = case_version(wide$Case, version_map),
    y4 = truth_wide(wide)[, 4],
    p4 = prediction[, 4],
    g = prediction[, 4] - truth_wide(wide)[, 4],
    h = prediction[, 4] * (1 - prediction[, 4])
  )
}

estimate_newton_delta <- function(source_wide, source_prediction,
                                  version_map, lambda) {
  component <- newton_components(
    source_wide, source_prediction, version_map
  )
  by_version <- split(seq_len(nrow(component)), component$version_id)
  result <- do.call(rbind, lapply(names(by_version), function(key) {
    index <- by_version[[key]]
    case_index <- split(index, component$Case[index])
    respondent_g <- vapply(
      case_index, function(i) sum(component$g[i]), numeric(1)
    )
    total_g <- sum(component$g[index])
    total_h <- sum(component$h[index])
    delta <- if (is.infinite(lambda)) {
      0
    } else {
      -total_g / (total_h + lambda)
    }
    denominator <- sum(abs(respondent_g))
    dominance <- if (denominator > 0) {
      max(abs(respondent_g)) / denominator
    } else {
      0
    }
    data.frame(
      version_id = as.integer(key),
      n_task = length(index),
      n_respondent = length(case_index),
      g = total_g,
      h = total_h,
      lambda = lambda,
      delta = delta,
      max_respondent_gradient_share = dominance,
      largest_abs_respondent_gradient = max(abs(respondent_g)),
      sum_abs_respondent_gradient = denominator
    )
  }))
  rownames(result) <- NULL
  result
}

check_optout_newton_derivatives <- function() {
  p4 <- c(0.08, 0.22, 0.41, 0.67, 0.31)
  y4 <- c(0, 1, 0, 1, 0)
  lambda <- 17
  objective <- function(delta) {
    denominator <- 1 - p4 + p4 * exp(delta)
    adjusted_p4 <- p4 * exp(delta) / denominator
    -sum(
      y4 * log(adjusted_p4) +
        (1 - y4) * log(1 - adjusted_p4)
    ) + 0.5 * lambda * delta^2
  }
  epsilon <- 1e-5
  numeric_gradient <- (
    objective(epsilon) - objective(-epsilon)
  ) / (2 * epsilon)
  numeric_hessian <- (
    objective(epsilon) - 2 * objective(0) + objective(-epsilon)
  ) / epsilon^2
  analytic_gradient <- sum(p4 - y4)
  analytic_hessian <- sum(p4 * (1 - p4)) + lambda
  data.frame(
    analytic_gradient = analytic_gradient,
    numeric_gradient = numeric_gradient,
    gradient_error = analytic_gradient - numeric_gradient,
    analytic_hessian = analytic_hessian,
    numeric_hessian = numeric_hessian,
    hessian_error = analytic_hessian - numeric_hessian,
    passed =
      abs(analytic_gradient - numeric_gradient) < 1e-8 &&
      abs(analytic_hessian - numeric_hessian) < 1e-4
  )
}

estimate_optout_delta <- function(source_wide, source_prediction,
                                  version_map, alpha) {
  source_wide <- source_wide[order(source_wide$No), , drop = FALSE]
  source_prediction <- normalize_probability(source_prediction)
  stopifnot(nrow(source_wide) == nrow(source_prediction))

  version <- case_version(source_wide$Case, version_map)
  residual <- truth_wide(source_wide)[, 4] - source_prediction[, 4]
  by_version <- split(seq_along(residual), version)
  result <- do.call(rbind, lapply(names(by_version), function(key) {
    index <- by_version[[key]]
    n_task <- length(index)
    shrinkage <- if (is.infinite(alpha)) {
      0
    } else {
      n_task / (n_task + alpha)
    }
    data.frame(
      version_id = as.integer(key),
      n_task = n_task,
      n_respondent = length(unique(source_wide$Case[index])),
      mean_residual = mean(residual[index]),
      shrinkage = shrinkage,
      delta = shrinkage * mean(residual[index])
    )
  }))
  rownames(result) <- NULL
  result
}

apply_optout_delta_vector <- function(prediction, delta) {
  prediction <- normalize_probability(prediction)
  stopifnot(length(delta) == nrow(prediction), !anyNA(delta))

  adjusted <- prediction
  multiplier <- exp(delta)
  denominator <- rowSums(prediction[, 1:3, drop = FALSE]) +
    prediction[, 4] * multiplier
  adjusted[, 1:3] <- prediction[, 1:3, drop = FALSE] / denominator
  adjusted[, 4] <- prediction[, 4] * multiplier / denominator
  adjusted <- normalize_probability(adjusted)
  list(
    pred = adjusted,
    delta = delta,
    zero_delta = delta == 0
  )
}

apply_optout_delta <- function(prediction, case, version_map,
                               delta_table) {
  version <- case_version(case, version_map)
  delta <- delta_table$delta[
    match(version, delta_table$version_id)
  ]
  delta[is.na(delta)] <- 0
  result <- apply_optout_delta_vector(prediction, delta)
  result$version_id <- version
  result
}

evaluate_alpha_grid <- function(source_wide, source_prediction,
                                target_wide, target_prediction,
                                version_map,
                                alpha_grid = version_alpha_grid) {
  target_wide <- target_wide[order(target_wide$No), , drop = FALSE]
  target_prediction <- normalize_probability(target_prediction)
  truth <- truth_wide(target_wide)
  baseline_loss <- log_loss_matrix(truth, target_prediction)
  adjusted <- vector("list", length(alpha_grid))
  rows <- vector("list", length(alpha_grid))

  for (index in seq_along(alpha_grid)) {
    alpha <- alpha_grid[[index]]
    delta_table <- estimate_optout_delta(
      source_wide, source_prediction, version_map, alpha
    )
    correction <- apply_optout_delta(
      target_prediction, target_wide$Case,
      version_map, delta_table
    )
    loss <- log_loss_matrix(truth, correction$pred)
    adjusted[[index]] <- correction$pred
    rows[[index]] <- data.frame(
      alpha = alpha,
      baseline_loss = baseline_loss,
      corrected_loss = loss,
      gain = baseline_loss - loss,
      mean_abs_delta = mean(abs(correction$delta)),
      max_abs_delta = max(abs(correction$delta)),
      zero_delta_rows = sum(correction$zero_delta),
      zero_delta_respondents = length(unique(
        target_wide$Case[correction$zero_delta]
      ))
    )
  }
  names(adjusted) <- ifelse(
    is.infinite(alpha_grid), "Inf", as.character(alpha_grid)
  )
  list(curve = do.call(rbind, rows), prediction = adjusted)
}

leave_respondent_out_newton <- function(wide, prediction,
                                        version_map, lambda) {
  wide <- wide[order(wide$No), , drop = FALSE]
  prediction <- normalize_probability(prediction)
  component <- newton_components(wide, prediction, version_map)

  version_g <- tapply(component$g, component$version_id, sum)
  version_h <- tapply(component$h, component$version_id, sum)
  version_n_case <- tapply(
    component$Case, component$version_id,
    function(x) length(unique(x))
  )
  case_key <- paste(component$version_id, component$Case, sep = "_")
  case_g <- tapply(component$g, case_key, sum)
  case_h <- tapply(component$h, case_key, sum)

  total_g <- as.numeric(version_g[
    match(as.character(component$version_id), names(version_g))
  ])
  total_h <- as.numeric(version_h[
    match(as.character(component$version_id), names(version_h))
  ])
  respondent_g <- as.numeric(case_g[match(case_key, names(case_g))])
  respondent_h <- as.numeric(case_h[match(case_key, names(case_h))])
  peer_count <- as.integer(version_n_case[
    match(as.character(component$version_id), names(version_n_case))
  ]) - 1L
  excluded_g <- total_g - respondent_g
  excluded_h <- total_h - respondent_h
  delta <- if (is.infinite(lambda)) {
    rep(0, nrow(component))
  } else {
    -excluded_g / (excluded_h + lambda)
  }
  delta[peer_count == 0L] <- 0
  correction <- apply_optout_delta_vector(prediction, delta)
  correction$version_id <- component$version_id
  correction$peer_count <- peer_count
  correction
}

evaluate_newton_lambda_grid <- function(wide, inner_oof_prediction,
                                        version_map,
                                        lambda_grid =
                                          version_lambda_grid) {
  wide <- wide[order(wide$No), , drop = FALSE]
  inner_oof_prediction <- normalize_probability(inner_oof_prediction)
  truth <- truth_wide(wide)
  baseline_loss <- log_loss_matrix(truth, inner_oof_prediction)
  adjusted <- vector("list", length(lambda_grid))
  rows <- vector("list", length(lambda_grid))
  for (index in seq_along(lambda_grid)) {
    lambda <- lambda_grid[[index]]
    correction <- leave_respondent_out_newton(
      wide, inner_oof_prediction, version_map, lambda
    )
    corrected_loss <- log_loss_matrix(truth, correction$pred)
    adjusted[[index]] <- correction$pred
    rows[[index]] <- data.frame(
      lambda = lambda,
      baseline_loss = baseline_loss,
      corrected_loss = corrected_loss,
      gain = baseline_loss - corrected_loss,
      mean_abs_delta = mean(abs(correction$delta)),
      max_abs_delta = max(abs(correction$delta)),
      zero_peer_respondents = length(unique(
        wide$Case[correction$peer_count == 0L]
      )),
      one_peer_respondents = length(unique(
        wide$Case[correction$peer_count == 1L]
      ))
    )
  }
  names(adjusted) <- ifelse(
    is.infinite(lambda_grid), "Inf", as.character(lambda_grid)
  )
  list(curve = do.call(rbind, rows), prediction = adjusted)
}

case_mean_gain <- function(truth, baseline, candidate, case) {
  baseline_row <- -rowSums(
    truth * log(pmin(pmax(normalize_probability(baseline), 1e-15), 1))
  )
  candidate_row <- -rowSums(
    truth * log(pmin(pmax(normalize_probability(candidate), 1e-15), 1))
  )
  tapply(baseline_row - candidate_row, case, mean)
}

bootstrap_case_gain <- function(truth, baseline, candidate, case,
                                replicates = 100000L,
                                seed = 4821L) {
  gain <- case_mean_gain(truth, baseline, candidate, case)
  set.seed(seed)
  bootstrap <- replicate(
    as.integer(replicates),
    mean(sample(gain, length(gain), replace = TRUE))
  )
  data.frame(
    point_gain = mean(gain),
    bootstrap_mean = mean(bootstrap),
    bootstrap_sd = sd(bootstrap),
    lower_95 = unname(quantile(bootstrap, 0.025)),
    upper_95 = unname(quantile(bootstrap, 0.975)),
    lower_99 = unname(quantile(bootstrap, 0.005)),
    upper_99 = unname(quantile(bootstrap, 0.995)),
    win_rate = mean(bootstrap > 0),
    replicates = as.integer(replicates)
  )
}
