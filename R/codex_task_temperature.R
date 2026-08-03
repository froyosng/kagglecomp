# Task-content-conditioned local temperature on frozen v14 utilities.
#
# Pre-registration:
#   codex_task_temperature_preregister.md
#
# Smoke test (1 outer fold, gradient check, no repeated CV):
#   Sys.setenv(TASK_TEMP_SMOKE = "1")
#   source("R/codex_task_temperature.R")
#   Sys.unsetenv("TASK_TEMP_SMOKE")
#
# Full run:
#   Sys.unsetenv("TASK_TEMP_SMOKE")
#   Sys.setenv(TASK_TEMP_REPEATED = "auto")
#   source("R/codex_task_temperature.R")

suppressPackageStartupMessages({
  library(numDeriv)
})

options(stringsAsFactors = FALSE)

experiment_id <- "task_temperature_v1"
output_dir <- file.path("data_processed", "codex_task_temperature")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

smoke_mode <- identical(Sys.getenv("TASK_TEMP_SMOKE", "0"), "1")
repeated_mode <- tolower(Sys.getenv("TASK_TEMP_REPEATED", "auto"))
stopifnot(repeated_mode %in% c("auto", "never", "always"))

canonical_seed <- 4821L
additional_seeds <- c(1907L, 2719L, 6151L, 8293L, 104729L)
all_outer_seeds <- c(canonical_seed, additional_seeds)
canonical_v14_target <- 1.1435331472708
near_miss_lower_limit <- -0.00075

bootstrap_replicates <- if (smoke_mode) {
  200L
} else {
  as.integer(Sys.getenv("TASK_TEMP_N_BOOT", "100000"))
}

v14_dir <- file.path("data_processed", "codex_set_context_network")
penalty_grid <- c(1.0, 0.1, 0.01, 0.001, 0.0)
attrs <- c(
  "CC", "GN", "NS", "BU", "FA", "LD", "BZ", "FC", "FP", "RP",
  "PP", "KA", "SC", "TS", "NV", "MA", "LB", "AF", "HU"
)

input_files <- c(
  file.path("csv files", "train.csv"),
  file.path(v14_dir, "canonical_result.rds")
)
missing_files <- input_files[!file.exists(input_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing required project file(s):\n  ",
    paste(missing_files, collapse = "\n  "),
    "\nRun this script from the kagglecomp repository root."
  )
}

## ---- generic helpers (same verified pattern as the other codex_* scripts) ----

validate_probability <- function(prediction, n_rows = NULL) {
  prediction <- as.matrix(prediction)
  if (!is.null(n_rows)) {
    stopifnot(identical(dim(prediction), c(as.integer(n_rows), 4L)))
  } else {
    stopifnot(ncol(prediction) == 4L)
  }
  stopifnot(!anyNA(prediction), all(is.finite(prediction)), min(prediction) >= -1e-12)
  prediction <- pmax(prediction, 1e-15)
  prediction <- prediction / rowSums(prediction)
  stopifnot(max(abs(rowSums(prediction) - 1)) < 1e-10)
  prediction
}

log_loss_matrix_local <- function(truth, prediction) {
  prediction <- validate_probability(prediction, nrow(truth))
  -mean(rowSums(as.matrix(truth) * log(pmax(prediction, 1e-15))))
}

row_log_loss_local <- function(truth, prediction) {
  prediction <- validate_probability(prediction, nrow(truth))
  -rowSums(as.matrix(truth) * log(pmax(prediction, 1e-15)))
}

respondent_gain <- function(truth, baseline, candidate, case) {
  row_gain <- row_log_loss_local(truth, baseline) - row_log_loss_local(truth, candidate)
  output <- tapply(row_gain, case, mean)
  output[order(as.integer(names(output)))]
}

bootstrap_case_means <- function(case_gain, replicates, seed = canonical_seed, chunk_size = 1000L) {
  case_gain <- as.numeric(case_gain)
  stopifnot(length(case_gain) > 1L, all(is.finite(case_gain)), replicates > 0L)
  set.seed(as.integer(seed))
  output <- numeric(replicates)
  start <- 1L
  while (start <= replicates) {
    count <- min(chunk_size, replicates - start + 1L)
    index <- matrix(
      sample.int(length(case_gain), length(case_gain) * count, replace = TRUE),
      nrow = length(case_gain), ncol = count
    )
    output[start:(start + count - 1L)] <- colMeans(matrix(
      case_gain[index], nrow = length(case_gain), ncol = count
    ))
    start <- start + count
  }
  output
}

summarize_case_gain <- function(case_gain, replicates, seed) {
  bootstrap <- bootstrap_case_means(case_gain, replicates, seed)
  list(
    summary = data.frame(
      point_gain = mean(case_gain),
      bootstrap_mean = mean(bootstrap),
      bootstrap_sd = sd(bootstrap),
      lower_95 = unname(quantile(bootstrap, 0.025)),
      upper_95 = unname(quantile(bootstrap, 0.975)),
      lower_99 = unname(quantile(bootstrap, 0.005)),
      upper_99 = unname(quantile(bootstrap, 0.995)),
      win_rate = mean(bootstrap > 0),
      n_boot = length(bootstrap)
    ),
    bootstrap = bootstrap
  )
}

bootstrap_summary <- function(truth, baseline, candidate, case, replicates, seed) {
  case_gain <- respondent_gain(truth, baseline, candidate, case)
  summarized <- summarize_case_gain(case_gain, replicates, seed)
  list(summary = summarized$summary, respondent_gain = case_gain, bootstrap = summarized$bootstrap)
}

## ---- task-difficulty features (model-free where possible; gap_top2 reuses
## v14's own frozen, honest, cross-fitted conditional-bundle probabilities,
## never the evaluation row's own truth) ----

compute_task_difficulty <- function(wide, v14_prediction) {
  inside <- v14_prediction[, 1:3, drop = FALSE]
  inside <- inside / rowSums(inside)
  sorted <- t(apply(inside, 1L, function(row) sort(row, decreasing = TRUE)))
  gap_top2 <- sorted[, 1L] - sorted[, 2L]

  price <- as.matrix(wide[, paste0("Price", 1:3), drop = FALSE])
  price_min <- apply(price, 1L, min)
  price_max <- apply(price, 1L, max)
  price_mean <- rowMeans(price)
  price_cv <- (price_max - price_min) / price_mean

  data.frame(gap_top2 = gap_top2, price_cv = price_cv)
}

fit_scaler <- function(values) {
  values <- as.matrix(values)
  centre <- colMeans(values)
  scale <- apply(values, 2L, sd)
  scale[!is.finite(scale) | scale == 0] <- 1
  list(centre = centre, scale = scale)
}

build_xdiff <- function(difficulty, scaler) {
  z <- sweep(sweep(as.matrix(difficulty), 2L, scaler$centre, "-"), 2L, scaler$scale, "/")
  cbind(intercept = 1, z)
}

## ---- the link itself ----
## msurp_ij = -log(p^v14_ij) >= 0. b_i = xdiff_i %*% theta is a per-task
## linear "local temperature exponent" (theta = 0 vector reproduces plain
## v14 softmax exactly, b_i = 0 for every task).

m_from_u <- function(u) pmax(-u, 1e-12)

softmax_rows <- function(eta) {
  eta <- eta - apply(eta, 1L, max)
  ez <- exp(eta)
  ez / rowSums(ez)
}

penalized_nll_temp <- function(theta, msurp, xdiff, truth, penalty) {
  b <- as.numeric(xdiff %*% theta)
  eta <- (-exp(b)) * msurp
  p <- softmax_rows(eta)
  nll <- -mean(rowSums(truth * log(pmax(p, 1e-15))))
  nll + penalty * sum(theta^2)
}

penalized_gradient_temp <- function(theta, msurp, xdiff, truth, penalty) {
  b <- as.numeric(xdiff %*% theta)
  eta <- (-exp(b)) * msurp
  p <- softmax_rows(eta)
  n <- nrow(msurp)
  d_nll_d_eta <- (p - truth) / n
  g <- rowSums(d_nll_d_eta * ((-exp(b)) * msurp))
  as.numeric(t(xdiff) %*% g) + 2 * penalty * theta
}

fit_temperature <- function(msurp, xdiff, truth, penalty) {
  start <- rep(0, ncol(xdiff))
  fitted <- optim(
    par = start,
    fn = penalized_nll_temp, gr = penalized_gradient_temp,
    msurp = msurp, xdiff = xdiff, truth = truth, penalty = penalty,
    method = "BFGS", control = list(maxit = 500L, reltol = 1e-12)
  )
  list(theta = fitted$par, objective = fitted$value, convergence = fitted$convergence)
}

predict_temperature <- function(msurp, xdiff, theta) {
  b <- as.numeric(xdiff %*% theta)
  eta <- (-exp(b)) * msurp
  validate_probability(softmax_rows(eta), nrow(msurp))
}

## ---- gradient check (finite difference vs. analytic, before trusting any fit) ----

run_gradient_check <- function() {
  set.seed(1L)
  n_check <- 500L
  fake_p <- matrix(runif(n_check * 4L, 0.01, 0.9), n_check, 4L)
  fake_p <- fake_p / rowSums(fake_p)
  fake_msurp <- m_from_u(log(fake_p))
  fake_xdiff <- cbind(intercept = 1, gap_top2 = rnorm(n_check), price_cv = rnorm(n_check))
  fake_truth <- matrix(0, n_check, 4L)
  fake_truth[cbind(seq_len(n_check), sample(1:4, n_check, replace = TRUE))] <- 1

  start <- rnorm(ncol(fake_xdiff), sd = 0.05)
  analytic <- penalized_gradient_temp(start, fake_msurp, fake_xdiff, fake_truth, penalty = 0.1)
  numeric_grad <- numDeriv::grad(
    penalized_nll_temp, start,
    msurp = fake_msurp, xdiff = fake_xdiff, truth = fake_truth, penalty = 0.1
  )
  max_diff <- max(abs(analytic - numeric_grad))
  cat(sprintf("gradient check task_temperature max|analytic-numeric| = %.3e\n", max_diff))
  stopifnot(max_diff < 1e-6)
  invisible(max_diff)
}

## ---- v14 artifact loader (same files codex_two_head_ensemble_v2.R verified) ----

load_v14_artifact <- function(seed) {
  path <- if (seed == canonical_seed) {
    file.path(v14_dir, "canonical_result.rds")
  } else {
    file.path(v14_dir, sprintf("repeat_result_%d.rds", seed))
  }
  if (!file.exists(path)) stop("Missing verified v14 artifact: ", path)
  object <- readRDS(path)
  required <- c("experiment_id", "fold_map", "candidate_prediction")
  stopifnot(all(required %in% names(object)))
  list(
    fold_map = object$fold_map,
    candidate_prediction = validate_probability(object$candidate_prediction)
  )
}

## ---- nested CV: inner-fold penalty selection, outer-fold evaluation ----

run_nested_outer <- function(wide, msurp, difficulty, truth, fold_map, outer_fold) {
  row_fold <- unname(fold_map[as.character(wide$Case)])
  outer_rows <- which(row_fold == outer_fold)
  train_rows <- which(row_fold != outer_fold)
  inner_folds <- setdiff(1:5, outer_fold)
  stopifnot(
    length(outer_rows) > 0L, length(train_rows) > 0L,
    length(intersect(unique(wide$Case[outer_rows]), unique(wide$Case[train_rows]))) == 0L
  )

  penalty_losses <- numeric(length(penalty_grid))
  for (penalty_index in seq_along(penalty_grid)) {
    penalty <- penalty_grid[[penalty_index]]
    inner_prediction <- matrix(NA_real_, nrow(msurp), 4L)
    for (inner_fold in inner_folds) {
      fit_rows <- which(row_fold != outer_fold & row_fold != inner_fold)
      validation_rows <- which(row_fold == inner_fold)
      scaler <- fit_scaler(difficulty[fit_rows, , drop = FALSE])
      xdiff_fit <- build_xdiff(difficulty[fit_rows, , drop = FALSE], scaler)
      xdiff_val <- build_xdiff(difficulty[validation_rows, , drop = FALSE], scaler)
      fitted <- fit_temperature(msurp[fit_rows, , drop = FALSE], xdiff_fit, truth[fit_rows, , drop = FALSE], penalty)
      inner_prediction[validation_rows, ] <- predict_temperature(
        msurp[validation_rows, , drop = FALSE], xdiff_val, fitted$theta
      )
    }
    tuning_rows <- setdiff(train_rows, which(row_fold == outer_fold))
    penalty_losses[[penalty_index]] <- log_loss_matrix_local(
      truth[tuning_rows, , drop = FALSE], inner_prediction[tuning_rows, , drop = FALSE]
    )
  }
  best_penalty <- penalty_grid[[which.min(penalty_losses)]]
  scaler <- fit_scaler(difficulty[train_rows, , drop = FALSE])
  xdiff_train <- build_xdiff(difficulty[train_rows, , drop = FALSE], scaler)
  xdiff_outer <- build_xdiff(difficulty[outer_rows, , drop = FALSE], scaler)
  fitted <- fit_temperature(msurp[train_rows, , drop = FALSE], xdiff_train, truth[train_rows, , drop = FALSE], best_penalty)
  prediction <- predict_temperature(msurp[outer_rows, , drop = FALSE], xdiff_outer, fitted$theta)
  list(outer_rows = outer_rows, prediction = prediction, selected_penalty = best_penalty, theta = fitted$theta)
}

run_family_oof <- function(wide, msurp, difficulty, truth, fold_map) {
  row_fold <- unname(fold_map[as.character(wide$Case)])
  prediction <- matrix(NA_real_, nrow(msurp), 4L)
  thetas <- vector("list", 5L)
  penalties <- numeric(5L)
  for (outer_fold in 1:5) {
    cat(sprintf("  task-temperature outer fold %d/5\n", outer_fold))
    result <- run_nested_outer(wide, msurp, difficulty, truth, fold_map, outer_fold)
    prediction[result$outer_rows, ] <- result$prediction
    thetas[[outer_fold]] <- result$theta
    penalties[[outer_fold]] <- result$selected_penalty
  }
  list(
    prediction = validate_probability(prediction, nrow(msurp)), row_fold = row_fold,
    thetas = thetas, penalties = penalties
  )
}

## ---- main experiment ----

run_smoke_test <- function(train, truth) {
  cat("\nRunning gradient check...\n")
  run_gradient_check()

  v14 <- load_v14_artifact(canonical_seed)
  msurp <- m_from_u(log(v14$candidate_prediction))
  difficulty <- compute_task_difficulty(train, v14$candidate_prediction)
  result <- run_nested_outer(train, msurp, difficulty, truth, v14$fold_map, outer_fold = 1L)
  baseline_loss <- log_loss_matrix_local(
    truth[result$outer_rows, , drop = FALSE],
    v14$candidate_prediction[result$outer_rows, , drop = FALSE]
  )
  candidate_loss <- log_loss_matrix_local(truth[result$outer_rows, , drop = FALSE], result$prediction)
  cat(sprintf(
    "\nSmoke test: fold-1 v14 loss %.6f, fold-1 temperature loss %.6f, theta=(%s), penalty=%.3g\n",
    baseline_loss, candidate_loss, paste(sprintf("%.4f", result$theta), collapse = ","), result$selected_penalty
  ))
  saveRDS(
    list(experiment_id = experiment_id, result = result, baseline_loss = baseline_loss, candidate_loss = candidate_loss),
    file.path(output_dir, "smoke_result.rds")
  )
  invisible(result)
}

run_experiment <- function() {
  train <- read.csv(file.path("csv files", "train.csv"))
  train <- train[order(train$No), , drop = FALSE]
  rownames(train) <- NULL
  truth <- as.matrix(train[, paste0("Ch", 1:4), drop = FALSE])
  stopifnot(
    nrow(train) == 21565L, length(unique(train$Case)) == 1135L,
    all(table(train$Case) == 19L), all(rowSums(truth) == 1L)
  )

  cat("\nRunning", experiment_id, "\n")

  if (smoke_mode) {
    return(run_smoke_test(train, truth))
  }

  v14_registry <- lapply(all_outer_seeds, load_v14_artifact)
  names(v14_registry) <- as.character(all_outer_seeds)
  canonical_v14 <- v14_registry[[as.character(canonical_seed)]]

  canonical_baseline_loss <- log_loss_matrix_local(truth, canonical_v14$candidate_prediction)
  stopifnot(abs(canonical_baseline_loss - canonical_v14_target) <= 1e-8)
  canonical_difficulty <- compute_task_difficulty(train, canonical_v14$candidate_prediction)
  canonical_msurp <- m_from_u(log(canonical_v14$candidate_prediction))

  canonical_fit <- run_family_oof(train, canonical_msurp, canonical_difficulty, truth, canonical_v14$fold_map)
  candidate_loss <- log_loss_matrix_local(truth, canonical_fit$prediction)
  canonical_bootstrap <- bootstrap_summary(
    truth, canonical_v14$candidate_prediction, canonical_fit$prediction,
    train$Case, bootstrap_replicates, canonical_seed
  )
  canonical_summary <- cbind(
    data.frame(
      experiment = experiment_id, stage = "canonical",
      baseline_logloss = canonical_baseline_loss, candidate_logloss = candidate_loss
    ),
    canonical_bootstrap$summary
  )
  canonical_summary$canonical_pass <- canonical_summary$point_gain > 0 & canonical_summary$lower_95 > 0
  canonical_summary$near_miss <-
    canonical_summary$point_gain > 0 &
    canonical_summary$lower_95 <= 0 &
    canonical_summary$lower_95 >= near_miss_lower_limit
  cat("\nCanonical result:\n")
  print(canonical_summary, digits = 9)
  mean_theta <- Reduce(`+`, canonical_fit$thetas) / length(canonical_fit$thetas)
  cat("Mean fold theta (intercept, gap_top2, price_cv):", paste(sprintf("%.5f", mean_theta), collapse = ", "), "\n")
  cat("Selected penalties per fold:", paste(canonical_fit$penalties, collapse = ", "), "\n")

  write.csv(canonical_summary, file.path(output_dir, "canonical_summary.csv"), row.names = FALSE)
  saveRDS(
    list(
      experiment_id = experiment_id, summary = canonical_summary,
      baseline_prediction = canonical_v14$candidate_prediction,
      candidate_prediction = canonical_fit$prediction, fold_map = canonical_v14$fold_map,
      thetas = canonical_fit$thetas, penalties = canonical_fit$penalties,
      respondent_gain = canonical_bootstrap$respondent_gain, bootstrap = canonical_bootstrap$bootstrap
    ),
    file.path(output_dir, "canonical_result.rds")
  )

  run_repeated <- repeated_mode == "always" ||
    (repeated_mode == "auto" && (isTRUE(canonical_summary$canonical_pass) || isTRUE(canonical_summary$near_miss)))

  if (!run_repeated) {
    verdict <- paste(
      "REJECT task-content-conditioned local temperature (canonical stage); retain exact v14.",
      sprintf(
        "Canonical gain %.9f; 95%% CI [%.9f, %.9f].",
        canonical_summary$point_gain, canonical_summary$lower_95, canonical_summary$upper_95
      )
    )
    writeLines(verdict, file.path(output_dir, "verdict.txt"))
    cat("\n", verdict, "\n", sep = "")
    return(invisible(list(canonical = canonical_summary, repeated = NULL, verdict = verdict)))
  }

  cat("\nRepeated-CV escalation activated (", repeated_mode, ").\n", sep = "")
  case_levels <- sort(unique(train$Case))
  seed_names <- as.character(all_outer_seeds)
  case_gain_matrix <- matrix(
    NA_real_, length(case_levels), length(seed_names),
    dimnames = list(as.character(case_levels), seed_names)
  )
  case_gain_matrix[, as.character(canonical_seed)] <- canonical_bootstrap$respondent_gain
  repeat_rows <- list(`4821` = data.frame(
    seed = canonical_seed, baseline_logloss = canonical_baseline_loss,
    candidate_logloss = candidate_loss, gain = canonical_baseline_loss - candidate_loss
  ))

  for (seed in additional_seeds) {
    seed_name <- as.character(seed)
    repeat_v14 <- v14_registry[[seed_name]]
    repeat_msurp <- m_from_u(log(repeat_v14$candidate_prediction))
    repeat_difficulty <- compute_task_difficulty(train, repeat_v14$candidate_prediction)
    repeat_fit <- run_family_oof(train, repeat_msurp, repeat_difficulty, truth, repeat_v14$fold_map)
    baseline_loss <- log_loss_matrix_local(truth, repeat_v14$candidate_prediction)
    seed_candidate_loss <- log_loss_matrix_local(truth, repeat_fit$prediction)
    case_gain <- respondent_gain(truth, repeat_v14$candidate_prediction, repeat_fit$prediction, train$Case)
    case_gain_matrix[, seed_name] <- case_gain
    repeat_rows[[seed_name]] <- data.frame(
      seed = seed, baseline_logloss = baseline_loss,
      candidate_logloss = seed_candidate_loss, gain = baseline_loss - seed_candidate_loss
    )
    cat(sprintf("  seed %d done, gain = %.9f\n", seed, baseline_loss - seed_candidate_loss))
  }

  repeated_by_seed <- do.call(rbind, repeat_rows)
  average_case_gain <- rowMeans(case_gain_matrix)
  pooled <- summarize_case_gain(average_case_gain, bootstrap_replicates, canonical_seed)
  repeated_summary <- cbind(
    data.frame(experiment = experiment_id, stage = "repeated_cv"),
    pooled$summary,
    data.frame(positive_repeats = sum(repeated_by_seed$gain > 0), n_repeats = nrow(repeated_by_seed))
  )
  repeated_summary$promote <-
    repeated_summary$point_gain > 0 & repeated_summary$lower_95 > 0 & repeated_summary$positive_repeats >= 5L

  write.csv(repeated_by_seed, file.path(output_dir, "repeated_cv_by_seed.csv"), row.names = FALSE)
  write.csv(repeated_summary, file.path(output_dir, "repeated_cv_summary.csv"), row.names = FALSE)

  verdict <- if (isTRUE(repeated_summary$promote)) {
    "PROMOTE task-content-conditioned local temperature for a separate full-data build audit."
  } else {
    "REJECT task-content-conditioned local temperature; retain the exact set-context v14 submission."
  }
  writeLines(verdict, file.path(output_dir, "verdict.txt"))
  cat("\nRepeated-CV result:\n")
  print(repeated_summary, digits = 9)
  cat("\n", verdict, "\n", sep = "")
  invisible(list(canonical = canonical_summary, repeated = repeated_summary, verdict = verdict))
}

run_experiment()
