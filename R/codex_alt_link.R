# Alternative-specific asymmetric probability link on frozen v14 utilities.
#
# Pre-registration:
#   codex_alt_link_preregister.md
#
# Smoke test (1 outer fold, gradient check, no repeated CV):
#   Sys.setenv(ALT_LINK_SMOKE = "1")
#   source("R/codex_alt_link.R")
#   Sys.unsetenv("ALT_LINK_SMOKE")
#
# Full run:
#   Sys.unsetenv("ALT_LINK_SMOKE")
#   Sys.setenv(ALT_LINK_REPEATED = "auto")
#   source("R/codex_alt_link.R")

suppressPackageStartupMessages({
  library(numDeriv)
})

options(stringsAsFactors = FALSE)

experiment_id <- "alt_link_v1"
output_dir <- file.path("data_processed", "codex_alt_link")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

smoke_mode <- identical(Sys.getenv("ALT_LINK_SMOKE", "0"), "1")
repeated_mode <- tolower(Sys.getenv("ALT_LINK_REPEATED", "auto"))
stopifnot(repeated_mode %in% c("auto", "never", "always"))

canonical_seed <- 4821L
additional_seeds <- c(1907L, 2719L, 6151L, 8293L, 104729L)
all_outer_seeds <- c(canonical_seed, additional_seeds)
canonical_v14_target <- 1.1435331472708
near_miss_lower_limit <- -0.00075

bootstrap_replicates <- if (smoke_mode) {
  200L
} else {
  as.integer(Sys.getenv("ALT_LINK_N_BOOT", "100000"))
}

v14_dir <- file.path("data_processed", "codex_set_context_network")
penalty_grid <- c(1.0, 0.1, 0.01, 0.001, 0.0)

families <- list(
  family1_global_scale = c(a = FALSE, b = TRUE, c = FALSE),
  family2_optout_shift = c(a = FALSE, b = TRUE, c = TRUE),
  family3_shape_scale = c(a = TRUE, b = TRUE, c = FALSE)
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

## ---- the link itself ----
## u_ij = log(p^v14_ij) <= 0 always (a valid log-probability). m_ij = -u_ij >= 0
## is that row's "surprisal" for alternative j. The link reshapes m by an
## optional power (a, "shape"), rescales by exp(b) ("scale"/temperature), and
## optionally adds a single alternative-4-only shift (c, "opt-out shift"),
## then renormalizes via softmax. theta = (a, b, c) = (0, 0, 0) reproduces
## plain v14 softmax exactly (eta = -m = u, unchanged).

m_from_u <- function(u) pmax(-u, 1e-12)

# NOTE: the surprisal matrix is deliberately named `msurp`, not `m` -- passing
# a named argument `m = ...` through `...` into optim()/numDeriv::grad() (both
# of which have their own `method`/`method.args` formals) triggers R's
# partial-argument-matching ambiguity ("argument matches multiple formal
# arguments"), caught by the gradient-check smoke test before any real fit.
link_eta <- function(msurp, theta) {
  raw <- msurp^(1 + theta[["a"]])
  eta <- -exp(theta[["b"]]) * raw
  eta[, 4] <- eta[, 4] + theta[["c"]]
  eta
}

softmax_rows <- function(eta) {
  eta <- eta - apply(eta, 1L, max)
  ez <- exp(eta)
  ez / rowSums(ez)
}

penalized_nll <- function(free_theta, active_names, msurp, truth, penalty) {
  theta <- c(a = 0, b = 0, c = 0)
  theta[active_names] <- free_theta
  eta <- link_eta(msurp, theta)
  p <- softmax_rows(eta)
  nll <- -mean(rowSums(truth * log(pmax(p, 1e-15))))
  nll + penalty * sum(free_theta^2)
}

penalized_gradient <- function(free_theta, active_names, msurp, truth, penalty) {
  theta <- c(a = 0, b = 0, c = 0)
  theta[active_names] <- free_theta
  raw <- msurp^(1 + theta[["a"]])
  eta <- -exp(theta[["b"]]) * raw
  eta[, 4] <- eta[, 4] + theta[["c"]]
  p <- softmax_rows(eta)
  n <- nrow(msurp)
  d_nll_d_eta <- (p - truth) / n

  d_eta_d_b <- -exp(theta[["b"]]) * raw
  d_eta_d_a <- -exp(theta[["b"]]) * raw * log(pmax(msurp, 1e-12))
  d_eta_d_c <- matrix(0, nrow(msurp), 4L)
  d_eta_d_c[, 4] <- 1

  full_gradient <- c(
    a = sum(d_nll_d_eta * d_eta_d_a),
    b = sum(d_nll_d_eta * d_eta_d_b),
    c = sum(d_nll_d_eta * d_eta_d_c)
  )
  full_gradient[active_names] + 2 * penalty * free_theta
}

fit_link <- function(msurp, truth, active_names, penalty) {
  start <- rep(0, length(active_names))
  fitted <- optim(
    par = start,
    fn = penalized_nll, gr = penalized_gradient,
    active_names = active_names, msurp = msurp, truth = truth, penalty = penalty,
    method = "BFGS", control = list(maxit = 500L, reltol = 1e-12)
  )
  theta <- c(a = 0, b = 0, c = 0)
  theta[active_names] <- fitted$par
  list(theta = theta, objective = fitted$value, convergence = fitted$convergence)
}

predict_link <- function(msurp, theta) {
  eta <- link_eta(msurp, theta)
  validate_probability(softmax_rows(eta), nrow(msurp))
}

## ---- gradient check (finite difference vs. analytic, before trusting any fit) ----

run_gradient_check <- function() {
  set.seed(1L)
  n_check <- 500L
  fake_p <- matrix(runif(n_check * 4L, 0.01, 0.9), n_check, 4L)
  fake_p <- fake_p / rowSums(fake_p)
  fake_u <- log(fake_p)
  fake_m <- m_from_u(fake_u)
  fake_truth <- matrix(0, n_check, 4L)
  fake_truth[cbind(seq_len(n_check), sample(1:4, n_check, replace = TRUE))] <- 1

  results <- list()
  for (family_name in names(families)) {
    active <- names(families[[family_name]])[families[[family_name]]]
    start <- rnorm(length(active), sd = 0.05)
    analytic <- penalized_gradient(start, active, fake_m, fake_truth, penalty = 0.1)
    numeric_grad <- numDeriv::grad(
      penalized_nll, start,
      active_names = active, msurp = fake_m, truth = fake_truth, penalty = 0.1
    )
    max_diff <- max(abs(analytic - numeric_grad))
    results[[family_name]] <- data.frame(
      family = family_name, max_abs_diff = max_diff,
      analytic = paste(sprintf("%.6f", analytic), collapse = ","),
      numeric = paste(sprintf("%.6f", numeric_grad), collapse = ",")
    )
    cat(sprintf(
      "gradient check %-24s max|analytic-numeric| = %.3e\n", family_name, max_diff
    ))
    stopifnot(max_diff < 1e-6)
  }
  do.call(rbind, results)
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

run_nested_outer_family <- function(m, truth, fold_map, case, active_names, outer_fold) {
  row_fold <- unname(fold_map[as.character(case)])
  outer_rows <- which(row_fold == outer_fold)
  train_rows <- which(row_fold != outer_fold)
  inner_folds <- setdiff(1:5, outer_fold)
  stopifnot(
    length(outer_rows) > 0L, length(train_rows) > 0L,
    length(intersect(unique(case[outer_rows]), unique(case[train_rows]))) == 0L
  )

  penalty_losses <- numeric(length(penalty_grid))
  for (penalty_index in seq_along(penalty_grid)) {
    penalty <- penalty_grid[[penalty_index]]
    inner_prediction <- matrix(NA_real_, nrow(m), 4L)
    for (inner_fold in inner_folds) {
      fit_rows <- which(row_fold != outer_fold & row_fold != inner_fold)
      validation_rows <- which(row_fold == inner_fold)
      fitted <- fit_link(m[fit_rows, , drop = FALSE], truth[fit_rows, , drop = FALSE], active_names, penalty)
      inner_prediction[validation_rows, ] <- predict_link(m[validation_rows, , drop = FALSE], fitted$theta)
    }
    tuning_rows <- setdiff(train_rows, which(row_fold == outer_fold))
    penalty_losses[[penalty_index]] <- log_loss_matrix_local(
      truth[tuning_rows, , drop = FALSE], inner_prediction[tuning_rows, , drop = FALSE]
    )
  }
  # penalty_grid ordered strongest to weakest; ties favor stronger shrinkage.
  best_penalty <- penalty_grid[[which.min(penalty_losses)]]
  fitted <- fit_link(m[train_rows, , drop = FALSE], truth[train_rows, , drop = FALSE], active_names, best_penalty)
  prediction <- predict_link(m[outer_rows, , drop = FALSE], fitted$theta)
  list(
    outer_rows = outer_rows, prediction = prediction,
    selected_penalty = best_penalty, theta = fitted$theta
  )
}

run_family_oof <- function(m, truth, fold_map, case, active_names) {
  row_fold <- unname(fold_map[as.character(case)])
  prediction <- matrix(NA_real_, nrow(m), 4L)
  thetas <- vector("list", 5L)
  penalties <- numeric(5L)
  for (outer_fold in 1:5) {
    result <- run_nested_outer_family(m, truth, fold_map, case, active_names, outer_fold)
    prediction[result$outer_rows, ] <- result$prediction
    thetas[[outer_fold]] <- result$theta
    penalties[[outer_fold]] <- result$selected_penalty
  }
  list(
    prediction = validate_probability(prediction, nrow(m)), row_fold = row_fold,
    thetas = thetas, penalties = penalties
  )
}

respondent_bootstrap_for_family <- function(truth, baseline, candidate, case, replicates, seed) {
  bootstrap_summary(truth, baseline, candidate, case, replicates, seed)
}

## ---- main experiment ----

run_smoke_test <- function(train, truth) {
  cat("\nRunning gradient checks...\n")
  run_gradient_check()

  v14 <- load_v14_artifact(canonical_seed)
  m <- m_from_u(log(v14$candidate_prediction))
  family_name <- "family3_shape_scale"
  active <- names(families[[family_name]])[families[[family_name]]]
  result <- run_nested_outer_family(m, truth, v14$fold_map, train$Case, active, outer_fold = 1L)
  baseline_loss <- log_loss_matrix_local(
    truth[result$outer_rows, , drop = FALSE],
    v14$candidate_prediction[result$outer_rows, , drop = FALSE]
  )
  candidate_loss <- log_loss_matrix_local(truth[result$outer_rows, , drop = FALSE], result$prediction)
  cat(sprintf(
    "\nSmoke test (%s): fold-1 v14 loss %.6f, fold-1 link loss %.6f, theta=(a=%.4f,b=%.4f,c=%.4f), penalty=%.3g\n",
    family_name, baseline_loss, candidate_loss,
    result$theta[["a"]], result$theta[["b"]], result$theta[["c"]], result$selected_penalty
  ))
  saveRDS(
    list(experiment_id = experiment_id, result = result, baseline_loss = baseline_loss, candidate_loss = candidate_loss),
    file.path(output_dir, "smoke_result.rds")
  )
  invisible(result)
}

run_one_family <- function(family_name, active_names, train, truth, v14_registry, penalty_note = "") {
  cat(sprintf("\n==== %s ====\n", family_name))
  canonical_v14 <- v14_registry[[as.character(canonical_seed)]]
  m_canonical <- m_from_u(log(canonical_v14$candidate_prediction))
  canonical_baseline_loss <- log_loss_matrix_local(truth, canonical_v14$candidate_prediction)
  stopifnot(abs(canonical_baseline_loss - canonical_v14_target) <= 1e-8)

  canonical_fit <- run_family_oof(m_canonical, truth, canonical_v14$fold_map, train$Case, active_names)
  candidate_loss <- log_loss_matrix_local(truth, canonical_fit$prediction)
  canonical_bootstrap <- bootstrap_summary(
    truth, canonical_v14$candidate_prediction, canonical_fit$prediction,
    train$Case, bootstrap_replicates, canonical_seed
  )
  canonical_summary <- cbind(
    data.frame(
      experiment = experiment_id, family = family_name, stage = "canonical",
      baseline_logloss = canonical_baseline_loss, candidate_logloss = candidate_loss
    ),
    canonical_bootstrap$summary
  )
  canonical_summary$canonical_pass <- canonical_summary$point_gain > 0 & canonical_summary$lower_95 > 0
  canonical_summary$near_miss <-
    canonical_summary$point_gain > 0 &
    canonical_summary$lower_95 <= 0 &
    canonical_summary$lower_95 >= near_miss_lower_limit
  cat("Canonical result:\n")
  print(canonical_summary, digits = 9)
  mean_theta <- Reduce(`+`, canonical_fit$thetas) / length(canonical_fit$thetas)
  cat("Mean fold theta:", paste(sprintf("%s=%.5f", names(mean_theta), mean_theta), collapse = ", "), "\n")
  cat("Selected penalties per fold:", paste(canonical_fit$penalties, collapse = ", "), "\n")

  family_dir <- file.path(output_dir, family_name)
  dir.create(family_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(canonical_summary, file.path(family_dir, "canonical_summary.csv"), row.names = FALSE)
  saveRDS(
    list(
      experiment_id = experiment_id, family = family_name, summary = canonical_summary,
      baseline_prediction = canonical_v14$candidate_prediction,
      candidate_prediction = canonical_fit$prediction, fold_map = canonical_v14$fold_map,
      thetas = canonical_fit$thetas, penalties = canonical_fit$penalties,
      respondent_gain = canonical_bootstrap$respondent_gain, bootstrap = canonical_bootstrap$bootstrap
    ),
    file.path(family_dir, "canonical_result.rds")
  )

  run_repeated <- repeated_mode == "always" ||
    (repeated_mode == "auto" && (isTRUE(canonical_summary$canonical_pass) || isTRUE(canonical_summary$near_miss)))

  if (!run_repeated) {
    verdict <- paste(
      sprintf("REJECT %s (canonical stage); retain exact v14.", family_name),
      sprintf(
        "Canonical gain %.9f; 95%% CI [%.9f, %.9f].",
        canonical_summary$point_gain, canonical_summary$lower_95, canonical_summary$upper_95
      )
    )
    writeLines(verdict, file.path(family_dir, "verdict.txt"))
    cat("\n", verdict, "\n", sep = "")
    return(list(canonical = canonical_summary, repeated = NULL, verdict = verdict))
  }

  cat(sprintf("\nRepeated-CV escalation activated for %s (%s).\n", family_name, repeated_mode))
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
    m_repeat <- m_from_u(log(repeat_v14$candidate_prediction))
    repeat_fit <- run_family_oof(m_repeat, truth, repeat_v14$fold_map, train$Case, active_names)
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
    data.frame(experiment = experiment_id, family = family_name, stage = "repeated_cv"),
    pooled$summary,
    data.frame(positive_repeats = sum(repeated_by_seed$gain > 0), n_repeats = nrow(repeated_by_seed))
  )
  repeated_summary$promote <-
    repeated_summary$point_gain > 0 & repeated_summary$lower_95 > 0 & repeated_summary$positive_repeats >= 5L

  write.csv(repeated_by_seed, file.path(family_dir, "repeated_cv_by_seed.csv"), row.names = FALSE)
  write.csv(repeated_summary, file.path(family_dir, "repeated_cv_summary.csv"), row.names = FALSE)

  verdict <- if (isTRUE(repeated_summary$promote)) {
    sprintf("PROMOTE %s for a separate full-data build audit.", family_name)
  } else {
    sprintf("REJECT %s; retain the exact set-context v14 submission.", family_name)
  }
  writeLines(verdict, file.path(family_dir, "verdict.txt"))
  cat("\nRepeated-CV result:\n")
  print(repeated_summary, digits = 9)
  cat("\n", verdict, "\n", sep = "")
  list(canonical = canonical_summary, repeated = repeated_summary, verdict = verdict)
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

  all_results <- list()
  for (family_name in names(families)) {
    active_names <- names(families[[family_name]])[families[[family_name]]]
    all_results[[family_name]] <- run_one_family(family_name, active_names, train, truth, v14_registry)
  }
  saveRDS(all_results, file.path(output_dir, "all_family_results.rds"))
  invisible(all_results)
}

run_experiment()
