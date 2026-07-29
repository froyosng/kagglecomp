## Pre-registered full evaluation (codex_bayes_mixed_preregister.md).
## Canonical 5-fold respondent-grouped CV of the Bayesian hierarchical mixed
## logit (bayesm::rhierMnlRwMixture, ncomp=1, 23 random coefficients: 19
## standardized attributes + standardized Price + ASC2/ASC3/ASC4), refit from
## scratch per fold (2 chains), scored on held-out respondents via the
## POPULATION-LEVEL posterior-predictive mechanism verified in
## codex_bayes_mixed_smoketest.R. Then: reconstruct the current-best
## (ensemble_v11 + MLP) baseline from raw cached OOF artifacts, fold-cross-fit
## a candidate blend weight, and run the pre-registered 100,000-replicate
## respondent-clustered paired bootstrap against that baseline.

suppressPackageStartupMessages(library(bayesm))
source("R/codex_bayes_mixed_common.R")

output_dir <- "data_processed/codex_bayes_mixed"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

R_DRAWS <- 8000L
KEEP <- 8L

train <- read.csv("csv files/train.csv")
truth <- as.matrix(train[, paste0("Ch", 1:4)])
saved_fold <- readRDS("data_processed/oof_ensemble_v10.rds")
fold_of_case <- saved_fold$fold_of_case
row_fold <- unname(fold_of_case[as.character(train$Case)])
stopifnot(!anyNA(row_fold), length(row_fold) == nrow(train))

n_case_total <- length(unique(train$Case))
stopifnot(n_case_total == 1135L)
cat("Fold sizes (cases):\n")
print(table(fold_of_case))

my_bayes_oof <- matrix(NA_real_, nrow(train), 4L)
convergence_all <- list()
loglike_all <- list()
timing_all <- list()

for (fold in 1:5) {
  cat(sprintf("\n========== FOLD %d / 5 ==========\n", fold))
  train_cases <- as.integer(names(fold_of_case)[fold_of_case != fold])
  held_cases <- as.integer(names(fold_of_case)[fold_of_case == fold])
  stopifnot(length(held_cases) == 227L)
  stopifnot(length(intersect(train_cases, held_cases)) == 0L)

  train_wide <- train[train$Case %in% train_cases, , drop = FALSE]
  held_wide <- train[train$Case %in% held_cases, , drop = FALSE]

  # Scaler computed from THIS FOLD'S TRAINING rows only (no leakage from the
  # held-out respondents into standardization), per the pre-registration.
  scaler <- bayes_scaler(train_wide)

  t0 <- proc.time()[["elapsed"]]
  lgtdata <- build_lgtdata(train, train_cases, scaler$ctr, scaler$scl)
  stopifnot(length(lgtdata) == length(train_cases))
  stopifnot(!anyNA(unlist(lapply(lgtdata, function(d) d$X))))

  fitted <- fit_bayes_fold(lgtdata, fold_seed_offset = fold * 100L,
                           R = R_DRAWS, keep = KEEP)
  t1 <- proc.time()[["elapsed"]]
  cat(sprintf("Fold %d: 2 chains fitted in %.1f seconds (%d pooled draws)\n",
              fold, t1 - t0, length(fitted$compdraw)))
  print(fitted$convergence)

  t2 <- proc.time()[["elapsed"]]
  pred <- population_predict(fitted$compdraw, held_wide, held_cases,
                             scaler$ctr, scaler$scl, seed = 4821L + fold)
  t3 <- proc.time()[["elapsed"]]
  cat(sprintf("Fold %d: population-level prediction for %d held-out respondents in %.1f seconds\n",
              fold, length(held_cases), t3 - t2))

  held_wide_ordered <- held_wide[order(match(held_wide$Case, held_cases), held_wide$Task), ]
  row_idx <- match(held_wide_ordered$No, train$No)
  stopifnot(!anyNA(row_idx), length(row_idx) == nrow(pred$prediction))
  my_bayes_oof[row_idx, ] <- pred$prediction

  convergence_all[[fold]] <- cbind(fold = fold, fitted$convergence)
  loglike_all[[fold]] <- data.frame(
    fold = fold,
    draw = seq_along(c(fitted$loglike1, fitted$loglike2)),
    chain = rep(1:2, times = c(length(fitted$loglike1), length(fitted$loglike2))),
    loglike = c(fitted$loglike1, fitted$loglike2)
  )
  timing_all[[fold]] <- data.frame(fold = fold, fit_seconds = t1 - t0,
                                    predict_seconds = t3 - t2)

  saveRDS(
    list(fold = fold, prediction = pred$prediction, held_cases = held_cases,
         convergence = fitted$convergence, var_names = fitted$var_names),
    file.path(output_dir, sprintf("fold_%d_result.rds", fold))
  )
}

stopifnot(!anyNA(my_bayes_oof))
stopifnot(all(abs(rowSums(my_bayes_oof) - 1) < 1e-8))

bayes_standalone_loss <- log_loss_matrix(truth, my_bayes_oof)
cat(sprintf("\n=== Bayesian mixed logit standalone canonical CV log loss: %.6f ===\n",
            bayes_standalone_loss))

convergence_summary <- do.call(rbind, convergence_all)
write.csv(convergence_summary, file.path(output_dir, "convergence_summary.csv"),
          row.names = FALSE)
write.csv(do.call(rbind, loglike_all), file.path(output_dir, "loglike_trace.csv"),
          row.names = FALSE)
write.csv(do.call(rbind, timing_all), file.path(output_dir, "timing.csv"),
          row.names = FALSE)

saveRDS(
  list(prediction = my_bayes_oof, truth = truth, row_fold = row_fold,
       standalone_logloss = bayes_standalone_loss),
  file.path(output_dir, "bayes_mixed_oof.rds")
)

## ---- Reconstruct the current-best baseline (raw cached artifacts) --------

cat("\n=== Reconstructing current-best (ensemble_v11 + MLP) baseline ===\n")
baseline <- reconstruct_current_best(row_fold)
cat(sprintf("Reconstructed current-best CV log loss: %.6f (expect 1.143789)\n",
            baseline$logloss))
write.csv(
  data.frame(fold = 1:5, mlp_weight = baseline$per_fold_mlp_weight),
  file.path(output_dir, "baseline_per_fold_mlp_weight.csv"),
  row.names = FALSE
)

## ---- Fold-cross-fitted candidate blend -----------------------------------

weight_grid <- seq(0, 0.30, by = 0.01)
candidate_oof <- matrix(NA_real_, nrow(truth), 4L)
blend_weight_per_fold <- numeric(5L)
for (fold in 1:5) {
  fit_rows <- row_fold != fold
  valid_rows <- row_fold == fold
  losses <- vapply(weight_grid, function(w) {
    log_loss_matrix(
      truth[fit_rows, ],
      (1 - w) * baseline$prediction[fit_rows, ] + w * my_bayes_oof[fit_rows, ]
    )
  }, numeric(1))
  best_w <- weight_grid[[which.min(losses)]]
  blend_weight_per_fold[fold] <- best_w
  candidate_oof[valid_rows, ] <-
    (1 - best_w) * baseline$prediction[valid_rows, ] + best_w * my_bayes_oof[valid_rows, ]
}
stopifnot(!anyNA(candidate_oof))
candidate_loss <- log_loss_matrix(truth, candidate_oof)
cat(sprintf("Fold-cross-fitted candidate blend CV log loss: %.6f\n", candidate_loss))
cat(sprintf("Point gain vs current best: %.6f\n", baseline$logloss - candidate_loss))
cat("Per-fold blend weights on the Bayesian mixed-logit component:\n")
print(data.frame(fold = 1:5, bayes_weight = blend_weight_per_fold))

write.csv(
  data.frame(fold = 1:5, bayes_weight = blend_weight_per_fold),
  file.path(output_dir, "candidate_blend_weights.csv"),
  row.names = FALSE
)

## ---- Respondent-clustered paired bootstrap (pre-registered promotion test) -

cat("\n=== Respondent-clustered paired bootstrap (100,000 replicates) ===\n")
boot_result <- respondent_bootstrap_gain(
  truth, baseline$prediction, candidate_oof, train$Case,
  seed = 4821L, n_boot = 100000L
)
print(boot_result$summary, digits = 7)

promoted <- boot_result$summary$lower_95 > 0
cat(sprintf("\nPROMOTION DECISION (lower_95 > 0): %s\n", promoted))

write.csv(boot_result$summary, file.path(output_dir, "bootstrap_summary.csv"),
          row.names = FALSE)
saveRDS(
  list(
    baseline_logloss = baseline$logloss,
    bayes_standalone_logloss = bayes_standalone_loss,
    candidate_logloss = candidate_loss,
    blend_weight_per_fold = blend_weight_per_fold,
    bootstrap_summary = boot_result$summary,
    respondent_gain = boot_result$respondent_gain,
    bootstrap_draws = boot_result$bootstrap,
    promoted = promoted
  ),
  file.path(output_dir, "final_result.rds")
)

## ---- Population-level parameter summaries (headline diagnostics) ---------

cat("\n=== Population-level posterior summaries (pooled across folds) ===\n")
var_names <- c(paste0("z_", bayes_vars), "ASC2", "ASC3", "ASC4")
param_rows <- list()
for (fold in 1:5) {
  res <- readRDS(file.path(output_dir, sprintf("fold_%d_result.rds", fold)))
  param_rows[[fold]] <- cbind(fold = fold, res$convergence)
}
param_summary <- do.call(rbind, param_rows)
print(param_summary, digits = 6)
write.csv(param_summary, file.path(output_dir, "population_param_summary.csv"),
          row.names = FALSE)

cat("\n=== DONE ===\n")
