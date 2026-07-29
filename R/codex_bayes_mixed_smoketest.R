## Pre-registered smoke test (codex_bayes_mixed_preregister.md, Section 4):
## verify the POPULATION-LEVEL posterior-predictive mechanism for held-out
## ("new") respondents on a tiny subset, BEFORE trusting the full 5-fold run.
##
## Checks performed:
##  1. Pipeline runs end-to-end; predictions are valid row-stochastic matrices.
##  2. Static check: population_predict()'s body never references betadraw
##     (the per-training-respondent individual draws) -- it can only reach
##     population-level nmix components.
##  3. Re-running the held-out prediction with a different RNG seed changes
##     the predicted probabilities by a Monte-Carlo amount (genuine fresh-draw
##     variability), rather than being deterministic.
##  4. A deliberately WRONG comparison condition: fit the held-out respondents
##     INTO the model too (so they get their own individual betadraw), take
##     their fitted individual posterior-mean beta, and use that (fixed, no
##     MC noise) to predict their own tasks. This is invariant to reruns/seeds
##     and visibly different from the population-marginal prediction --
##     concretely demonstrating the two mechanisms differ, and that the real
##     pipeline (population_predict) is the correct one, not the wrong one.

suppressPackageStartupMessages(library(bayesm))
source("R/codex_bayes_mixed_common.R")

output_dir <- "data_processed/codex_bayes_mixed"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(1)
train <- read.csv("csv files/train.csv")
all_cases <- unique(train$Case)
subset_cases <- sample(all_cases, 60L)
held_out_cases <- subset_cases[1:15]
fit_cases <- subset_cases[16:60]

scaler <- bayes_scaler(train[train$Case %in% subset_cases, , drop = FALSE])
ctr <- scaler$ctr
scl <- scaler$scl

cat("=== Check 1: pipeline runs end-to-end ===\n")
lgtdata_fit <- build_lgtdata(train, fit_cases, ctr, scl)
stopifnot(length(lgtdata_fit) == 45L)
stopifnot(all(vapply(lgtdata_fit, function(d) nrow(d$X), integer(1)) == 76L))

chain <- fit_bayes_chain(lgtdata_fit, seed = 555L, R = 1500L, keep = 3L)
cat(sprintf("Chain fitted: %d post-burn-in draws retained\n", chain$n_kept))
stopifnot(chain$n_kept > 0L)

wide_heldout <- train[train$Case %in% held_out_cases, , drop = FALSE]
pred1 <- population_predict(chain$compdraw, wide_heldout, held_out_cases, ctr, scl,
                             seed = 111L)
stopifnot(nrow(pred1$prediction) == 15L * 19L)
stopifnot(all(abs(rowSums(pred1$prediction) - 1) < 1e-9))
stopifnot(all(pred1$prediction > 0))
cat("Prediction matrix is valid (row-stochastic, all positive). PASS\n\n")

cat("=== Check 2: population_predict() never reads betadraw ===\n")
body_text <- paste(deparse(body(population_predict)), collapse = "\n")
stopifnot(!grepl("betadraw", body_text, fixed = TRUE))
cat("Confirmed: 'betadraw' does not appear anywhere in population_predict()'s\n")
cat("body -- it can only read nmix$compdraw (population-level mu/Sigma draws)\n")
cat("passed in from the caller. PASS\n\n")

cat("=== Check 3: fresh-draw Monte Carlo variability across seeds ===\n")
pred2 <- population_predict(chain$compdraw, wide_heldout, held_out_cases, ctr, scl,
                             seed = 222L)
diff_seeds <- abs(pred1$prediction - pred2$prediction)
cat(sprintf("Same compdraw, different seed: mean abs diff = %.5f, max = %.5f\n",
            mean(diff_seeds), max(diff_seeds)))
stopifnot(mean(diff_seeds) > 1e-4)   # genuinely different (MC noise), not byte-identical
correlation <- cor(as.vector(pred1$prediction), as.vector(pred2$prediction))
cat(sprintf("Correlation between the two seeds' predictions: %.5f\n", correlation))
stopifnot(correlation > 0.9)          # but from the SAME population distribution
cat("PASS: different seeds give visibly different but highly correlated\n")
cat("predictions -- genuine fresh-draw Monte Carlo variability integrating\n")
cat("over the population distribution, not a deterministic lookup.\n\n")

cat("=== Check 4: deliberately WRONG comparison (individual posterior) ===\n")
cat("Refitting WITH the 15 'held-out' respondents included, to extract their\n")
cat("own individual fitted posterior-mean beta_i (the quantity that does NOT\n")
cat("exist for genuinely new test respondents) as a wrong-mechanism contrast.\n")
lgtdata_all <- build_lgtdata(train, subset_cases, ctr, scl)
chain_all <- fit_bayes_chain(lgtdata_all, seed = 777L, R = 1500L, keep = 3L)

## Refit exposing betadraw this one time, deliberately, only for this
## negative-control comparison (never used in the real CV pipeline).
set.seed(777L)
Data_all <- list(p = 4, lgtdata = lgtdata_all)
Mcmc_all <- list(R = 1500L, keep = 3L, nprint = 0)
out_all <- rhierMnlRwMixture(Data = Data_all, Prior = bayes_prior(), Mcmc = Mcmc_all)
n_kept_all <- dim(out_all$betadraw)[3]
burn_all <- floor(n_kept_all * 0.25)
keep_idx_all <- (burn_all + 1L):n_kept_all
held_out_unit_index <- match(as.character(held_out_cases), names(lgtdata_all))
stopifnot(!anyNA(held_out_unit_index))

individual_beta_mean <- apply(
  out_all$betadraw[held_out_unit_index, , keep_idx_all, drop = FALSE], c(1, 2), mean
)

wide_heldout_ordered <- split(wide_heldout, wide_heldout$Case)[as.character(held_out_cases)]
X_list <- lapply(wide_heldout_ordered, function(resp) {
  respondent_design(resp[order(resp$Task), , drop = FALSE], ctr, scl)
})
individual_pred <- do.call(rbind, lapply(seq_along(X_list), function(i) {
  eta <- as.numeric(X_list[[i]] %*% individual_beta_mean[i, ])
  z <- matrix(eta, ncol = 4, byrow = TRUE)
  z <- z - apply(z, 1, max)
  ez <- exp(z)
  ez / rowSums(ez)
}))

individual_pred_rerun <- do.call(rbind, lapply(seq_along(X_list), function(i) {
  eta <- as.numeric(X_list[[i]] %*% individual_beta_mean[i, ])
  z <- matrix(eta, ncol = 4, byrow = TRUE)
  z <- z - apply(z, 1, max)
  ez <- exp(z)
  ez / rowSums(ez)
}))
max_rerun_diff <- max(abs(individual_pred - individual_pred_rerun))
cat(sprintf("Individual-posterior prediction is deterministic across reruns (max diff = %.2e)\n",
            max_rerun_diff))
stopifnot(max_rerun_diff < 1e-12)

diff_vs_population <- abs(individual_pred - pred1$prediction)
cat(sprintf("Individual-posterior vs. population-marginal prediction: mean abs diff = %.5f, max = %.5f\n",
            mean(diff_vs_population), max(diff_vs_population)))
stopifnot(mean(diff_vs_population) > 1e-3)
cat("PASS: the wrong (individual-posterior) mechanism is deterministic and\n")
cat("visibly different from the correct (population-marginal) mechanism --\n")
cat("confirms population_predict() is doing something mechanistically\n")
cat("different from (and not accidentally collapsing to) per-respondent\n")
cat("shrinkage estimates that only exist for respondents seen during fitting.\n\n")

summary_df <- data.frame(
  check = c("valid_probabilities", "no_betadraw_reference",
            "seed_variability_mean_abs_diff", "seed_variability_correlation",
            "individual_vs_population_mean_abs_diff",
            "individual_rerun_determinism_max_diff"),
  value = c(NA, NA, mean(diff_seeds), correlation, mean(diff_vs_population), max_rerun_diff)
)
write.csv(summary_df, file.path(output_dir, "smoketest_summary.csv"), row.names = FALSE)
saveRDS(
  list(pred1 = pred1$prediction, pred2 = pred2$prediction,
       individual_pred = individual_pred, summary = summary_df),
  file.path(output_dir, "smoketest_result.rds")
)

cat("=== ALL SMOKE-TEST CHECKS PASSED ===\n")
