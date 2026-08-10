# No-refit shrinkage diagnostic for the relative price-gap near-miss.
# Blends p0 (m8trpg-alone OOF) with p1 (m8trpg+relative-gap OOF) via
# p_alpha = (1-alpha)*p0 + alpha*p1 for alpha in {0.25, 0.5, 0.75, 1}.
# Alpha is chosen using ONLY the canonical seed, then frozen and evaluated
# purely out-of-sample on the other 5 repeated-CV seeds -- no refitting,
# every prediction matrix used here was already computed and cached by
# R/codex_relative_price_gap_cv.R and R/codex_relative_price_gap_repeated_cv.R.
#
# Run:
#   source("R/codex_relative_price_gap_shrinkage.R")

output_dir <- file.path("data_processed", "codex_relative_price_gap")
additional_seeds <- c(1907L, 2719L, 6151L, 8293L, 104729L)
alpha_grid <- c(0.25, 0.5, 0.75, 1.0)

log_loss_matrix <- function(truth, prediction) {
  prediction <- pmax(prediction, 1e-15)
  prediction <- prediction / rowSums(prediction)
  -mean(rowSums(truth * log(prediction)))
}
row_log_loss <- function(truth, prediction) {
  prediction <- pmax(prediction, 1e-15)
  prediction <- prediction / rowSums(prediction)
  -rowSums(truth * log(prediction))
}
bootstrap_case_means <- function(case_gain, replicates, seed, chunk_size = 1000L) {
  case_gain <- as.numeric(case_gain)
  set.seed(seed)
  n_case <- length(case_gain)
  output <- numeric(replicates)
  start <- 1L
  while (start <= replicates) {
    count <- min(chunk_size, replicates - start + 1L)
    idx <- matrix(sample.int(n_case, n_case * count, replace = TRUE), n_case, count)
    output[start:(start + count - 1L)] <- colMeans(matrix(case_gain[idx], n_case, count))
    start <- start + count
  }
  output
}

train <- read.csv(file.path("csv files", "train.csv"))
train <- train[order(train$No), , drop = FALSE]
rownames(train) <- NULL
truth_all <- as.matrix(train[, paste0("Ch", 1:4), drop = FALSE])
stopifnot(nrow(train) == 21565L, all(rowSums(truth_all) == 1L))

## ---- step 1: choose alpha using the canonical seed only ----
canonical <- readRDS(file.path(output_dir, "canonical_result.rds"))
p0_canonical <- canonical$oof_baseline # m8trpg-alone OOF
p1_canonical <- canonical$oof_relative # m8trpg + relative-gap-terms OOF
baseline_loss_canonical <- log_loss_matrix(truth_all, p0_canonical)

cat("Choosing alpha on the canonical seed only:\n")
canonical_losses <- vapply(alpha_grid, function(alpha) {
  p_alpha <- (1 - alpha) * p0_canonical + alpha * p1_canonical
  log_loss_matrix(truth_all, p_alpha)
}, numeric(1))
print(data.frame(alpha = alpha_grid, canonical_logloss = canonical_losses,
                  gain_vs_baseline = baseline_loss_canonical - canonical_losses))
best_alpha <- alpha_grid[[which.min(canonical_losses)]]
cat(sprintf("\nFrozen alpha (chosen on canonical seed only): %.2f\n", best_alpha))

## ---- step 2: evaluate the frozen alpha out-of-sample on the other 5 seeds ----
case_gain_by_seed <- list()
seed_summary_rows <- list()
for (seed in additional_seeds) {
  seed_name <- as.character(seed)
  repeat_cache <- readRDS(file.path("data_processed", "codex_repeat_cv", sprintf("repeat_seed_%d.rds", seed)))
  p0_seed <- repeat_cache$components$mlogit
  p1_seed <- readRDS(file.path(output_dir, sprintf("relative_oof_seed_%d.rds", seed)))
  p_alpha_seed <- (1 - best_alpha) * p0_seed + best_alpha * p1_seed

  baseline_loss <- log_loss_matrix(truth_all, p0_seed)
  candidate_loss <- log_loss_matrix(truth_all, p_alpha_seed)
  case_gain <- tapply(row_log_loss(truth_all, p0_seed) - row_log_loss(truth_all, p_alpha_seed), train$Case, mean)
  case_gain <- case_gain[order(as.integer(names(case_gain)))]
  case_gain_by_seed[[seed_name]] <- case_gain
  seed_summary_rows[[seed_name]] <- data.frame(
    seed = seed, baseline_logloss = baseline_loss, candidate_logloss = candidate_loss,
    gain = baseline_loss - candidate_loss
  )
  cat(sprintf("  seed %d (out-of-sample): baseline %.6f, alpha=%.2f candidate %.6f, gain %.6f\n",
              seed, baseline_loss, best_alpha, candidate_loss, baseline_loss - candidate_loss))
}
seed_summary <- do.call(rbind, seed_summary_rows)

case_gain_matrix <- do.call(cbind, case_gain_by_seed)
average_case_gain <- rowMeans(case_gain_matrix)

n_boot <- 100000L
boot <- bootstrap_case_means(average_case_gain, n_boot, seed = 4821)
result <- data.frame(
  experiment = "relative_price_gap_shrinkage", stage = "out_of_sample_5_seed",
  frozen_alpha = best_alpha,
  point_gain = mean(average_case_gain), bootstrap_mean = mean(boot), bootstrap_sd = sd(boot),
  lower_95 = unname(quantile(boot, 0.025)), upper_95 = unname(quantile(boot, 0.975)),
  win_rate = mean(boot > 0), n_boot = n_boot,
  positive_seeds = sum(seed_summary$gain > 0), n_seeds = nrow(seed_summary)
)
cat("\nOut-of-sample (5-seed) pooled result, alpha frozen from the canonical seed alone:\n")
print(result, digits = 9)

write.csv(seed_summary, file.path(output_dir, "shrinkage_by_seed.csv"), row.names = FALSE)
write.csv(result, file.path(output_dir, "shrinkage_summary.csv"), row.names = FALSE)
saveRDS(
  list(best_alpha = best_alpha, canonical_losses = canonical_losses, seed_summary = seed_summary,
       case_gain_matrix = case_gain_matrix, average_case_gain = average_case_gain, bootstrap = boot, result = result),
  file.path(output_dir, "shrinkage_result.rds")
)
cat("\nDone.\n")
