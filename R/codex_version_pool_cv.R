## Real evaluation (Section 5 of codex_version_pool_preregister.md). Only run
## after the smoke test (R/codex_version_pool_smoketest.R) passed.

source("R/codex_version_pool_driver.R")

result <- run_version_pool_cv(
  k_grid = c(0L, 5L, 10L, 20L, 40L),
  lambda_grid = c(5, 10, 20, 40, 80, 160, 320, Inf),
  out_dir = "data_processed/codex_version_pool",
  tag = "full"
)

cat("\n=== Fold summary ===\n")
print(result$fold_summary[, c("outer_fold","k_star","lambda_star",
                               "primary_gain","secondary_gain")])

cat("\n=== Overall (primary = vs exact fixed submitted OOF, current best) ===\n")
print(result$overall)

cat("\n=== Bootstrap (primary) ===\n")
print(result$bootstrap_primary)

cat("\n=== Bootstrap (secondary) ===\n")
print(result$bootstrap_secondary)

promoted <- result$bootstrap_primary$lower_95 > 0
cat(sprintf("\nPROMOTION BAR (95%% CI lower bound > 0 vs current best): %s\n",
            ifelse(promoted, "CLEARED", "NOT CLEARED")))
