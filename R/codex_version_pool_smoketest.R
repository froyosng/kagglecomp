## Smoke test (Section 7 of codex_version_pool_preregister.md). Two checks:
##
## 1. Neighbor graph sanity: k=0 gives identity weights; k>0 gives exactly k
##    off-diagonal weights per row, all in (0,1], decreasing in distance.
## 2. The k=0 special case of the FULL nested pipeline (grid restricted to
##    k=0, same 8-value lambda grid) must reproduce the rejected experiment's
##    own published numbers: per-fold selected lambda {160,Inf,Inf,160,160},
##    per-fold outer gains vs the exact fixed OOF baseline
##    {+0.0000865,0,0,-0.0000591,-0.0008852}, pooled gain -0.000172 with 95% CI
##    approximately [-0.0008, +0.00046] (codex_version_shrinkage_findings.md).
##    If this does not reproduce, the new pooling machinery has a bug and no
##    k>0 result should be trusted.

source("R/codex_version_pool_driver.R")

cat("=== Check 1: neighbor graph sanity ===\n")
version_map <- load_version_map()
design_raw <- build_version_design_matrix(version_map)
design_std <- standardize_columns(design_raw)

W0 <- build_weight_matrix(design_std, 0)
stopifnot(identical(W0, diag(299)))
cat("k=0 -> identity: OK\n")

for (k_check in c(5, 10, 20, 40)) {
  Wk <- build_weight_matrix(design_std, k_check)
  offdiag_nonzero <- rowSums(Wk > 0) - 1L  # exclude the guaranteed self=1
  stopifnot(all(offdiag_nonzero == k_check))
  stopifnot(all(diag(Wk) == 1))
  stopifnot(all(Wk >= 0 & Wk <= 1))
  # weight should be non-increasing as distance increases, for a sample of rows
  d <- as.matrix(dist(design_std))
  ok_monotone <- vapply(sample(seq_len(299), 15), function(v) {
    nn <- which(Wk[v, ] > 0 & seq_len(299) != v)
    dv <- d[v, nn]
    wv <- Wk[v, nn]
    all(diff(wv[order(dv)]) <= 1e-12)
  }, logical(1))
  stopifnot(all(ok_monotone))
  cat(sprintf("k=%d -> exactly %d neighbors/row, weights in [0,1], monotone in distance: OK\n",
              k_check, k_check))
}

cat("\n=== Check 2: k=0 special case reproduces the rejected experiment ===\n")
result <- run_version_pool_cv(
  k_grid = 0L,
  lambda_grid = c(5, 10, 20, 40, 80, 160, 320, Inf),
  out_dir = "data_processed/codex_version_pool",
  tag = "smoketest_k0"
)

print(result$fold_summary[, c("outer_fold", "k_star", "lambda_star", "primary_gain")])
cat("\noverall:\n")
print(result$overall[, c("primary_baseline_loss","primary_corrected_loss","primary_gain",
                          "primary_boot_lower95","primary_boot_upper95")])

published_lambda <- c(160, Inf, Inf, 160, 160)
published_gain <- c(0.0000865, 0, 0, -0.0000591, -0.0008852)
lambda_match <- mapply(function(a, b) (is.infinite(a) && is.infinite(b)) || isTRUE(all.equal(a, b)),
                        result$fold_summary$lambda_star, published_lambda)
gain_diff <- abs(result$fold_summary$primary_gain - published_gain)
cat("\nlambda_star matches published per-fold selection:", all(lambda_match), "\n")
cat("max abs diff vs published per-fold gains:", max(gain_diff), "\n")
cat("published pooled gain -0.000172; observed pooled gain:", result$overall$primary_gain, "\n")

if (!all(lambda_match) || max(gain_diff) > 1e-4) {
  stop("SMOKE TEST FAILED: k=0 special case does not reproduce the rejected experiment. Fix before trusting k>0 results.")
} else {
  cat("\nSMOKE TEST PASSED: k=0 special case reproduces the rejected experiment's numbers.\n")
}
