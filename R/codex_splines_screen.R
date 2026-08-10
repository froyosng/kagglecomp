# Stage 1: single-split (seed 7402) screen for the smooth-spline covariate
# candidate family. Pre-registered in codex_splines_preregister.md BEFORE
# this script was run (verify via `git log --format="%h %ci %s"`).
#
# Screen-then-freeze rule (already established in this project, see
# R/codex_triple_interactions.R's stage=="screen"): only candidates whose
# single-split validation log loss beats the current best's screen number
# (mlogit_m8trpg baseline, verified below to reproduce 1.15968144721113)
# proceed to canonical 5-fold CV.

source("R/codex_splines_common.R")

output_dir <- "data_processed/codex_splines"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

split <- readRDS("data_processed/train_val_split.rds")
tr_long <- prepare_saved_long(split$train_long_tr)
va_long <- prepare_saved_long(split$train_long_val)
va_wide <- split$train_wide_val
va_wide <- va_wide[order(va_wide$No), , drop = FALSE]
va_truth <- as.matrix(va_wide[, paste0("Ch", 1:4)])

cat("Screen: train respondents =", length(unique(tr_long$Case)),
    " validation respondents =", length(unique(va_long$Case)), "\n")

baseline_fit <- fit_predict_m8trpg(tr_long, va_long, case_weights = NULL, attr_rank_scope = "none")
baseline_pred <- baseline_fit$pred[match(va_wide$No, baseline_fit$no), , drop = FALSE]
baseline_loss <- log_loss_matrix(va_truth, baseline_pred)
stopifnot(abs(baseline_loss - 1.15968144721113) < 1e-8)
cat(sprintf("baseline m8trpg (verified): %.11f\n", baseline_loss))

results <- list()
predictions <- list()
coefficients <- list()

for (candidate in spline_candidate_names) {
  cat("\n--- Screening", candidate, "---\n")
  fitted <- tryCatch(fit_predict_spline(tr_long, va_long, candidate), error = function(e) e)
  if (inherits(fitted, "error")) {
    results[[candidate]] <- data.frame(
      candidate = candidate, status = "fit_failed",
      logloss = NA_real_, delta_vs_base = NA_real_,
      max_abs_extra_coef = NA_real_, passes_screen = FALSE,
      detail = conditionMessage(fitted)
    )
    cat("FAILED:", conditionMessage(fitted), "\n")
    next
  }
  idx <- match(va_wide$No, fitted$no)
  pred <- fitted$pred[idx, , drop = FALSE]
  loss <- log_loss_matrix(va_truth, pred)
  passes <- loss < baseline_loss
  results[[candidate]] <- data.frame(
    candidate = candidate, status = "ok",
    logloss = loss, delta_vs_base = loss - baseline_loss,
    max_abs_extra_coef = max(abs(fitted$extra_coef)),
    passes_screen = passes,
    detail = "natural cubic spline (ns), knots on training respondents only"
  )
  predictions[[candidate]] <- pred
  coefficients[[candidate]] <- data.frame(
    candidate = candidate, term = names(fitted$extra_coef),
    coefficient = unname(fitted$extra_coef)
  )
  cat(sprintf("%s: logloss %.6f; delta %+.6f vs base; passes_screen=%s; max|extra coef|=%.4f\n",
              candidate, loss, loss - baseline_loss, passes, max(abs(fitted$extra_coef))))
}

result <- do.call(rbind, results)
coef_result <- do.call(rbind, coefficients)
write.csv(result, file.path(output_dir, "splines_screen.csv"), row.names = FALSE)
write.csv(coef_result, file.path(output_dir, "splines_screen_coefficients.csv"), row.names = FALSE)
saveRDS(
  list(result = result, predictions = predictions, baseline_pred = baseline_pred,
       baseline_loss = baseline_loss, validation_no = va_wide$No),
  file.path(output_dir, "splines_screen.rds")
)

cat("\n=== Screen summary ===\n")
print(result, digits = 8, row.names = FALSE)
cat("\nCandidates proceeding to canonical CV:\n")
print(result$candidate[isTRUE(result$status == "ok") & result$passes_screen])
