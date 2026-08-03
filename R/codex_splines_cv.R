# Stage 2: canonical respondent-grouped 5-fold CV (seed 4821,
# oof_ensemble_v10.rds$fold_of_case) for whichever candidates passed the
# Stage-1 single-split screen (R/codex_splines_screen.R). Pre-registered in
# codex_splines_preregister.md before this script was run.
#
# Refits the candidate spline mlogit component from scratch in each fold
# (fresh z-scaler and fresh ns() knots from that fold's 908 training
# respondents only); original_xgb and shallow_mlp are reused byte-for-byte
# from the already-cached canonical OOF matrices, exactly as pre-registered.

source("R/codex_splines_common.R")

output_dir <- "data_processed/codex_splines"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

screen <- read.csv(file.path(output_dir, "splines_screen.csv"))
candidates <- screen$candidate[screen$status == "ok" & screen$passes_screen]
cat("Candidates passing Stage 1 screen, proceeding to canonical CV:",
    paste(candidates, collapse = ", "), "\n")
if (length(candidates) == 0L) {
  cat("No candidate passed the screen; canonical CV correctly skipped.\n")
  quit(save = "no", status = 0)
}

train <- read.csv("csv files/train.csv")
train <- train[order(train$No), , drop = FALSE]
stopifnot(identical(train$No, seq_len(nrow(train))))
truth <- as.matrix(train[, paste0("Ch", 1:4), drop = FALSE])
train_long <- reshape_choice_long(train)

base <- readRDS("data_processed/oof_ensemble_v10.rds")
fold_map <- base$fold_of_case
stopifnot(length(fold_map) == 1135L, all(table(fold_map) == 227L))
baseline_mlogit <- base$oof_mlogit
xgb_oof <- base$oof_xgb
mlp <- readRDS("data_processed/codex_behavioral_round/mlp_oof.rds")
shallow_mlp_oof <- mlp$oof[["h08_d0.100"]]

baseline_mlogit_loss <- log_loss_matrix(truth, baseline_mlogit)
stopifnot(abs(baseline_mlogit_loss - 1.147021211) < 1e-6)
v11 <- 0.8 * baseline_mlogit + 0.2 * xgb_oof
v11_loss <- log_loss_matrix(truth, v11)
stopifnot(abs(v11_loss - 1.145094213) < 1e-6)
current_best <- 0.85 * v11 + 0.15 * shallow_mlp_oof
current_best_loss <- log_loss_matrix(truth, current_best)
stopifnot(abs(current_best_loss - 1.143686618134879) < 1e-8)
cat(sprintf("Verified current-best CV logloss: %.9f\n", current_best_loss))

candidate_oof <- lapply(candidates, function(x) matrix(NA_real_, nrow(train), 4L))
names(candidate_oof) <- candidates
fold_rows <- list()
coef_rows <- list()

for (candidate in candidates) {
  cat("\n=== Canonical CV:", candidate, "===\n")
  for (fold in 1:5) {
    val_cases <- as.integer(names(fold_map)[fold_map == fold])
    tr_long_fold <- train_long[!(train_long$Case %in% val_cases), , drop = FALSE]
    va_long_fold <- train_long[train_long$Case %in% val_cases, , drop = FALSE]

    fitted <- fit_predict_spline(tr_long_fold, va_long_fold, candidate)
    idx <- match(fitted$no, train$No)
    candidate_oof[[candidate]][idx, ] <- fitted$pred

    fold_truth <- truth[idx, , drop = FALSE]
    fold_baseline <- baseline_mlogit[idx, , drop = FALSE]
    fold_candidate <- fitted$pred
    fold_rows[[length(fold_rows) + 1L]] <- data.frame(
      candidate = candidate, fold = fold,
      baseline_mlogit_logloss = log_loss_matrix(fold_truth, fold_baseline),
      candidate_mlogit_logloss = log_loss_matrix(fold_truth, fold_candidate),
      mlogit_gain = log_loss_matrix(fold_truth, fold_baseline) -
        log_loss_matrix(fold_truth, fold_candidate),
      max_abs_extra_coef = max(abs(fitted$extra_coef))
    )
    coef_rows[[length(coef_rows) + 1L]] <- data.frame(
      candidate = candidate, fold = fold,
      term = names(fitted$extra_coef), coefficient = unname(fitted$extra_coef)
    )
    cat(sprintf("  fold %d: baseline mlogit %.6f; candidate mlogit %.6f; gain %+.6f\n",
                fold, fold_rows[[length(fold_rows)]]$baseline_mlogit_logloss,
                fold_rows[[length(fold_rows)]]$candidate_mlogit_logloss,
                fold_rows[[length(fold_rows)]]$mlogit_gain))
  }
  stopifnot(!anyNA(candidate_oof[[candidate]]))
}

fold_result <- do.call(rbind, fold_rows)
coef_result <- do.call(rbind, coef_rows)
write.csv(fold_result, file.path(output_dir, "splines_cv_folds.csv"), row.names = FALSE)
write.csv(coef_result, file.path(output_dir, "splines_cv_coefficients.csv"), row.names = FALSE)

## ---- Ensemble-level comparison + respondent-clustered bootstrap ----------

summary_rows <- list()
bootstrap_rows <- list()
case_gain_store <- list()

for (candidate in candidates) {
  pred <- candidate_oof[[candidate]]
  candidate_mlogit_loss <- log_loss_matrix(truth, pred)
  candidate_blend <- 0.85 * (0.8 * pred + 0.2 * xgb_oof) + 0.15 * shallow_mlp_oof
  candidate_blend_loss <- log_loss_matrix(truth, candidate_blend)

  summary_rows[[length(summary_rows) + 1L]] <- data.frame(
    candidate = candidate,
    baseline_mlogit_logloss = baseline_mlogit_loss,
    candidate_mlogit_logloss = candidate_mlogit_loss,
    mlogit_gain = baseline_mlogit_loss - candidate_mlogit_loss,
    current_best_logloss = current_best_loss,
    candidate_blend_logloss = candidate_blend_loss,
    blend_gain = current_best_loss - candidate_blend_loss
  )

  case_gain <- case_gain_vector(truth, current_best, candidate_blend, train$Case)
  case_gain_store[[candidate]] <- case_gain
  boot <- spline_bootstrap_gain(matrix(case_gain, ncol = 1), n_boot = 100000L, seed = 4821L)
  bootstrap_rows[[length(bootstrap_rows) + 1L]] <- cbind(
    data.frame(candidate = candidate), boot$summary
  )
  cat(sprintf(
    "\n%s: mlogit-alone gain %+.6f; blend gain %+.6f; bootstrap 95%% CI [%+.6f, %+.6f]; win_rate=%.4f\n",
    candidate, baseline_mlogit_loss - candidate_mlogit_loss,
    current_best_loss - candidate_blend_loss,
    boot$summary$lower_95, boot$summary$upper_95, boot$summary$win_rate
  ))
}

summary_result <- do.call(rbind, summary_rows)
bootstrap_result <- do.call(rbind, bootstrap_rows)
write.csv(summary_result, file.path(output_dir, "splines_cv_summary.csv"), row.names = FALSE)
write.csv(bootstrap_result, file.path(output_dir, "splines_cv_bootstrap.csv"), row.names = FALSE)
saveRDS(
  list(candidates = candidates, oof = candidate_oof, summary = summary_result,
       bootstrap = bootstrap_result, fold_result = fold_result,
       case_gain = case_gain_store, truth = truth, current_best = current_best,
       xgb_oof = xgb_oof, shallow_mlp_oof = shallow_mlp_oof),
  file.path(output_dir, "splines_cv.rds")
)

cat("\n=== Canonical CV summary ===\n")
print(summary_result, digits = 9, row.names = FALSE)
cat("\n=== Bootstrap summary ===\n")
print(bootstrap_result, digits = 9, row.names = FALSE)

## ---- Decision per pre-registered escalation rule --------------------------
decision_rows <- list()
for (candidate in candidates) {
  b <- bootstrap_result[bootstrap_result$candidate == candidate, ]
  decisive_reject <- b$upper_95 < -0.0003   # comfortably below zero, no ambiguity
  needs_repeat <- !decisive_reject
  decision_rows[[candidate]] <- data.frame(
    candidate = candidate, point_gain = b$point_gain,
    lower_95 = b$lower_95, upper_95 = b$upper_95,
    decisive_canonical_reject = decisive_reject,
    escalate_to_repeated_cv = needs_repeat
  )
}
decision <- do.call(rbind, decision_rows)
write.csv(decision, file.path(output_dir, "splines_cv_decision.csv"), row.names = FALSE)
cat("\n=== Escalation decision ===\n")
print(decision, digits = 8, row.names = FALSE)
