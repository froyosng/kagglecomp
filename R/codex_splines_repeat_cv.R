# Stage 3: repeated CV escalation, for any candidate whose canonical-CV
# bootstrap CI (R/codex_splines_cv.R) was not a decisive rejection, per the
# escalation rule fixed in codex_splines_preregister.md BEFORE any CV result
# was seen.
#
# Reuses the project's existing 5 additional respondent-grouped fold
# assignments (seeds 1907/2719/6151/8293/104729) and the cached
# original_xgb/shallow_mlp/baseline-mlogit components already saved per
# seed/fold in data_processed/codex_repeat_cv/checkpoints/ from the
# project's own prior repeated-CV round -- only the candidate spline mlogit
# component is refit here (25 new fits: 5 seeds x 5 folds). The canonical
# seed's (4821) contribution reuses the already-computed Stage-2 CV output
# directly (R/codex_splines_cv.R's splines_cv.rds), not refit again.

source("R/codex_splines_common.R")

output_dir <- "data_processed/codex_splines"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

decision <- read.csv(file.path(output_dir, "splines_cv_decision.csv"))
candidates <- decision$candidate[decision$escalate_to_repeated_cv]
cat("Candidates escalating to repeated CV:", paste(candidates, collapse = ", "), "\n")
if (length(candidates) == 0L) {
  cat("No candidate needs repeated CV; all canonical results were decisive.\n")
  quit(save = "no", status = 0)
}

train <- read.csv("csv files/train.csv")
train <- train[order(train$No), , drop = FALSE]
stopifnot(identical(train$No, seq_len(nrow(train))))
truth <- as.matrix(train[, paste0("Ch", 1:4), drop = FALSE])
train_long <- reshape_choice_long(train)
cases <- unique(train$Case)
stopifnot(length(cases) == 1135L)

repeat_additional_seeds <- c(1907L, 2719L, 6151L, 8293L, 104729L)
repeat_all_seeds <- c(4821L, repeat_additional_seeds)
n_repeats <- length(repeat_all_seeds)

repeat_fold_map <- function(cases, seed) {
  cases <- as.integer(cases)
  set.seed(seed)
  fold <- sample(rep(1:5, length.out = length(cases)))
  names(fold) <- as.character(cases)
  stopifnot(length(fold) == length(cases), all(table(fold) == 227L))
  fold
}

## ---- Load canonical-seed (4821) results already computed at Stage 2 ------
canonical_cv <- readRDS(file.path(output_dir, "splines_cv.rds"))

## ---- Verify: fold-1 checkpoints for seed 1907 give the SAME held-out row
## set this script's own repeat_fold_map()/reshape reconstructs, before
## trusting anything downstream (matches this project's established
## verification discipline, e.g. codex_price_history_only.R's Validation A/B). ----
verify_checkpoint_alignment <- function(seed) {
  fold_map <- repeat_fold_map(cases, seed)
  val_cases_1 <- as.integer(names(fold_map)[fold_map == 1])
  target_no <- train$No[train$Case %in% val_cases_1]
  checkpoint <- readRDS(sprintf(
    "data_processed/codex_repeat_cv/checkpoints/seed_%d_fold_1.rds", seed
  ))
  ok <- identical(sort(as.integer(checkpoint$validation_no)), sort(as.integer(target_no)))
  cat(sprintf("Checkpoint alignment check, seed %d fold 1: %s (n=%d vs n=%d)\n",
              seed, ifelse(ok, "OK", "MISMATCH"), length(checkpoint$validation_no), length(target_no)))
  stopifnot(ok)
}
for (seed in repeat_additional_seeds) verify_checkpoint_alignment(seed)
cat("All 5 additional-seed fold-1 checkpoints align with this script's own fold reconstruction.\n\n")

## ---- Repeated CV: refit candidate spline mlogit only, per seed/fold -------

repeat_case_gain <- list()
for (candidate in candidates) {
  repeat_case_gain[[candidate]] <- matrix(
    NA_real_, nrow = length(cases), ncol = n_repeats,
    dimnames = list(as.character(cases), as.character(repeat_all_seeds))
  )
  # Fill in the canonical seed's (4821) per-respondent gain from Stage 2.
  repeat_case_gain[[candidate]][, "4821"] <- canonical_cv$case_gain[[candidate]]
}

fold_rows <- list()
checkpoint_progress <- file.path(output_dir, "repeat_cv_progress.rds")
completed_seeds <- integer(0)
if (file.exists(checkpoint_progress)) {
  saved <- readRDS(checkpoint_progress)
  repeat_case_gain <- saved$repeat_case_gain
  fold_rows <- saved$fold_rows
  completed_seeds <- saved$completed_seeds
  cat("Resumed from checkpoint; completed additional seeds:",
      paste(completed_seeds, collapse = ", "), "\n")
}

for (seed in repeat_additional_seeds) {
  if (seed %in% completed_seeds) {
    cat("Seed", seed, "already completed, skipping.\n")
    next
  }
  cat("\n=== Repeated CV, seed", seed, "===\n")
  fold_map <- repeat_fold_map(cases, seed)
  for (fold in 1:5) {
    val_cases <- as.integer(names(fold_map)[fold_map == fold])
    tr_long_fold <- train_long[!(train_long$Case %in% val_cases), , drop = FALSE]
    va_long_fold <- train_long[train_long$Case %in% val_cases, , drop = FALSE]
    target_rows <- train$Case %in% val_cases
    target_no <- train$No[target_rows]

    checkpoint <- readRDS(sprintf(
      "data_processed/codex_repeat_cv/checkpoints/seed_%d_fold_%d.rds", seed, fold
    ))
    stopifnot(identical(sort(as.integer(checkpoint$validation_no)), sort(as.integer(target_no))))
    order_in_checkpoint <- match(target_no, checkpoint$validation_no)
    xgb_component <- checkpoint$predictions$original_xgb[order_in_checkpoint, , drop = FALSE]
    mlp_component <- checkpoint$predictions$shallow_mlp[order_in_checkpoint, , drop = FALSE]
    mlogit_component <- checkpoint$predictions$mlogit[order_in_checkpoint, , drop = FALSE]
    baseline_fold <- 0.85 * (0.8 * mlogit_component + 0.2 * xgb_component) + 0.15 * mlp_component
    # train is sorted by No and target_rows is a logical mask, so
    # truth[target_rows,] is already in ascending target_no order.
    fold_truth <- truth[target_rows, , drop = FALSE]

    for (candidate in candidates) {
      fitted <- fit_predict_spline(tr_long_fold, va_long_fold, candidate)
      idx_in_fitted <- match(target_no, fitted$no)
      candidate_mlogit <- fitted$pred[idx_in_fitted, , drop = FALSE]
      candidate_fold <- 0.85 * (0.8 * candidate_mlogit + 0.2 * xgb_component) + 0.15 * mlp_component

      fold_rows[[length(fold_rows) + 1L]] <- data.frame(
        candidate = candidate, repeat_seed = seed, fold = fold,
        baseline_loss = log_loss_matrix(fold_truth, baseline_fold),
        candidate_loss = log_loss_matrix(fold_truth, candidate_fold),
        gain = log_loss_matrix(fold_truth, baseline_fold) -
          log_loss_matrix(fold_truth, candidate_fold),
        max_abs_extra_coef = max(abs(fitted$extra_coef))
      )
      row_gain <- row_loss(fold_truth, baseline_fold) - row_loss(fold_truth, candidate_fold)
      case_gain_fold <- tapply(row_gain, train$Case[target_rows], mean)
      repeat_case_gain[[candidate]][names(case_gain_fold), as.character(seed)] <- case_gain_fold
      cat(sprintf("  %s seed %d fold %d: baseline %.6f; candidate %.6f; gain %+.6f\n",
                  candidate, seed, fold,
                  fold_rows[[length(fold_rows)]]$baseline_loss,
                  fold_rows[[length(fold_rows)]]$candidate_loss,
                  fold_rows[[length(fold_rows)]]$gain))
    }
  }
  completed_seeds <- c(completed_seeds, seed)
  saveRDS(
    list(repeat_case_gain = repeat_case_gain, fold_rows = fold_rows, completed_seeds = completed_seeds),
    checkpoint_progress
  )
}

for (candidate in candidates) stopifnot(!anyNA(repeat_case_gain[[candidate]]))

fold_result <- do.call(rbind, fold_rows)
write.csv(fold_result, file.path(output_dir, "splines_repeat_cv_folds.csv"), row.names = FALSE)

## ---- Summaries + bootstrap over the pooled (across-repeat) per-respondent
## gain, matching R/codex_repeat_cv_common.R's repeat_bootstrap_average_gain
## exactly (average the 6 repeat-level gains within respondent first, then
## bootstrap those 1,135 respondent-level values). ----

repeat_rows <- list()
bootstrap_rows <- list()
decision_rows <- list()
for (candidate in candidates) {
  cg <- repeat_case_gain[[candidate]]
  for (seed in repeat_all_seeds) {
    repeat_rows[[length(repeat_rows) + 1L]] <- data.frame(
      candidate = candidate, repeat_seed = seed, mean_case_gain = mean(cg[, as.character(seed)])
    )
  }
  boot <- spline_bootstrap_gain(cg, n_boot = 100000L, seed = 4821L)
  bootstrap_rows[[length(bootstrap_rows) + 1L]] <- cbind(data.frame(candidate = candidate), boot$summary)

  positive_repeats <- sum(colMeans(cg) > 0)
  candidate_folds <- fold_result[fold_result$candidate == candidate, ]
  # canonical-seed fold-level rows come from Stage 2's splines_cv_folds.csv
  canonical_folds <- read.csv(file.path(output_dir, "splines_cv_folds.csv"))
  canonical_folds <- canonical_folds[canonical_folds$candidate == candidate, ]
  n_total_folds <- nrow(candidate_folds) + nrow(canonical_folds)

  decision_rows[[candidate]] <- data.frame(
    candidate = candidate,
    point_gain = boot$summary$point_gain,
    lower_95 = boot$summary$lower_95,
    upper_95 = boot$summary$upper_95,
    win_rate = boot$summary$win_rate,
    positive_repeats = positive_repeats,
    total_repeats = n_repeats,
    total_folds = n_total_folds,
    promote = (boot$summary$point_gain > 0 && boot$summary$lower_95 > 0)
  )
}
repeat_summary <- do.call(rbind, repeat_rows)
bootstrap_summary <- do.call(rbind, bootstrap_rows)
decision <- do.call(rbind, decision_rows)

write.csv(repeat_summary, file.path(output_dir, "splines_repeat_cv_summary.csv"), row.names = FALSE)
write.csv(bootstrap_summary, file.path(output_dir, "splines_repeat_cv_bootstrap.csv"), row.names = FALSE)
write.csv(decision, file.path(output_dir, "splines_repeat_cv_decision.csv"), row.names = FALSE)
saveRDS(
  list(fold_result = fold_result, repeat_summary = repeat_summary,
       bootstrap_summary = bootstrap_summary, decision = decision,
       repeat_case_gain = repeat_case_gain),
  file.path(output_dir, "splines_repeat_cv.rds")
)

cat("\n=== Repeated CV per-repeat summary ===\n")
print(repeat_summary, digits = 8, row.names = FALSE)
cat("\n=== Repeated CV bootstrap (pooled across 6 fold assignments) ===\n")
print(bootstrap_summary, digits = 9, row.names = FALSE)
cat("\n=== Final decision ===\n")
print(decision, digits = 8, row.names = FALSE)
