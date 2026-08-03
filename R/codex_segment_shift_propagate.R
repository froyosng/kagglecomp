# Propagate the promoted segment-shift pruning fix (In_seg3, In_seg5 dropped
# from the dominant mlogit component) through the full v14 ensemble. Uses
# the fixed 0.80/0.20 mlogit/xgb and 0.85/0.15 +MLP weights (matching this
# project's established propagation convention of not re-optimizing outer
# blend weights when swapping in an improved base component), but the
# set-context layer uses the SAME PER-FOLD weights crossfit_blend() already
# selected for the official v14 CV result (not the single fixed 0.111,
# which is only used in the separate full-data test-submission build) --
# verified by an exact reproduction check against the official OOF first.
#
# Run:
#   source("R/codex_segment_shift_propagate.R")

output_dir <- file.path("data_processed", "codex_segment_shift")
canonical_seed <- 4821L
additional_seeds <- c(1907L, 2719L, 6151L, 8293L, 104729L)
all_seeds <- c(canonical_seed, additional_seeds)

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
summarize_gain <- function(case_gain, replicates, seed) {
  boot <- bootstrap_case_means(case_gain, replicates, seed)
  data.frame(
    point_gain = mean(case_gain), bootstrap_mean = mean(boot), bootstrap_sd = sd(boot),
    lower_95 = unname(quantile(boot, 0.025)), upper_95 = unname(quantile(boot, 0.975)),
    win_rate = mean(boot > 0), n_boot = replicates
  )
}

train <- read.csv(file.path("csv files", "train.csv"))
train <- train[order(train$No), , drop = FALSE]
rownames(train) <- NULL
truth_all <- as.matrix(train[, paste0("Ch", 1:4), drop = FALSE])
stopifnot(nrow(train) == 21565L, all(rowSums(truth_all) == 1L))

v14_dir <- file.path("data_processed", "codex_set_context_network")
segshift <- readRDS(file.path(output_dir, "result.rds"))
segshift_repeated <- readRDS(file.path(output_dir, "repeated_cv_result.rds"))

## helper: load everything needed for one seed and build both the ORIGINAL
## and PRUNED full v14-stack OOF prediction. The set-context blend uses
## PER-FOLD weights from crossfit_blend()'s own grid search (stored in
## v14_seed$weights), not the single fixed 0.111 used only in the separate
## full-data test-submission build -- reconstructed exactly the way
## codex_two_head_ensemble_v2.R's reconstruct_v14() already verified.
reconstruct_v14_blend <- function(baseline, set_context, row_fold, saved_weights) {
  weight_by_fold <- saved_weights$set_context_weight[match(1:5, saved_weights$fold)]
  stopifnot(length(weight_by_fold) == 5L, !anyNA(weight_by_fold))
  row_weight <- weight_by_fold[row_fold]
  blended <- (1 - row_weight) * baseline + row_weight * set_context
  blended / rowSums(blended)
}

build_v14_variant <- function(seed, pruned_mlogit_oof) {
  base_saved <- if (seed == canonical_seed) {
    readRDS(file.path("data_processed", "oof_ensemble_v10.rds"))
  } else {
    NULL
  }
  repeat_cache <- if (seed != canonical_seed) {
    readRDS(file.path("data_processed", "codex_repeat_cv", sprintf("repeat_seed_%d.rds", seed)))
  } else {
    NULL
  }
  if (seed == canonical_seed) {
    original_mlogit <- base_saved$oof_mlogit
    xgb <- base_saved$oof_xgb
  } else {
    original_mlogit <- repeat_cache$components$mlogit
    xgb <- repeat_cache$components$original_xgb
  }
  shallow_mlp <- if (seed == canonical_seed) {
    readRDS(file.path("data_processed", "codex_behavioral_round", "mlp_oof.rds"))$oof[["h08_d0.100"]]
  } else {
    repeat_cache$components$shallow_mlp
  }

  v14_seed <- if (seed == canonical_seed) {
    readRDS(file.path(v14_dir, "canonical_result.rds"))
  } else {
    readRDS(file.path(v14_dir, sprintf("repeat_result_%d.rds", seed)))
  }
  set_context <- v14_seed$set_context_prediction
  fold_map <- v14_seed$fold_map
  row_fold <- unname(fold_map[as.character(train$Case)])
  stopifnot(!anyNA(row_fold))

  make_stack <- function(mlogit_oof) {
    ensemble_v11 <- 0.80 * mlogit_oof + 0.20 * xgb
    with_mlp <- 0.85 * ensemble_v11 + 0.15 * shallow_mlp
    reconstruct_v14_blend(with_mlp, set_context, row_fold, v14_seed$weights)
  }

  list(
    original = make_stack(original_mlogit),
    pruned = make_stack(pruned_mlogit_oof),
    v14_reference = v14_seed$candidate_prediction
  )
}

## canonical seed
canonical_stack <- build_v14_variant(canonical_seed, segshift$oof_pruned)
reproduction_diff <- max(abs(canonical_stack$original - canonical_stack$v14_reference))
cat(sprintf("Reproduction check: max|reconstructed_v14 - official_v14_OOF| = %.3e (should be ~0)\n", reproduction_diff))
stopifnot(reproduction_diff < 1e-8)

case_gain_matrix <- matrix(
  NA_real_, length(unique(train$Case)), length(all_seeds),
  dimnames = list(as.character(sort(unique(train$Case))), as.character(all_seeds))
)
gain_canonical <- row_log_loss(truth_all, canonical_stack$original) - row_log_loss(truth_all, canonical_stack$pruned)
case_gain_canonical <- tapply(gain_canonical, train$Case, mean)
case_gain_matrix[, as.character(canonical_seed)] <- case_gain_canonical[rownames(case_gain_matrix)]

cat(sprintf(
  "\nSeed %d: full v14-stack logloss original %.6f vs pruned %.6f, gain %.6f\n",
  canonical_seed, log_loss_matrix(truth_all, canonical_stack$original),
  log_loss_matrix(truth_all, canonical_stack$pruned),
  log_loss_matrix(truth_all, canonical_stack$original) - log_loss_matrix(truth_all, canonical_stack$pruned)
))

seed_summary_rows <- list(`4821` = data.frame(
  seed = canonical_seed,
  original_logloss = log_loss_matrix(truth_all, canonical_stack$original),
  pruned_logloss = log_loss_matrix(truth_all, canonical_stack$pruned),
  gain = log_loss_matrix(truth_all, canonical_stack$original) - log_loss_matrix(truth_all, canonical_stack$pruned)
))

for (seed in additional_seeds) {
  seed_name <- as.character(seed)
  pruned_oof_seed <- readRDS(file.path(output_dir, sprintf("pruned_oof_seed_%d.rds", seed)))
  stack <- build_v14_variant(seed, pruned_oof_seed)
  gain_row <- row_log_loss(truth_all, stack$original) - row_log_loss(truth_all, stack$pruned)
  case_gain_seed <- tapply(gain_row, train$Case, mean)
  case_gain_matrix[, seed_name] <- case_gain_seed[rownames(case_gain_matrix)]
  original_loss <- log_loss_matrix(truth_all, stack$original)
  pruned_loss <- log_loss_matrix(truth_all, stack$pruned)
  seed_summary_rows[[seed_name]] <- data.frame(
    seed = seed, original_logloss = original_loss, pruned_logloss = pruned_loss, gain = original_loss - pruned_loss
  )
  cat(sprintf("Seed %d: full v14-stack logloss original %.6f vs pruned %.6f, gain %.6f\n",
              seed, original_loss, pruned_loss, original_loss - pruned_loss))
}

seed_summary <- do.call(rbind, seed_summary_rows)
average_case_gain <- rowMeans(case_gain_matrix)
pooled <- summarize_gain(average_case_gain, 100000L, canonical_seed)
pooled$positive_repeats <- sum(seed_summary$gain > 0)
pooled$n_repeats <- nrow(seed_summary)
pooled$promote <- pooled$point_gain > 0 & pooled$lower_95 > 0 & pooled$positive_repeats >= 5L

cat("\nPer-seed full v14-stack gain:\n")
print(seed_summary, digits = 9)
cat("\nPooled full v14-stack gain (segment-shift pruning propagated through the whole ensemble):\n")
print(pooled, digits = 9)

write.csv(seed_summary, file.path(output_dir, "v14_propagation_by_seed.csv"), row.names = FALSE)
write.csv(pooled, file.path(output_dir, "v14_propagation_pooled.csv"), row.names = FALSE)
saveRDS(
  list(seed_summary = seed_summary, pooled = pooled, case_gain_matrix = case_gain_matrix,
       average_case_gain = average_case_gain, canonical_stack = canonical_stack),
  file.path(output_dir, "v14_propagation_result.rds")
)

verdict <- if (isTRUE(pooled$promote)) {
  "PROMOTE: segment-shift pruning improves the FULL v14 ensemble; proceed to full-data build."
} else {
  "The mlogit-level gain does NOT propagate to a confirmed full-v14-ensemble improvement; do not build or submit."
}
writeLines(verdict, file.path(output_dir, "v14_propagation_verdict.txt"))
cat("\n", verdict, "\n", sep = "")
