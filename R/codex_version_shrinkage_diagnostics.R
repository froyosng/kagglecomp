## Diagnostics for the completed nested Newton version correction.

source("R/codex_version_shrinkage_common.R")

result <- readRDS(
  file.path(version_output_dir, "newton_cv_result.rds")
)
train <- read.csv("csv files/train.csv")
train <- train[order(train$No), , drop = FALSE]
truth <- truth_wide(train)
baseline <- result$official_baseline_oof
candidates <- list(
  newton = result$official_corrected_oof,
  exclude_one_peer = result$official_peer_robust_oof
)

row_loss <- function(prediction) {
  prediction <- normalize_probability(prediction)
  -rowSums(truth * log(pmin(pmax(prediction, 1e-15), 1)))
}
baseline_row_loss <- row_loss(baseline)

summarize_slice <- function(group, group_type) {
  rows <- lapply(names(candidates), function(candidate_name) {
    candidate_loss <- row_loss(candidates[[candidate_name]])
    split_index <- split(seq_along(group), group)
    do.call(rbind, lapply(names(split_index), function(level) {
      index <- split_index[[level]]
      data.frame(
        group_type = group_type,
        group = level,
        candidate = candidate_name,
        respondents = length(unique(train$Case[index])),
        tasks = length(index),
        baseline_loss = mean(baseline_row_loss[index]),
        corrected_loss = mean(candidate_loss[index]),
        gain = mean(
          baseline_row_loss[index] - candidate_loss[index]
        )
      )
    }))
  })
  do.call(rbind, rows)
}

outcome_group <- ifelse(truth[, 4] == 1, "opt_out_chosen", "inside_chosen")
task_band <- cut(
  train$Task,
  breaks = c(0, 6, 13, 19),
  labels = c("tasks_01_06", "tasks_07_13", "tasks_14_19")
)
fold_group <- paste0("fold_", result$row_fold)

case_score <- read.csv(
  "data_processed/codex_shared_utility/test_like_case_scores.csv"
)
case_score <- case_score[order(case_score$test_probability), , drop = FALSE]
case_score$propensity_group <- rep(
  c("low", "middle", "high"),
  times = c(
    floor(nrow(case_score) / 3),
    floor(nrow(case_score) / 3),
    nrow(case_score) - 2 * floor(nrow(case_score) / 3)
  )
)
propensity_group <- case_score$propensity_group[
  match(train$Case, case_score$Case)
]
stopifnot(!anyNA(propensity_group))

slice_results <- rbind(
  summarize_slice(outcome_group, "chosen_alternative"),
  summarize_slice(task_band, "task_band"),
  summarize_slice(propensity_group, "test_propensity"),
  summarize_slice(fold_group, "outer_fold")
)

calibration_bin <- ceiling(
  rank(baseline[, 4], ties.method = "first") /
    nrow(baseline) * 10
)
calibration_bin[calibration_bin < 1] <- 1
calibration_bin[calibration_bin > 10] <- 10
calibration <- do.call(rbind, lapply(
  c(list(baseline = baseline), candidates),
  function(prediction) {
    split_index <- split(seq_len(nrow(train)), calibration_bin)
    do.call(rbind, lapply(names(split_index), function(bin) {
      index <- split_index[[bin]]
      data.frame(
        bin = as.integer(bin),
        tasks = length(index),
        mean_predicted_optout = mean(prediction[index, 4]),
        observed_optout = mean(truth[index, 4]),
        calibration_error =
          mean(prediction[index, 4]) - mean(truth[index, 4])
      )
    }))
  }
))
calibration$model <- rep(
  c("baseline", names(candidates)),
  each = length(unique(calibration_bin))
)

delta <- result$delta_tables
delta$raw_direction <- sign(-delta$g)
version_sign <- do.call(rbind, lapply(
  split(seq_len(nrow(delta)), delta$version_id),
  function(index) {
    direction <- delta$raw_direction[index]
    nonzero <- direction[direction != 0]
    data.frame(
      version_id = delta$version_id[index[[1]]],
      folds_available = length(index),
      positive_folds = sum(direction > 0),
      negative_folds = sum(direction < 0),
      zero_folds = sum(direction == 0),
      sign_consistency = if (length(nonzero) == 0L) {
        NA_real_
      } else {
        max(sum(nonzero > 0), sum(nonzero < 0)) / length(nonzero)
      },
      unanimous_nonzero = length(nonzero) > 1L &&
        abs(sum(sign(nonzero))) == length(nonzero)
    )
  }
))

dominance_summary <- do.call(rbind, lapply(
  split(seq_len(nrow(delta)), delta$n_respondent),
  function(index) {
    dominance <- delta$max_respondent_gradient_share[index]
    data.frame(
      training_respondents = delta$n_respondent[index[[1]]],
      version_fold_estimates = length(index),
      median_dominance = median(dominance),
      p90_dominance = unname(quantile(dominance, 0.9)),
      max_dominance = max(dominance),
      share_over_half = mean(dominance > 0.5)
    )
  }
))

heldout_peers <- result$heldout_peers
peer_summary <- do.call(rbind, lapply(
  split(
    seq_len(nrow(heldout_peers)),
    heldout_peers$outer_fold
  ),
  function(index) {
    peer <- heldout_peers$peer_count[index]
    dominance <- heldout_peers$max_respondent_gradient_share[index]
    data.frame(
      outer_fold = heldout_peers$outer_fold[index[[1]]],
      heldout_respondents = length(index),
      zero_peers = sum(peer == 0L),
      one_peer = sum(peer == 1L),
      two_or_more_peers = sum(peer >= 2L),
      mean_peers = mean(peer),
      median_peers = median(peer),
      max_peers = max(peer),
      dominance_over_half_with_two_plus =
        mean(dominance[peer >= 2L] > 0.5),
      max_dominance_with_two_plus = max(dominance[peer >= 2L])
    )
  }
))

write.csv(
  slice_results,
  file.path(version_output_dir, "newton_loss_slices.csv"),
  row.names = FALSE
)
write.csv(
  calibration,
  file.path(version_output_dir, "newton_optout_calibration.csv"),
  row.names = FALSE
)
write.csv(
  version_sign,
  file.path(version_output_dir, "newton_sign_consistency.csv"),
  row.names = FALSE
)
write.csv(
  dominance_summary,
  file.path(version_output_dir, "newton_dominance_summary.csv"),
  row.names = FALSE
)
write.csv(
  peer_summary,
  file.path(version_output_dir, "newton_peer_summary.csv"),
  row.names = FALSE
)
saveRDS(
  list(
    loss_slices = slice_results,
    optout_calibration = calibration,
    sign_consistency = version_sign,
    dominance_summary = dominance_summary,
    peer_summary = peer_summary
  ),
  file.path(version_output_dir, "newton_diagnostics.rds")
)

cat("Loss slices:\n")
print(slice_results, digits = 8)
cat("\nPeer summary:\n")
print(peer_summary, digits = 8)
cat("\nDominance summary:\n")
print(dominance_summary, digits = 8)
cat(sprintf(
  "\nRaw Newton direction unanimous in %.1f%% of versions with >=2 estimates.\n",
  100 * mean(
    version_sign$unanimous_nonzero[
      version_sign$folds_available >= 2
    ]
  )
))
