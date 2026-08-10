# Independent artifact audit for the repeated-CV and eight-component round.
#
# This script performs no model fitting and writes no project log files. It
# verifies the saved OOF results and candidate submission directly from their
# probability matrices.

options(stringsAsFactors = FALSE)

artifact_dir <- file.path("data_processed", "codex_repeat_cv")
expected_seeds <- c(4821L, 1907L, 2719L, 6151L, 8293L, 104729L)
component_names <- c(
  "mlogit", "original_xgb", "rank_ndcg", "retuned_xgb",
  "cox", "shallow_mlp", "triple_mlogit", "deep_mlp"
)
expected_weights <- c(
  mlogit = 0.06246872652458,
  original_xgb = 0.00474541986529,
  rank_ndcg = 0.09330490494295,
  retuned_xgb = 0.01555896281673,
  cox = 0.13111978085328,
  shallow_mlp = 0.08008528400844,
  triple_mlogit = 0.47925772846576,
  deep_mlp = 0.13345919252297
)

assert_close <- function(x, y, tolerance = 1e-10) {
  stopifnot(
    identical(dim(x), dim(y)),
    max(abs(as.numeric(x) - as.numeric(y))) < tolerance
  )
}

validate_probability <- function(x, n_expected) {
  stopifnot(
    is.matrix(x),
    identical(dim(x), c(n_expected, 4L)),
    all(is.finite(x)),
    all(x > 0),
    max(abs(rowSums(x) - 1)) < 1e-10
  )
  invisible(TRUE)
}

matrix_logloss <- function(truth, prediction) {
  stopifnot(identical(dim(truth), dim(prediction)))
  -mean(log(pmax(rowSums(truth * prediction), 1e-15)))
}

case_gain <- function(truth, baseline, candidate, case) {
  baseline_loss <- -log(pmax(rowSums(truth * baseline), 1e-15))
  candidate_loss <- -log(pmax(rowSums(truth * candidate), 1e-15))
  unname(tapply(baseline_loss - candidate_loss, case, mean))
}

bootstrap_summary_from_saved <- function(bootstrap, family_size) {
  family_alpha <- 0.05 / family_size
  c(
    bootstrap_mean = mean(bootstrap),
    bootstrap_sd = sd(bootstrap),
    lower_95 = unname(quantile(bootstrap, 0.025)),
    upper_95 = unname(quantile(bootstrap, 0.975)),
    lower_99 = unname(quantile(bootstrap, 0.005)),
    upper_99 = unname(quantile(bootstrap, 0.995)),
    lower_family = unname(quantile(bootstrap, family_alpha / 2)),
    upper_family = unname(quantile(
      bootstrap, 1 - family_alpha / 2
    )),
    win_rate = mean(bootstrap > 0)
  )
}

train <- read.csv(
  file.path("csv files", "train.csv"), check.names = FALSE
)
test <- read.csv(
  file.path("csv files", "test.csv"), check.names = FALSE
)
truth <- as.matrix(train[paste0("Ch", 1:4)])
stopifnot(
  nrow(train) == 21565L,
  length(unique(train$Case)) == 1135L,
  all(rowSums(truth) == 1)
)

result <- readRDS(file.path(artifact_dir, "repeat_cv_results.rds"))
summary_csv <- read.csv(file.path(
  artifact_dir, "repeat_cv_summary.csv"
))
bootstrap_csv <- read.csv(file.path(
  artifact_dir, "repeat_cv_bootstrap.csv"
))
decision_csv <- read.csv(file.path(
  artifact_dir, "repeat_cv_decision.csv"
))

stopifnot(
  identical(result$seeds, expected_seeds),
  identical(
    as.integer(result$repeat_summary$repeat_seed),
    as.integer(summary_csv$repeat_seed)
  ),
  identical(
    as.character(result$repeat_summary$candidate),
    as.character(summary_csv$candidate)
  )
)
numeric_summary_columns <- c(
  "baseline_loss", "candidate_loss", "gain"
)
for (column in numeric_summary_columns) {
  assert_close(
    as.matrix(result$repeat_summary[column]),
    as.matrix(summary_csv[column])
  )
}

history_gain_recomputed <- matrix(
  NA_real_, nrow = 1135L, ncol = 5L
)
eight_gain_recomputed <- history_gain_recomputed
colnames(history_gain_recomputed) <- as.character(expected_seeds[-1L])
colnames(eight_gain_recomputed) <- colnames(history_gain_recomputed)

for (seed in expected_seeds[-1L]) {
  seed_name <- as.character(seed)
  repeat_result <- readRDS(file.path(
    artifact_dir, paste0("repeat_seed_", seed, ".rds")
  ))
  stopifnot(
    identical(repeat_result$repeat_seed, seed),
    identical(names(repeat_result$components), component_names),
    length(repeat_result$fold_map) == 1135L,
    identical(sort(unique(repeat_result$fold_map)), 1:5),
    identical(
      as.integer(repeat_result$row_fold),
      as.integer(repeat_result$fold_map[
        as.character(train$Case)
      ])
    )
  )
  fold_counts <- tabulate(repeat_result$fold_map, nbins = 5L)
  stopifnot(max(fold_counts) - min(fold_counts) <= 1L)

  for (prediction in repeat_result$components) {
    validate_probability(prediction, nrow(train))
  }
  validate_probability(
    repeat_result$current$prediction, nrow(train)
  )
  validate_probability(
    repeat_result$augmented8$prediction, nrow(train)
  )
  validate_probability(
    repeat_result$history$baseline, nrow(train)
  )
  validate_probability(
    repeat_result$history$candidate, nrow(train)
  )

  current_loss <- matrix_logloss(
    truth, repeat_result$current$prediction
  )
  eight_loss <- matrix_logloss(
    truth, repeat_result$augmented8$prediction
  )
  history_baseline_loss <- matrix_logloss(
    truth, repeat_result$history$baseline
  )
  history_candidate_loss <- matrix_logloss(
    truth, repeat_result$history$candidate
  )
  stopifnot(
    abs(current_loss - repeat_result$current$logloss) < 1e-12,
    abs(eight_loss - repeat_result$augmented8$logloss) < 1e-12
  )

  eight_row <- summary_csv[
    summary_csv$candidate == "augmented8_arithmetic" &
      summary_csv$repeat_seed == seed, ,
    drop = FALSE
  ]
  history_row <- summary_csv[
    summary_csv$candidate == "history_both_k3" &
      summary_csv$repeat_seed == seed, ,
    drop = FALSE
  ]
  stopifnot(
    nrow(eight_row) == 1L,
    nrow(history_row) == 1L,
    abs(eight_row$baseline_loss - current_loss) < 1e-12,
    abs(eight_row$candidate_loss - eight_loss) < 1e-12,
    abs(
      history_row$baseline_loss - history_baseline_loss
    ) < 1e-12,
    abs(
      history_row$candidate_loss - history_candidate_loss
    ) < 1e-12
  )

  history_gain_recomputed[, seed_name] <- case_gain(
    truth,
    repeat_result$history$baseline,
    repeat_result$history$candidate,
    train$Case
  )
  eight_gain_recomputed[, seed_name] <- case_gain(
    truth,
    repeat_result$current$prediction,
    repeat_result$augmented8$prediction,
    train$Case
  )
}

assert_close(
  history_gain_recomputed,
  result$history_case_gain[, -1L, drop = FALSE]
)
assert_close(
  eight_gain_recomputed,
  result$eight_case_gain[, -1L, drop = FALSE]
)
stopifnot(
  all(colMeans(result$history_case_gain) > 0),
  all(colMeans(result$eight_case_gain) > 0),
  abs(
    mean(result$history_case_gain) -
      bootstrap_csv$point_gain[
        bootstrap_csv$candidate == "history_both_k3"
      ]
  ) < 1e-12,
  abs(
    mean(result$eight_case_gain) -
      bootstrap_csv$point_gain[
        bootstrap_csv$candidate == "augmented8_arithmetic"
      ]
  ) < 1e-12
)

for (candidate in c("history", "eight")) {
  saved <- result[[paste0(candidate, "_bootstrap")]]
  recomputed <- bootstrap_summary_from_saved(
    saved$bootstrap,
    if (candidate == "history") 13L else 6L
  )
  candidate_name <- if (candidate == "history") {
    "history_both_k3"
  } else {
    "augmented8_arithmetic"
  }
  csv_row <- bootstrap_csv[
    bootstrap_csv$candidate == candidate_name, ,
    drop = FALSE
  ]
  assert_close(
    matrix(recomputed, nrow = 1L),
    as.matrix(csv_row[names(recomputed)])
  )
}
stopifnot(
  identical(as.logical(decision_csv$promote), c(FALSE, FALSE)),
  identical(as.integer(decision_csv$positive_repeats), c(6L, 6L)),
  all(decision_csv$family_lower < 0)
)

joint <- readRDS(file.path(artifact_dir, "joint_screen.rds"))
stopifnot(
  nrow(joint$result) == 1L,
  isFALSE(joint$result$passed_screen),
  joint$result$gain_vs_triple < 0,
  !file.exists(file.path(artifact_dir, "joint_cv.rds"))
)

submission <- read.csv(
  "submission_codex_8component_candidate.csv",
  check.names = FALSE
)
sample_submission <- read.csv(
  file.path("csv files", "sample_submission.csv"),
  check.names = FALSE
)
submission_audit <- readRDS(file.path(
  artifact_dir, "submission_8component_audit.rds"
))
submission_components <- readRDS(file.path(
  artifact_dir, "submission_8component_components.rds"
))
candidate_matrix <- as.matrix(submission[paste0("Ch", 1:4)])

stopifnot(
  identical(submission$No, sample_submission$No),
  identical(names(submission), names(sample_submission)),
  setequal(names(submission_components), component_names),
  identical(names(submission_audit$weights), component_names),
  max(abs(submission_audit$weights - expected_weights)) < 1e-14,
  abs(sum(expected_weights) - 1) < 1e-12
)
validate_probability(candidate_matrix, nrow(test))
for (prediction in submission_components) {
  validate_probability(prediction, nrow(test))
}

candidate_rebuilt <- matrix(0, nrow(test), 4L)
for (component in component_names) {
  candidate_rebuilt <- candidate_rebuilt +
    expected_weights[[component]] *
    submission_components[[component]]
}
assert_close(candidate_matrix, candidate_rebuilt, tolerance = 1e-12)
assert_close(
  candidate_matrix, submission_audit$candidate,
  tolerance = 1e-12
)

current_submission <- read.csv(
  "submission_codex_mlp_v12_candidate.csv",
  check.names = FALSE
)
current_matrix <- as.matrix(current_submission[paste0("Ch", 1:4)])
assert_close(
  current_matrix, submission_audit$current,
  tolerance = 1e-12
)
absolute_change <- abs(candidate_matrix - current_matrix)
row_maximum_change <- apply(absolute_change, 1L, max)
maximum_index <- which(absolute_change == max(absolute_change),
                       arr.ind = TRUE)[1L, ]
stopifnot(
  submission$No[maximum_index[[1L]]] == 22637L,
  abs(max(absolute_change) - 0.362629886864792) < 1e-12,
  sum(row_maximum_change > 0.05) == 159L,
  sum(row_maximum_change > 0.10) == 50L,
  sum(row_maximum_change > 0.15) == 30L
)
outlier_case <- test$Case[test$No == 22637L]
outlier_rows <- test$Case == outlier_case
stopifnot(
  length(outlier_case) == 1L,
  sum(outlier_rows) == 19L,
  all(row_maximum_change[outlier_rows] > 0.05),
  sum(row_maximum_change[outlier_rows] > 0.10) == 17L,
  sum(row_maximum_change[outlier_rows] > 0.15) == 12L
)

cat("Repeated-CV and submission artifact audit passed.\n")
cat(sprintf(
  "History: mean gain %.9f, ordinary 95%% [%.9f, %.9f], adjusted lower %.9f.\n",
  bootstrap_csv$point_gain[1L],
  bootstrap_csv$lower_95[1L],
  bootstrap_csv$upper_95[1L],
  bootstrap_csv$lower_family[1L]
))
cat(sprintf(
  "Eight component: mean gain %.9f, ordinary 95%% [%.9f, %.9f], adjusted lower %.9f.\n",
  bootstrap_csv$point_gain[2L],
  bootstrap_csv$lower_95[2L],
  bootstrap_csv$upper_95[2L],
  bootstrap_csv$lower_family[2L]
))
cat(sprintf(
  "Submission audit: max change %.9f at No 22637; %d/19 outlier-respondent rows exceed 0.10.\n",
  max(absolute_change),
  sum(row_maximum_change[outlier_rows] > 0.10)
))
