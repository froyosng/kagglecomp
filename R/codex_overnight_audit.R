# Independent artifact audit for the 2026-07-28 overnight queue.
#
# This script performs no fitting. It recomputes the reported losses from the
# saved OOF matrices, checks row/fold alignment, verifies the preregistered
# architecture registry and selection, and confirms the version-correction
# headline against its saved CSV artifacts.

options(stringsAsFactors = FALSE)

output_dir <- "data_processed/codex_overnight_queue"
version_dir <- "data_processed/codex_version_shrinkage"
expected_baseline <- 1.143686618134879
tolerance <- 5e-10

log_loss_matrix <- function(truth, prediction) {
  stopifnot(
    identical(dim(truth), dim(prediction)),
    !anyNA(prediction),
    all(prediction > 0),
    max(abs(rowSums(prediction) - 1)) < 5e-8
  )
  -mean(log(rowSums(truth * prediction)))
}

fold_losses <- function(truth, prediction, row_fold) {
  vapply(1:5, function(fold) {
    rows <- row_fold == fold
    log_loss_matrix(
      truth[rows, , drop = FALSE],
      prediction[rows, , drop = FALSE]
    )
  }, numeric(1))
}

# Version-level Newton correction.
version_overall <- read.csv(
  file.path(version_dir, "newton_overall.csv")
)
version_bootstrap <- read.csv(
  file.path(version_dir, "newton_bootstrap.csv")
)
version_official <- version_overall[
  version_overall$model == "official_fixed15",
  ,
  drop = FALSE
]
version_official_boot <- version_bootstrap[
  version_bootstrap$model == "official_fixed15",
  ,
  drop = FALSE
]
stopifnot(
  nrow(version_official) == 1L,
  nrow(version_official_boot) == 1L,
  abs(version_official$baseline_loss - expected_baseline) < tolerance,
  abs(
    version_official$baseline_loss -
      version_official$corrected_loss -
      version_official$gain
  ) < tolerance,
  version_official$gain < 0,
  version_official_boot$lower_95 < 0,
  version_official_boot$upper_95 > 0
)

# Historical population-prior smoothing.
history <- readRDS(file.path(output_dir, "history_prior_cv.rds"))
history_screen <- read.csv(
  file.path(output_dir, "history_prior_screen.csv")
)
stopifnot(
  length(unique(history_screen$candidate)) == 5L,
  history$family_size == 5L,
  identical(history$candidates, "both_k3"),
  identical(history$no, seq_len(nrow(history$truth))),
  identical(history$case, as.integer(history$case)),
  identical(history$row_fold, as.integer(history$row_fold)),
  all(table(history$case) == 19L),
  identical(sort(unique(history$row_fold)), 1:5),
  abs(log_loss_matrix(history$truth, history$baseline) -
        expected_baseline) < tolerance
)
history_candidate <- history$predictions[["both_k3"]]
history_baseline_loss <- log_loss_matrix(
  history$truth, history$baseline
)
history_candidate_loss <- log_loss_matrix(
  history$truth, history_candidate
)
history_fold_baseline <- fold_losses(
  history$truth, history$baseline, history$row_fold
)
history_fold_candidate <- fold_losses(
  history$truth, history_candidate, history$row_fold
)
stopifnot(
  abs(history$summary$baseline_loss - history_baseline_loss) <
    tolerance,
  abs(history$summary$candidate_loss - history_candidate_loss) <
    tolerance,
  abs(
    history$summary$gain -
      (history_baseline_loss - history_candidate_loss)
  ) < tolerance,
  max(abs(
    history$folds$baseline_loss - history_fold_baseline
  )) < tolerance,
  max(abs(
    history$folds$candidate_loss - history_fold_candidate
  )) < tolerance,
  history$summary$folds_improved == 4L,
  history$bootstrap$lower_95 < 0,
  history$bootstrap$upper_95 > 0
)

# Wider deep-MLP search.
deep_screen <- readRDS(file.path(output_dir, "deep_wide_screen.rds"))
deep <- readRDS(file.path(output_dir, "deep_wide_cv.rds"))
registry <- deep_screen$registry
screen_result <- deep_screen$result
stopifnot(
  nrow(registry) == 24L,
  length(unique(registry$config)) == 24L,
  identical(sort(registry$config), sort(screen_result$config)),
  sum(registry$config == "h128_064__B") == 1L,
  deep$family_size == 24L,
  deep$frozen_config ==
    screen_result$config[[which.min(screen_result$joint_loss)]],
  deep$frozen_config == deep_screen$selection$frozen_config,
  deep_screen$selection$passed_screen,
  identical(deep$no, history$no),
  identical(deep$case, history$case),
  identical(deep$row_fold, history$row_fold),
  identical(deep$truth, history$truth),
  max(abs(deep$baseline - history$baseline)) < tolerance,
  abs(log_loss_matrix(deep$truth, deep$baseline) -
        expected_baseline) < tolerance,
  all(abs(rowSums(deep$primary$weights) - 1) < tolerance),
  all(deep$primary$weights >= 0),
  all(deep$primary$weights <= 1)
)
deep_primary_loss <- log_loss_matrix(
  deep$truth, deep$primary$prediction
)
deep_fold_baseline <- fold_losses(
  deep$truth, deep$baseline, deep$row_fold
)
deep_fold_primary <- fold_losses(
  deep$truth, deep$primary$prediction, deep$row_fold
)
stopifnot(
  abs(deep$summary$primary_joint_loss - deep_primary_loss) <
    tolerance,
  abs(
    deep$summary$primary_gain -
      (expected_baseline - deep_primary_loss)
  ) < tolerance,
  max(abs(
    deep$fold_results$baseline_loss - deep_fold_baseline
  )) < tolerance,
  max(abs(
    deep$fold_results$primary_loss - deep_fold_primary
  )) < tolerance,
  deep$summary$folds_improved == 4L,
  deep$bootstrap$lower_95 < 0,
  deep$bootstrap$lower_bonferroni_24 < 0
)

audit_summary <- data.frame(
  experiment = c(
    "version_newton",
    "history_population_prior",
    "deep_mlp_wide"
  ),
  baseline_loss = c(
    version_official$baseline_loss,
    history_baseline_loss,
    expected_baseline
  ),
  candidate_loss = c(
    version_official$corrected_loss,
    history_candidate_loss,
    deep_primary_loss
  ),
  gain = c(
    version_official$gain,
    history_baseline_loss - history_candidate_loss,
    expected_baseline - deep_primary_loss
  ),
  lower_95 = c(
    version_official_boot$lower_95,
    history$bootstrap$lower_95,
    deep$bootstrap$lower_95
  ),
  upper_95 = c(
    version_official_boot$upper_95,
    history$bootstrap$upper_95,
    deep$bootstrap$upper_95
  )
)

print(audit_summary, digits = 12, row.names = FALSE)
cat("All overnight artifact audits passed.\n")
