## Canonical respondent-held-out screen for the literal mean-residual
## questionnaire-version correction.

source("R/codex_version_shrinkage_common.R")

split_data <- readRDS("data_processed/train_val_split.rds")
full_train <- read.csv("csv files/train.csv")
reference <- feature_reference(full_train)
version_map <- load_questionnaire_versions()
source_wide <- as.data.frame(split_data$train_wide_tr)
target_wide <- as.data.frame(split_data$train_wide_val)
source_wide <- source_wide[order(source_wide$No), , drop = FALSE]
target_wide <- target_wide[order(target_wide$No), , drop = FALSE]
stopifnot(!any(source_wide$Case %in% target_wide$Case))

fit_path <- file.path(version_output_dir, "screen_base_fit.rds")
if (file.exists(fit_path)) {
  fit <- readRDS(fit_path)
  stopifnot(
    identical(fit$source_no, as.integer(source_wide$No)),
    identical(fit$target_no, as.integer(target_wide$No))
  )
} else {
  fit <- fit_submitted_architecture(
    source_wide, target_wide, reference, seed_offset = 0L
  )
  saveRDS(fit, fit_path)
}

evaluation <- evaluate_alpha_grid(
  source_wide, fit$source_pred,
  target_wide, fit$target_pred,
  version_map, version_alpha_grid
)
curve <- evaluation$curve
best_index <- which.min(curve$corrected_loss)
best_alpha <- curve$alpha[[best_index]]
best_key <- if (is.infinite(best_alpha)) {
  "Inf"
} else {
  as.character(best_alpha)
}
source_versions <- unique(data.frame(
  Case = source_wide$Case,
  version_id = case_version(source_wide$Case, version_map)
))
target_versions <- unique(data.frame(
  Case = target_wide$Case,
  version_id = case_version(target_wide$Case, version_map)
))
covered <- merge(target_versions, source_versions, by = "version_id")
coverage <- data.frame(
  train_respondents = length(unique(source_wide$Case)),
  validation_respondents = length(unique(target_wide$Case)),
  validation_respondents_with_source = length(unique(covered$Case.x)),
  validation_versions = length(unique(target_versions$version_id)),
  validation_versions_with_source = length(unique(covered$version_id))
)

write.csv(
  curve,
  file.path(version_output_dir, "screen_alpha_curve.csv"),
  row.names = FALSE
)
write.csv(
  coverage,
  file.path(version_output_dir, "screen_coverage.csv"),
  row.names = FALSE
)
saveRDS(
  list(
    curve = curve,
    coverage = coverage,
    best_alpha = best_alpha,
    best_prediction = evaluation$prediction[[best_key]],
    baseline_prediction = fit$target_pred,
    truth = truth_wide(target_wide),
    case = target_wide$Case,
    no = target_wide$No
  ),
  file.path(version_output_dir, "screen_result.rds")
)
print(curve)
print(coverage)
