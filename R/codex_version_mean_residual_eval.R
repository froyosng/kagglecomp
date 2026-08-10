## Reconstruct the pre-specified empirical-Bayes mean-residual correction from
## the cached nested-CV base fits. This is the literal user-requested rule:
##
##   delta_v = n_v / (n_v + alpha) * mean(y4 - p4)
##
## Alpha is selected independently inside each outer fold. This script performs
## no model fitting; R/codex_version_shrinkage_cv.R creates the required frozen
## base-architecture caches.

source("R/codex_version_shrinkage_common.R")

train <- read.csv("csv files/train.csv")
train <- train[order(train$No), , drop = FALSE]
truth <- truth_wide(train)
version_map <- load_questionnaire_versions()
official <- current_fixed_oof(train)
official_pred <- official$pred
fold_of_case <- official$fold_of_case
row_fold <- as.integer(
  fold_of_case[match(as.character(train$Case), names(fold_of_case))]
)
stopifnot(
  !anyNA(row_fold),
  identical(sort(unique(row_fold)), 1:5)
)

alpha_key <- function(alpha) {
  if (is.infinite(alpha)) "Inf" else as.character(alpha)
}

required_cache <- function(label, source_wide, target_wide,
                           seed_offset) {
  path <- file.path(
    version_output_dir, paste0(label, "_base_fit.rds")
  )
  stopifnot(file.exists(path))
  fit <- readRDS(path)
  stopifnot(
    identical(fit$source_no, as.integer(source_wide$No)),
    identical(fit$target_no, as.integer(target_wide$No)),
    identical(fit$seed_offset, as.integer(seed_offset))
  )
  fit
}

corrected_oof <- matrix(NA_real_, nrow(train), 4L)
fresh_baseline_oof <- matrix(NA_real_, nrow(train), 4L)
fresh_corrected_oof <- matrix(NA_real_, nrow(train), 4L)
inner_fit_rows <- list()
inner_pooled_rows <- list()
outer_rows <- list()
delta_rows <- list()

for (outer_fold in 1:5) {
  outer_train_rows <- row_fold != outer_fold
  outer_valid_rows <- row_fold == outer_fold
  outer_train <- train[outer_train_rows, , drop = FALSE]
  outer_valid <- train[outer_valid_rows, , drop = FALSE]
  pooled_truth <- matrix(
    NA_real_, nrow(outer_train), 4L,
    dimnames = list(as.character(outer_train$No), NULL)
  )
  pooled_baseline <- pooled_truth
  pooled_adjusted <- lapply(
    version_alpha_grid, function(x) pooled_truth
  )
  names(pooled_adjusted) <- vapply(
    version_alpha_grid, alpha_key, character(1)
  )

  for (inner_fold in setdiff(1:5, outer_fold)) {
    source_rows <- !row_fold %in% c(outer_fold, inner_fold)
    target_rows <- row_fold == inner_fold
    source_wide <- train[source_rows, , drop = FALSE]
    target_wide <- train[target_rows, , drop = FALSE]
    label <- sprintf("outer%d_inner%d", outer_fold, inner_fold)
    fit <- required_cache(
      label, source_wide, target_wide,
      outer_fold * 10000L + inner_fold * 100L
    )
    evaluation <- evaluate_alpha_grid(
      source_wide, fit$source_pred,
      target_wide, fit$target_pred,
      version_map, version_alpha_grid
    )
    curve <- evaluation$curve
    curve$outer_fold <- outer_fold
    curve$inner_fold <- inner_fold
    inner_fit_rows[[label]] <- curve
    key <- as.character(target_wide$No)
    pooled_truth[key, ] <- truth_wide(target_wide)
    pooled_baseline[key, ] <- fit$target_pred
    for (alpha in version_alpha_grid) {
      alpha_name <- alpha_key(alpha)
      pooled_adjusted[[alpha_name]][key, ] <-
        evaluation$prediction[[alpha_name]]
    }
  }
  stopifnot(
    !anyNA(pooled_truth),
    !anyNA(pooled_baseline),
    all(vapply(pooled_adjusted, function(x) !anyNA(x), logical(1)))
  )

  pooled_curve <- data.frame(
    alpha = version_alpha_grid,
    baseline_loss = log_loss_matrix(pooled_truth, pooled_baseline),
    corrected_loss = vapply(
      pooled_adjusted,
      function(pred) log_loss_matrix(pooled_truth, pred),
      numeric(1)
    )
  )
  pooled_curve$gain <- with(
    pooled_curve, baseline_loss - corrected_loss
  )
  selected_index <- which.min(pooled_curve$corrected_loss)
  selected_alpha <- pooled_curve$alpha[[selected_index]]
  pooled_curve$selected <- seq_len(nrow(pooled_curve)) == selected_index
  pooled_curve$outer_fold <- outer_fold
  inner_pooled_rows[[as.character(outer_fold)]] <- pooled_curve

  outer_fit <- required_cache(
    sprintf("outer%d_final", outer_fold),
    outer_train, outer_valid, outer_fold * 100L
  )
  delta_table <- estimate_optout_delta(
    outer_train, outer_fit$source_pred,
    version_map, selected_alpha
  )
  delta_table$outer_fold <- outer_fold
  delta_table$selected_alpha <- selected_alpha
  delta_rows[[as.character(outer_fold)]] <- delta_table

  official_baseline <- official_pred[
    outer_valid_rows, , drop = FALSE
  ]
  official_correction <- apply_optout_delta(
    official_baseline, outer_valid$Case,
    version_map, delta_table
  )
  fresh_correction <- apply_optout_delta(
    outer_fit$target_pred, outer_valid$Case,
    version_map, delta_table
  )
  corrected_oof[outer_valid_rows, ] <- official_correction$pred
  fresh_baseline_oof[outer_valid_rows, ] <- outer_fit$target_pred
  fresh_corrected_oof[outer_valid_rows, ] <- fresh_correction$pred

  outer_truth <- truth[outer_valid_rows, , drop = FALSE]
  official_loss <- log_loss_matrix(outer_truth, official_baseline)
  official_corrected_loss <- log_loss_matrix(
    outer_truth, official_correction$pred
  )
  fresh_loss <- log_loss_matrix(outer_truth, outer_fit$target_pred)
  fresh_corrected_loss <- log_loss_matrix(
    outer_truth, fresh_correction$pred
  )
  outer_rows[[as.character(outer_fold)]] <- data.frame(
    outer_fold = outer_fold,
    selected_alpha = selected_alpha,
    inner_baseline_loss = pooled_curve$baseline_loss[[1]],
    inner_selected_loss =
      pooled_curve$corrected_loss[[selected_index]],
    inner_gain = pooled_curve$gain[[selected_index]],
    official_baseline_loss = official_loss,
    official_corrected_loss = official_corrected_loss,
    official_gain = official_loss - official_corrected_loss,
    fresh_baseline_loss = fresh_loss,
    fresh_corrected_loss = fresh_corrected_loss,
    fresh_gain = fresh_loss - fresh_corrected_loss,
    validation_respondents = length(unique(outer_valid$Case)),
    corrected_respondents = length(unique(
      outer_valid$Case[!official_correction$zero_delta]
    )),
    zero_delta_respondents = length(unique(
      outer_valid$Case[official_correction$zero_delta]
    )),
    mean_abs_delta = mean(abs(official_correction$delta)),
    max_abs_delta = max(abs(official_correction$delta))
  )
}

stopifnot(
  !anyNA(corrected_oof),
  !anyNA(fresh_baseline_oof),
  !anyNA(fresh_corrected_oof)
)
inner_fit_curves <- do.call(rbind, inner_fit_rows)
inner_pooled_curves <- do.call(rbind, inner_pooled_rows)
outer_results <- do.call(rbind, outer_rows)
delta_tables <- do.call(rbind, delta_rows)
overall <- data.frame(
  model = c("fresh_refit", "official_fixed15"),
  baseline_loss = c(
    log_loss_matrix(truth, fresh_baseline_oof),
    log_loss_matrix(truth, official_pred)
  ),
  corrected_loss = c(
    log_loss_matrix(truth, fresh_corrected_oof),
    log_loss_matrix(truth, corrected_oof)
  )
)
overall$gain <- overall$baseline_loss - overall$corrected_loss
bootstrap <- rbind(
  cbind(
    model = "fresh_refit",
    bootstrap_case_gain(
      truth, fresh_baseline_oof, fresh_corrected_oof,
      train$Case, replicates = 100000L, seed = 4821L
    )
  ),
  cbind(
    model = "official_fixed15",
    bootstrap_case_gain(
      truth, official_pred, corrected_oof,
      train$Case, replicates = 100000L, seed = 4821L
    )
  )
)

write.csv(
  inner_fit_curves,
  file.path(version_output_dir, "mean_residual_inner_fit_curves.csv"),
  row.names = FALSE
)
write.csv(
  inner_pooled_curves,
  file.path(
    version_output_dir, "mean_residual_inner_pooled_curves.csv"
  ),
  row.names = FALSE
)
write.csv(
  outer_results,
  file.path(version_output_dir, "mean_residual_outer_results.csv"),
  row.names = FALSE
)
write.csv(
  delta_tables,
  file.path(version_output_dir, "mean_residual_delta_tables.csv"),
  row.names = FALSE
)
write.csv(
  overall,
  file.path(version_output_dir, "mean_residual_overall.csv"),
  row.names = FALSE
)
write.csv(
  bootstrap,
  file.path(version_output_dir, "mean_residual_bootstrap.csv"),
  row.names = FALSE
)
saveRDS(
  list(
    official_baseline_oof = official_pred,
    official_corrected_oof = corrected_oof,
    fresh_baseline_oof = fresh_baseline_oof,
    fresh_corrected_oof = fresh_corrected_oof,
    inner_fit_curves = inner_fit_curves,
    inner_pooled_curves = inner_pooled_curves,
    outer_results = outer_results,
    delta_tables = delta_tables,
    overall = overall,
    bootstrap = bootstrap,
    truth = truth,
    case = train$Case,
    no = train$No,
    row_fold = row_fold,
    alpha_grid = version_alpha_grid
  ),
  file.path(version_output_dir, "mean_residual_cv_result.rds")
)

cat("\n===== FINAL MEAN-RESIDUAL VERSION-CV RESULT =====\n")
print(outer_results)
cat("\nOverall:\n")
print(overall)
cat("\nRespondent-clustered bootstrap:\n")
print(bootstrap)
