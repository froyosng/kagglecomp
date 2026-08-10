## Independent artifact-level audit of the Newton version correction.

source("R/codex_version_shrinkage_common.R")

result <- readRDS(
  file.path(version_output_dir, "newton_cv_result.rds")
)
train <- read.csv("csv files/train.csv")
train <- train[order(train$No), , drop = FALSE]
truth <- truth_wide(train)
version_map <- load_questionnaire_versions()
row_fold <- as.integer(result$row_fold)

prediction_names <- c(
  "fresh_baseline_oof",
  "fresh_corrected_oof",
  "official_baseline_oof",
  "official_corrected_oof",
  "fresh_peer_robust_oof",
  "official_peer_robust_oof"
)
for (name in prediction_names) {
  prediction <- result[[name]]
  stopifnot(
    identical(dim(prediction), c(nrow(train), 4L)),
    all(is.finite(prediction)),
    all(prediction > 0),
    max(abs(rowSums(prediction) - 1)) < 1e-12
  )
}
stopifnot(
  abs(
    log_loss_matrix(truth, result$official_baseline_oof) -
      1.143686618134879
  ) < 1e-10
)

audit_rows <- list()
for (outer_fold in 1:5) {
  outer_train_rows <- row_fold != outer_fold
  outer_valid_rows <- row_fold == outer_fold
  outer_train <- train[outer_train_rows, , drop = FALSE]
  outer_valid <- train[outer_valid_rows, , drop = FALSE]
  inner_oof <- matrix(
    NA_real_, nrow(outer_train), 4L,
    dimnames = list(as.character(outer_train$No), NULL)
  )

  for (inner_fold in setdiff(1:5, outer_fold)) {
    fit_label <- sprintf(
      "outer%d_inner%d", outer_fold, inner_fold
    )
    fit <- readRDS(file.path(
      version_output_dir, paste0(fit_label, "_base_fit.rds")
    ))
    expected_source <- train[
      !row_fold %in% c(outer_fold, inner_fold), ,
      drop = FALSE
    ]
    expected_target <- train[row_fold == inner_fold, , drop = FALSE]
    stopifnot(
      identical(fit$source_no, as.integer(expected_source$No)),
      identical(fit$target_no, as.integer(expected_target$No)),
      !any(fit$source_case %in% fit$target_case),
      !any(fit$source_case %in% outer_valid$Case),
      !any(fit$target_case %in% outer_valid$Case)
    )
    inner_oof[as.character(expected_target$No), ] <- fit$target_pred
  }
  stopifnot(!anyNA(inner_oof))

  reevaluated <- evaluate_newton_lambda_grid(
    outer_train, inner_oof, version_map,
    lambda_grid = version_lambda_grid
  )$curve
  saved_curve <- result$lambda_curves[
    result$lambda_curves$outer_fold == outer_fold, ,
    drop = FALSE
  ]
  stopifnot(
    identical(reevaluated$lambda, saved_curve$lambda),
    max(abs(
      reevaluated$corrected_loss -
        saved_curve$corrected_loss
    )) < 1e-14,
    which.min(reevaluated$corrected_loss) ==
      which(saved_curve$selected)
  )
  selected_lambda <- saved_curve$lambda[saved_curve$selected]
  recomputed_delta <- estimate_newton_delta(
    outer_train, inner_oof, version_map, selected_lambda
  )
  saved_delta <- result$delta_tables[
    result$delta_tables$outer_fold == outer_fold, ,
    drop = FALSE
  ]
  saved_delta <- saved_delta[
    match(recomputed_delta$version_id, saved_delta$version_id), ,
    drop = FALSE
  ]
  stopifnot(
    identical(recomputed_delta$version_id, saved_delta$version_id),
    max(abs(recomputed_delta$g - saved_delta$g)) < 1e-14,
    max(abs(recomputed_delta$h - saved_delta$h)) < 1e-14,
    max(abs(recomputed_delta$delta - saved_delta$delta)) < 1e-14
  )
  if (is.infinite(selected_lambda)) {
    stopifnot(all(recomputed_delta$delta == 0))
  } else {
    stopifnot(max(abs(
      recomputed_delta$delta -
        (-recomputed_delta$g /
          (recomputed_delta$h + selected_lambda))
    )) < 1e-14)
  }

  outer_fit <- readRDS(file.path(
    version_output_dir,
    sprintf("outer%d_final_base_fit.rds", outer_fold)
  ))
  stopifnot(
    identical(outer_fit$source_no, as.integer(outer_train$No)),
    identical(outer_fit$target_no, as.integer(outer_valid$No)),
    !any(outer_fit$source_case %in% outer_fit$target_case)
  )
  fresh_correction <- apply_optout_delta(
    outer_fit$target_pred, outer_valid$Case,
    version_map, recomputed_delta
  )$pred
  official_correction <- apply_optout_delta(
    result$official_baseline_oof[
      outer_valid_rows, , drop = FALSE
    ],
    outer_valid$Case, version_map, recomputed_delta
  )$pred
  stopifnot(
    max(abs(
      fresh_correction -
        result$fresh_corrected_oof[
          outer_valid_rows, , drop = FALSE
        ]
    )) < 1e-14,
    max(abs(
      official_correction -
        result$official_corrected_oof[
          outer_valid_rows, , drop = FALSE
        ]
    )) < 1e-14
  )
  audit_rows[[as.character(outer_fold)]] <- data.frame(
    outer_fold = outer_fold,
    selected_lambda = selected_lambda,
    inner_oof_rows = nrow(inner_oof),
    inner_oof_missing = sum(is.na(inner_oof)),
    outer_source_respondents = length(unique(outer_train$Case)),
    outer_target_respondents = length(unique(outer_valid$Case)),
    fresh_fold_gain =
      log_loss_matrix(
        truth[outer_valid_rows, , drop = FALSE],
        result$fresh_baseline_oof[
          outer_valid_rows, , drop = FALSE
        ]
      ) -
      log_loss_matrix(
        truth[outer_valid_rows, , drop = FALSE],
        result$fresh_corrected_oof[
          outer_valid_rows, , drop = FALSE
        ]
      ),
    official_fold_gain =
      log_loss_matrix(
        truth[outer_valid_rows, , drop = FALSE],
        result$official_baseline_oof[
          outer_valid_rows, , drop = FALSE
        ]
      ) -
      log_loss_matrix(
        truth[outer_valid_rows, , drop = FALSE],
        result$official_corrected_oof[
          outer_valid_rows, , drop = FALSE
        ]
      ),
    passed = TRUE
  )
}

audit <- do.call(rbind, audit_rows)
write.csv(
  audit,
  file.path(version_output_dir, "newton_audit.csv"),
  row.names = FALSE
)
cat("Artifact audit passed:\n")
print(audit, digits = 10)
