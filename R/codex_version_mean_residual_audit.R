## Independent artifact-level audit of the literal empirical-Bayes
## mean-residual version correction.

source("R/codex_version_shrinkage_common.R")

result <- readRDS(file.path(
  version_output_dir, "mean_residual_cv_result.rds"
))
train <- read.csv("csv files/train.csv")
train <- train[order(train$No), , drop = FALSE]
truth <- truth_wide(train)
version_map <- load_questionnaire_versions()
row_fold <- as.integer(result$row_fold)
stopifnot(
  abs(
    log_loss_matrix(truth, result$official_baseline_oof) -
      1.143686618134879
  ) < 1e-10,
  all(is.finite(result$official_corrected_oof)),
  all(result$official_corrected_oof > 0),
  max(abs(rowSums(result$official_corrected_oof) - 1)) < 1e-12
)

audit_rows <- list()
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
  names(pooled_adjusted) <- ifelse(
    is.infinite(version_alpha_grid),
    "Inf", as.character(version_alpha_grid)
  )

  for (inner_fold in setdiff(1:5, outer_fold)) {
    fit <- readRDS(file.path(
      version_output_dir,
      sprintf(
        "outer%d_inner%d_base_fit.rds",
        outer_fold, inner_fold
      )
    ))
    expected_source <- train[
      !row_fold %in% c(outer_fold, inner_fold), ,
      drop = FALSE
    ]
    expected_target <- train[
      row_fold == inner_fold, , drop = FALSE
    ]
    stopifnot(
      identical(fit$source_no, as.integer(expected_source$No)),
      identical(fit$target_no, as.integer(expected_target$No)),
      !any(fit$source_case %in% fit$target_case),
      !any(fit$source_case %in% outer_valid$Case)
    )
    reevaluated <- evaluate_alpha_grid(
      expected_source, fit$source_pred,
      expected_target, fit$target_pred,
      version_map, version_alpha_grid
    )
    key <- as.character(expected_target$No)
    pooled_truth[key, ] <- truth_wide(expected_target)
    pooled_baseline[key, ] <- fit$target_pred
    for (alpha_name in names(pooled_adjusted)) {
      pooled_adjusted[[alpha_name]][key, ] <-
        reevaluated$prediction[[alpha_name]]
    }
  }
  stopifnot(
    !anyNA(pooled_truth),
    !anyNA(pooled_baseline),
    all(vapply(pooled_adjusted, function(x) !anyNA(x), logical(1)))
  )
  recomputed_loss <- vapply(
    pooled_adjusted,
    function(pred) log_loss_matrix(pooled_truth, pred),
    numeric(1)
  )
  saved_curve <- result$inner_pooled_curves[
    result$inner_pooled_curves$outer_fold == outer_fold, ,
    drop = FALSE
  ]
  stopifnot(
    max(abs(recomputed_loss - saved_curve$corrected_loss)) < 1e-14,
    which.min(recomputed_loss) == which(saved_curve$selected)
  )
  selected_alpha <- saved_curve$alpha[saved_curve$selected]
  outer_fit <- readRDS(file.path(
    version_output_dir,
    sprintf("outer%d_final_base_fit.rds", outer_fold)
  ))
  recomputed_delta <- estimate_optout_delta(
    outer_train, outer_fit$source_pred,
    version_map, selected_alpha
  )
  finite_row <- !is.infinite(selected_alpha)
  if (finite_row) {
    stopifnot(max(abs(
      recomputed_delta$delta -
        recomputed_delta$shrinkage *
          recomputed_delta$mean_residual
    )) < 1e-14)
  } else {
    stopifnot(all(recomputed_delta$delta == 0))
  }
  recomputed_prediction <- apply_optout_delta(
    result$official_baseline_oof[
      outer_valid_rows, , drop = FALSE
    ],
    outer_valid$Case, version_map, recomputed_delta
  )$pred
  stopifnot(max(abs(
    recomputed_prediction -
      result$official_corrected_oof[
        outer_valid_rows, , drop = FALSE
      ]
  )) < 1e-14)
  audit_rows[[as.character(outer_fold)]] <- data.frame(
    outer_fold = outer_fold,
    selected_alpha = selected_alpha,
    outer_source_respondents = length(unique(outer_train$Case)),
    outer_target_respondents = length(unique(outer_valid$Case)),
    official_fold_gain =
      log_loss_matrix(
        truth[outer_valid_rows, , drop = FALSE],
        result$official_baseline_oof[
          outer_valid_rows, , drop = FALSE
        ]
      ) -
      log_loss_matrix(
        truth[outer_valid_rows, , drop = FALSE],
        recomputed_prediction
      ),
    passed = TRUE
  )
}

audit <- do.call(rbind, audit_rows)
write.csv(
  audit,
  file.path(version_output_dir, "mean_residual_audit.csv"),
  row.names = FALSE
)
cat("Mean-residual artifact audit passed:\n")
print(audit, digits = 10)
