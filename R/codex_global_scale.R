source("R/codex_shared_utility_common.R")

dir.create(
  shared_utility_output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

gradient_check <- check_scale_gradient()
stopifnot(gradient_check$passed)
write_result_csv(
  gradient_check,
  file.path(shared_utility_output_dir, "scale_gradient_check.csv")
)

train <- read.csv("csv files/train.csv")
base_cache <- readRDS("data_processed/oof_ensemble_v10.rds")
truth <- base_cache$oof_truth
base_mlogit <- base_cache$oof_mlogit
original_xgb <- base_cache$oof_xgb
v11 <- 0.8 * base_mlogit + 0.2 * original_xgb
fold_map <- base_cache$fold_of_case
row_fold <- unname(fold_map[as.character(train$Case)])

mlp_oof <- readRDS(
  "data_processed/codex_behavioral_round/mlp_oof.rds"
)$oof[[1]]
mlp_precision <- readRDS(
  "data_processed/codex_behavioral_round/mlp_precision.rds"
)
current <- mlp_precision$crossfit_prediction
mlp_weight_by_fold <- mlp_precision$fold_results$mlp_weight[
  match(1:5, mlp_precision$fold_results$fold)
]
row_mlp_weight <- mlp_weight_by_fold[row_fold]
reconstructed_current <-
  (1 - row_mlp_weight) * v11 + row_mlp_weight * mlp_oof
stopifnot(
  max(abs(reconstructed_current - current)) < 1e-12,
  abs(log_loss_matrix(truth, base_mlogit) - 1.147021) < 1e-6,
  abs(log_loss_matrix(truth, v11) - 1.145094) < 1e-6,
  abs(log_loss_matrix(truth, current) - 1.143789442) < 1e-8
)

utility <- log(pmax(base_mlogit, 1e-15))

make_scale_design <- function(fit_rows, apply_rows) {
  income_center <- mean(train$incomea[fit_rows])
  income_scale <- sd(train$incomea[fit_rows])
  stopifnot(is.finite(income_scale), income_scale > 0)
  cbind(
    global = 1,
    Task_c = (as.numeric(train$Task[apply_rows]) - 10) / 9,
    income_z =
      (as.numeric(train$incomea[apply_rows]) - income_center) /
      income_scale,
    segment2 = as.integer(train$segmentind[apply_rows] == 2L),
    segment3 = as.integer(train$segmentind[apply_rows] == 3L),
    segment4 = as.integer(train$segmentind[apply_rows] == 4L),
    segment5 = as.integer(train$segmentind[apply_rows] == 5L),
    segment6 = as.integer(train$segmentind[apply_rows] == 6L)
  )
}

ridges <- c(0, 0.001, 0.01)
summary_rows <- list()
coefficient_rows <- list()
prediction_list <- list()
bootstrap_rows <- list()

for (ridge in ridges) {
  scaled_mlogit <- matrix(NA_real_, nrow(train), 4L)
  coefficients <- matrix(
    NA_real_,
    nrow = 5L,
    ncol = 8L,
    dimnames = list(
      paste0("fold", 1:5),
      colnames(make_scale_design(row_fold != 1L, row_fold == 1L))
    )
  )
  for (fold in 1:5) {
    fit_rows <- which(row_fold != fold)
    valid_rows <- which(row_fold == fold)
    fit_design <- make_scale_design(fit_rows, fit_rows)
    valid_design <- make_scale_design(fit_rows, valid_rows)
    fitted <- fit_scale_model(
      utility[fit_rows, , drop = FALSE],
      truth[fit_rows, , drop = FALSE],
      fit_design,
      ridge = ridge
    )
    coefficients[fold, ] <- fitted$par
    predicted <- scale_softmax_evaluate(
      fitted$par,
      utility[valid_rows, , drop = FALSE],
      truth[valid_rows, , drop = FALSE],
      valid_design,
      ridge = 0
    )$probability
    scaled_mlogit[valid_rows, ] <- predicted
    cat(sprintf(
      "ridge %.3g fold %d: scale mlogit %.9f, gamma norm %.5f\n",
      ridge, fold,
      log_loss_matrix(
        truth[valid_rows, , drop = FALSE], predicted
      ),
      sqrt(sum(fitted$par^2))
    ))
  }
  stopifnot(!anyNA(scaled_mlogit))
  replacement_v11 <- 0.8 * scaled_mlogit + 0.2 * original_xgb
  adjusted_current <-
    (1 - row_mlp_weight) * replacement_v11 +
    row_mlp_weight * mlp_oof
  label <- sprintf("ridge_%g", ridge)
  summary_rows[[label]] <- data.frame(
    candidate = label,
    ridge = ridge,
    base_mlogit_logloss = log_loss_matrix(truth, base_mlogit),
    scaled_mlogit_logloss = log_loss_matrix(truth, scaled_mlogit),
    mlogit_gain =
      log_loss_matrix(truth, base_mlogit) -
      log_loss_matrix(truth, scaled_mlogit),
    v11_logloss = log_loss_matrix(truth, v11),
    replacement_v11_logloss =
      log_loss_matrix(truth, replacement_v11),
    replacement_v11_gain =
      log_loss_matrix(truth, v11) -
      log_loss_matrix(truth, replacement_v11),
    current_logloss = log_loss_matrix(truth, current),
    adjusted_current_logloss =
      log_loss_matrix(truth, adjusted_current),
    adjusted_current_gain =
      log_loss_matrix(truth, current) -
      log_loss_matrix(truth, adjusted_current)
  )
  coefficient_rows[[label]] <- do.call(rbind, lapply(1:5, function(fold) {
    data.frame(
      candidate = label,
      fold = fold,
      term = colnames(coefficients),
      coefficient = coefficients[fold, ]
    )
  }))
  prediction_list[[label]] <- list(
    scaled_mlogit = scaled_mlogit,
    replacement_v11 = replacement_v11,
    adjusted_current = adjusted_current
  )
  bootstrap <- respondent_bootstrap_comparison(
    truth,
    current,
    adjusted_current,
    train$Case
  )
  bootstrap$candidate <- label
  bootstrap_rows[[label]] <- bootstrap
}

summary <- do.call(rbind, summary_rows)
coefficients <- do.call(rbind, coefficient_rows)
bootstrap <- do.call(rbind, bootstrap_rows)
write_result_csv(
  summary,
  file.path(shared_utility_output_dir, "global_scale_summary.csv")
)
write_result_csv(
  coefficients,
  file.path(shared_utility_output_dir, "global_scale_coefficients.csv")
)
write_result_csv(
  bootstrap,
  file.path(shared_utility_output_dir, "global_scale_bootstrap.csv")
)
saveRDS(
  list(
    summary = summary,
    coefficients = coefficients,
    bootstrap = bootstrap,
    predictions = prediction_list
  ),
  file.path(shared_utility_output_dir, "global_scale.rds")
)
print(summary, digits = 9)
print(bootstrap, digits = 9)
