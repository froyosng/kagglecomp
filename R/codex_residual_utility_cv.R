source("R/codex_shared_utility_common.R")

dir.create(
  shared_utility_output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

screen <- read.csv(file.path(
  shared_utility_output_dir,
  "shared_utility_screen.csv"
))
frozen <- screen[
  screen$config == "residual_d1" & screen$rounds == 200L, ,
  drop = FALSE
]
stopifnot(
  nrow(frozen) == 1L,
  frozen$primary_gain_vs_v11 > 0
)

train <- read.csv("csv files/train.csv")
train_long <- reshape_choice_long(train)
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

stopifnot(
  abs(log_loss_matrix(truth, base_mlogit) - 1.147021) < 1e-6,
  abs(log_loss_matrix(truth, v11) - 1.145094) < 1e-6,
  abs(log_loss_matrix(truth, current) - 1.143789442) < 1e-8
)

params <- shared_xgb_params(
  max_depth = 1L,
  eta = 0.03,
  min_child_weight = 50,
  lambda = 30
)
nrounds <- 200L
oof_residual <- matrix(NA_real_, nrow(train), 4L)
fold_rows <- list()
importance_rows <- list()

for (fold in 1:5) {
  cat(sprintf(
    "\nResidual exact-softmax CV fold %d/5: fitting m8trpg offset...\n",
    fold
  ))
  valid_cases <- as.integer(names(fold_map)[fold_map == fold])
  fold_train <- ensure_choice_long(train_long[
    !(train_long$Case %in% valid_cases), ,
    drop = FALSE
  ])
  fold_valid <- ensure_choice_long(train_long[
    train_long$Case %in% valid_cases, ,
    drop = FALSE
  ])
  valid_rows <- which(row_fold == fold)
  valid_no <- train$No[valid_rows]

  base_fit <- fit_m8trpg_model(fold_train)
  train_offset <- predict_m8trpg_margin(base_fit, fold_train)
  valid_offset <- predict_m8trpg_margin(base_fit, fold_valid)
  fold_base_pred <- aligned_prediction(valid_offset, valid_no)
  stopifnot(
    max(abs(
      fold_base_pred -
        base_mlogit[valid_rows, , drop = FALSE]
    )) < 1e-7
  )

  cat("Fitting frozen depth-1 residual correction...\n")
  model <- fit_group_softmax_xgb(
    fold_train,
    params = params,
    nrounds = nrounds,
    base_margin = train_offset$margin
  )
  residual_prediction <- predict_group_softmax_xgb(
    model,
    fold_valid,
    base_margin = valid_offset$margin,
    nrounds = nrounds
  )$pred
  stopifnot(nrow(residual_prediction) == length(valid_rows))
  oof_residual[valid_rows, ] <- residual_prediction

  fold_rows[[fold]] <- data.frame(
    fold = fold,
    base_mlogit_logloss = log_loss_matrix(
      truth[valid_rows, , drop = FALSE],
      base_mlogit[valid_rows, , drop = FALSE]
    ),
    residual_mlogit_logloss = log_loss_matrix(
      truth[valid_rows, , drop = FALSE],
      residual_prediction
    )
  )
  fold_rows[[fold]]$mlogit_gain <-
    fold_rows[[fold]]$base_mlogit_logloss -
    fold_rows[[fold]]$residual_mlogit_logloss
  importance <- xgb.importance(
    feature_names = colnames(shared_feature_matrix(fold_train)),
    model = model
  )
  if (nrow(importance) > 0L) {
    importance$fold <- fold
    importance_rows[[fold]] <- as.data.frame(importance)
  }
  saveRDS(
    list(
      completed_fold = fold,
      oof_residual = oof_residual,
      fold_results = fold_rows,
      importance = importance_rows
    ),
    file.path(
      shared_utility_output_dir,
      "residual_utility_cv_checkpoint.rds"
    )
  )
  cat(sprintf(
    "fold %d: base %.9f, residual %.9f, gain %+.9f\n",
    fold,
    fold_rows[[fold]]$base_mlogit_logloss,
    fold_rows[[fold]]$residual_mlogit_logloss,
    fold_rows[[fold]]$mlogit_gain
  ))
  flush.console()
}
stopifnot(!anyNA(oof_residual))

replacement_v11 <- 0.8 * oof_residual + 0.2 * original_xgb
adjusted_current <-
  (1 - row_mlp_weight) * replacement_v11 +
  row_mlp_weight * mlp_oof
diagnostic_blend <- crossfit_two_way_blend(
  current,
  adjusted_current,
  truth,
  row_fold,
  weights = seq(0, 1, by = 0.01)
)

summary <- data.frame(
  config = "residual_d1_r200",
  base_mlogit_logloss = log_loss_matrix(truth, base_mlogit),
  residual_mlogit_logloss = log_loss_matrix(truth, oof_residual),
  mlogit_gain =
    log_loss_matrix(truth, base_mlogit) -
    log_loss_matrix(truth, oof_residual),
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
    log_loss_matrix(truth, adjusted_current),
  diagnostic_crossfit_blend_logloss =
    diagnostic_blend$logloss,
  diagnostic_crossfit_blend_gain =
    log_loss_matrix(truth, current) -
    diagnostic_blend$logloss,
  diagnostic_fold_candidate_weights =
    paste(
      sprintf("%.2f", diagnostic_blend$fold_weights),
      collapse = "/"
    )
)
bootstrap <- respondent_bootstrap_comparison(
  truth,
  current,
  adjusted_current,
  train$Case
)
bootstrap$comparison <- "adjusted_current_vs_current"
diagnostic_bootstrap <- respondent_bootstrap_comparison(
  truth,
  current,
  diagnostic_blend$pred,
  train$Case
)
diagnostic_bootstrap$comparison <-
  "diagnostic_crossfit_blend_vs_current"
bootstrap <- rbind(bootstrap, diagnostic_bootstrap)

fold_results <- do.call(rbind, fold_rows)
importance <- if (length(importance_rows) > 0L) {
  do.call(rbind, importance_rows)
} else {
  data.frame()
}
write_result_csv(
  summary,
  file.path(shared_utility_output_dir, "residual_utility_cv_summary.csv")
)
write_result_csv(
  fold_results,
  file.path(shared_utility_output_dir, "residual_utility_cv_folds.csv")
)
if (nrow(importance) > 0L) {
  write_result_csv(
    importance,
    file.path(
      shared_utility_output_dir,
      "residual_utility_importance.csv"
    )
  )
}
write_result_csv(
  bootstrap,
  file.path(shared_utility_output_dir, "residual_utility_bootstrap.csv")
)
saveRDS(
  list(
    summary = summary,
    fold_results = fold_results,
    bootstrap = bootstrap,
    oof_residual = oof_residual,
    replacement_v11 = replacement_v11,
    adjusted_current = adjusted_current,
    diagnostic_blend = diagnostic_blend,
    importance = importance
  ),
  file.path(shared_utility_output_dir, "residual_utility_cv.rds")
)
print(summary, digits = 9)
print(fold_results, digits = 9)
print(bootstrap, digits = 9)
