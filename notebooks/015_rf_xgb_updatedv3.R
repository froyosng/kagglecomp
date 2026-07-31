# ==============================================================================
# 012_rf_xgb_ensemble.R
# Random Forest + XGBoost, Respondent-Grouped 5-Fold CV, CV-weighted blend
# ==============================================================================

library(tidyverse)
library(ranger)
library(xgboost)

# 1. Load Data
train_raw <- read_csv("../data/train.csv")
test_raw  <- read_csv("../data/test.csv")
sample_sub <- read_csv("../data/sample_submission.csv")

# 2. Data Preprocessing
process_data <- function(df, is_train = TRUE) {
  df_processed <- df
  if (is_train) {
    df_processed <- df_processed %>%
      mutate(
        Choice = case_when(
          Ch1 == 1 ~ "Alternative_1",
          Ch2 == 1 ~ "Alternative_2",
          Ch3 == 1 ~ "Alternative_3",
          Ch4 == 1 ~ "Alternative_4"
        ),
        Choice = as.factor(Choice)
      ) %>%
      select(-Ch1, -Ch2, -Ch3, -Ch4)
  }
  df_processed <- df_processed %>% select(-any_of(c("Case", "No", "Task", "Ch1", "Ch2", "Ch3", "Ch4")))
  return(df_processed)
}

train_clean <- process_data(train_raw, is_train = TRUE)
test_clean  <- process_data(test_raw, is_train = FALSE)

# Keep the respondent/task identifier around (outside train_clean) purely for
# building a grouped-CV sanity check later. It is never used as a model
# feature -- process_data() already strips it from train_clean/test_clean.
case_ids <- if ("Case" %in% names(train_raw)) train_raw$Case else NULL
if (is.null(case_ids)) {
  print("No 'Case' column found in train_raw -- grouped-CV sanity check will be skipped.")
}

# Multi-Class Log Loss Helper
calculate_log_loss <- function(actual_factor, predicted_probs) {
  actual_matrix <- model.matrix(~ actual_factor - 1)
  eps <- 1e-15
  predicted_probs <- pmax(pmin(predicted_probs, 1 - eps), eps)
  -mean(rowSums(actual_matrix * log(predicted_probs)))
}

set.seed(42)
# CRITICAL FIX: folds must be grouped by respondent (Case), not assigned
# row-by-row. Each respondent answers many tasks with identical covariates
# (income, age, segment, etc.), so row-based folds let a flexible model key
# off a respondent's recurring covariate combination -- effectively a
# disguised respondent ID -- rather than learning genuine transferable
# preference patterns. That inflates CV performance during BOTH
# hyperparameter tuning and blend-weight selection, not just the final
# reported number, so the fix has to live here, upstream of everything else.
if (!is.null(case_ids)) {
  unique_cases <- unique(case_ids)
  case_fold_lookup <- setNames(sample(rep(1:5, length.out = length(unique_cases))), unique_cases)
  fold_assignments <- unname(case_fold_lookup[as.character(case_ids)])
  print("Using respondent-grouped 5-fold CV (each respondent's rows stay entirely in one fold).")
} else {
  fold_assignments <- sample(rep(1:5, length.out = nrow(train_clean)))
  print("WARNING: no 'Case' column found -- falling back to row-based folds.")
  print("CV numbers below may be optimistic if multiple rows share a respondent.")
}

# ==============================================================================
# 3. RF grid search (respondent-grouped CV -- see fold_assignments above)
# ==============================================================================
rf_grid <- expand.grid(
  mtry = c(2, 4, 6, 8, floor(sqrt(ncol(train_clean) - 1))),
  min.node.size = c(5, 10, 20, 50),
  cv_log_loss = NA
) %>% distinct(mtry, min.node.size, .keep_all = TRUE)

print(paste("RF grid search over", nrow(rf_grid), "combinations..."))

for (i in 1:nrow(rf_grid)) {
  fold_losses <- numeric(5)
  for (f in 1:5) {
    val_idx <- which(fold_assignments == f)
    fold_train <- train_clean[-val_idx, ]
    fold_val   <- train_clean[val_idx, ]
    rf_model <- ranger(
      formula = Choice ~ ., data = fold_train, num.trees = 500,
      mtry = rf_grid$mtry[i], min.node.size = rf_grid$min.node.size[i],
      probability = TRUE, seed = 42
    )
    val_probs <- predict(rf_model, data = fold_val)$predictions
    val_probs <- val_probs[, levels(train_clean$Choice)]
    fold_losses[f] <- calculate_log_loss(fold_val$Choice, val_probs)
  }
  rf_grid$cv_log_loss[i] <- mean(fold_losses)
  print(paste0("RF ", i, "/", nrow(rf_grid), " | loss: ", round(rf_grid$cv_log_loss[i], 5)))
}

best_rf <- rf_grid[which.min(rf_grid$cv_log_loss), ]
print("Best RF params:"); print(best_rf)

# ==============================================================================
# 4. XGBoost grid search
#    Gradient boosting typically beats RF on log-loss for tabular choice data,
#    because it directly optimizes a smooth loss (mlogloss) rather than the
#    Gini/variance splits RF uses internally.
# ==============================================================================

# xgboost needs numeric labels 0..(m-1) and a numeric feature matrix
label_map <- levels(train_clean$Choice)
y_num <- as.integer(train_clean$Choice) - 1

X <- train_clean %>% select(-Choice)

# Combine train + test BEFORE encoding so factor levels/columns are guaranteed
# to line up, and so we can safely handle any constant (single-level) column
# instead of letting model.matrix error out on it.
X_all <- bind_rows(
  X %>% mutate(.split = "train"),
  test_clean %>% mutate(.split = "test")
)

# Drop ANY column with fewer than 2 unique non-NA values, regardless of its
# type (numeric, character, factor, or logical all hit the same
# model.matrix contrast error if they're constant). Checking by type alone
# is fragile -- e.g. a logical (TRUE/FALSE) column that's constant slips
# past a character/factor-only filter, which is what happened above.
all_feature_cols <- setdiff(names(X_all), ".split")
n_unique <- sapply(all_feature_cols, function(cn) length(unique(na.omit(X_all[[cn]]))))
const_cols <- names(n_unique)[n_unique < 2]
if (length(const_cols) > 0) {
  print(paste("Dropping constant column(s) with <2 unique values:", paste(const_cols, collapse = ", ")))
  X_all <- X_all %>% select(-all_of(const_cols))
}

X_all_mm <- model.matrix(~ . - 1, data = X_all %>% select(-.split))
X_mm      <- X_all_mm[X_all$.split == "train", , drop = FALSE]
X_test_mm <- X_all_mm[X_all$.split == "test", , drop = FALSE]

xgb_grid <- expand.grid(
  max_depth = c(3, 4, 6),
  eta = c(0.02, 0.05, 0.1),
  min_child_weight = c(1, 5, 10),
  subsample = 0.8,
  colsample_bytree = 0.8,
  cv_log_loss = NA,
  best_nrounds = NA
)

print(paste("XGBoost grid search over", nrow(xgb_grid), "combinations..."))

for (i in 1:nrow(xgb_grid)) {
  params <- list(
    objective = "multi:softprob",
    num_class = length(label_map),
    eval_metric = "mlogloss",
    max_depth = xgb_grid$max_depth[i],
    eta = xgb_grid$eta[i],
    min_child_weight = xgb_grid$min_child_weight[i],
    subsample = xgb_grid$subsample[i],
    colsample_bytree = xgb_grid$colsample_bytree[i]
  )
  
  dtrain_full <- xgb.DMatrix(data = X_mm, label = y_num)
  
  cv_res <- xgb.cv(
    params = params,
    data = dtrain_full,
    nrounds = 2000,
    folds = lapply(1:5, function(f) which(fold_assignments == f)),
    early_stopping_rounds = 50,
    verbose = 0
  )
  
  # xgboost's R API moved `best_iteration` from the top level of the cv
  # object into `cv_res$early_stop$best_iteration` in recent package
  # versions. Check both locations so this works regardless of which
  # xgboost version is installed.
  best_iter <- cv_res$early_stop$best_iteration
  if (is.null(best_iter)) best_iter <- cv_res$best_iteration
  if (is.null(best_iter) || length(best_iter) == 0) {
    # Fallback: derive it directly from the evaluation log instead of
    # relying on the early-stop object at all. This is version-proof.
    best_iter <- which.min(cv_res$evaluation_log$test_mlogloss_mean)
  }
  best_score <- cv_res$evaluation_log$test_mlogloss_mean[best_iter]
  
  xgb_grid$cv_log_loss[i] <- best_score
  xgb_grid$best_nrounds[i] <- best_iter
  
  print(paste0("XGB ", i, "/", nrow(xgb_grid),
               " | depth: ", xgb_grid$max_depth[i],
               " | eta: ", xgb_grid$eta[i],
               " | min_child: ", xgb_grid$min_child_weight[i],
               " | best_nrounds: ", best_iter,
               " | loss: ", round(best_score, 5)))
}

best_xgb <- xgb_grid[which.min(xgb_grid$cv_log_loss), ]
print("Best XGB params:"); print(best_xgb)

# ==============================================================================
# 5. Compute out-of-fold predictions for BOTH models with their best params,
#    then find the optimal blend weight on held-out folds (not on training loss)
# ==============================================================================
oof_rf  <- matrix(NA, nrow(train_clean), length(label_map), dimnames = list(NULL, label_map))
oof_xgb <- matrix(NA, nrow(train_clean), length(label_map), dimnames = list(NULL, label_map))

xgb_params_best <- list(
  objective = "multi:softprob",
  num_class = length(label_map),
  eval_metric = "mlogloss",
  max_depth = best_xgb$max_depth,
  eta = best_xgb$eta,
  min_child_weight = best_xgb$min_child_weight,
  subsample = best_xgb$subsample,
  colsample_bytree = best_xgb$colsample_bytree
)

for (f in 1:5) {
  val_idx <- which(fold_assignments == f)
  
  # RF
  rf_model <- ranger(
    formula = Choice ~ ., data = train_clean[-val_idx, ], num.trees = 500,
    mtry = best_rf$mtry, min.node.size = best_rf$min.node.size,
    probability = TRUE, seed = 42
  )
  oof_rf[val_idx, ] <- predict(rf_model, data = train_clean[val_idx, ])$predictions[, label_map]
  
  # XGB
  dtr <- xgb.DMatrix(data = X_mm[-val_idx, ], label = y_num[-val_idx])
  dval <- xgb.DMatrix(data = X_mm[val_idx, ])
  xgb_model <- xgb.train(params = xgb_params_best, data = dtr, nrounds = best_xgb$best_nrounds, verbose = 0)
  xgb_pred <- predict(xgb_model, dval)
  # Recent xgboost versions already return an [nrows, ngroups] MATRIX for
  # multiclass objectives -- wrapping that in matrix(..., byrow=TRUE) would
  # flatten and re-shuffle the values incorrectly. Only reshape manually if
  # predict() gave back a flat vector (older xgboost versions).
  if (!is.matrix(xgb_pred)) {
    xgb_pred <- matrix(xgb_pred, ncol = length(label_map), byrow = TRUE)
  }
  colnames(xgb_pred) <- label_map
  oof_xgb[val_idx, ] <- xgb_pred[, label_map]
}

# Search blend weight w on a fine grid: w * xgb + (1-w) * rf
blend_grid <- seq(0, 1, by = 0.02)
blend_losses <- sapply(blend_grid, function(w) {
  blended <- w * oof_xgb + (1 - w) * oof_rf
  calculate_log_loss(train_clean$Choice, blended)
})
best_w <- blend_grid[which.min(blend_losses)]
print(paste0("Best blend weight (xgb weight) = ", best_w,
             " | OOF blended log loss = ", round(min(blend_losses), 5)))
print(paste0("OOF RF alone (respondent-grouped CV): ", round(calculate_log_loss(train_clean$Choice, oof_rf), 5)))
print(paste0("OOF XGB alone (respondent-grouped CV): ", round(calculate_log_loss(train_clean$Choice, oof_xgb), 5)))

# ==============================================================================
# 6. Train final models on 100% of data and blend for the submission
# ==============================================================================
print("Retraining final models on full training data...")

final_rf <- ranger(
  formula = Choice ~ ., data = train_clean, num.trees = 1500,
  mtry = best_rf$mtry, min.node.size = best_rf$min.node.size,
  probability = TRUE, seed = 42
)
final_rf_probs <- predict(final_rf, data = test_clean)$predictions[, label_map]

dtrain_all <- xgb.DMatrix(data = X_mm, label = y_num)
final_xgb <- xgb.train(params = xgb_params_best, data = dtrain_all, nrounds = best_xgb$best_nrounds, verbose = 0)
final_xgb_pred <- predict(final_xgb, xgb.DMatrix(data = X_test_mm))
if (!is.matrix(final_xgb_pred)) {
  final_xgb_pred <- matrix(final_xgb_pred, ncol = length(label_map), byrow = TRUE)
}
colnames(final_xgb_pred) <- label_map
final_xgb_probs <- final_xgb_pred[, label_map]

final_probs <- best_w * final_xgb_probs + (1 - best_w) * final_rf_probs
# renormalize defensively (should already sum to ~1)
final_probs <- final_probs / rowSums(final_probs)

submission_012 <- sample_sub %>%
  mutate(
    Ch1 = final_probs[, "Alternative_1"],
    Ch2 = final_probs[, "Alternative_2"],
    Ch3 = final_probs[, "Alternative_3"],
    Ch4 = final_probs[, "Alternative_4"]
  ) %>%
  select(all_of(names(sample_sub)))

write_csv(submission_012, "../results/submission_015_rf_xgb_blend.csv")
print("submission_015_rf_xgb_blend.csv has been generated!")