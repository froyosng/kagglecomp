# ==============================================================================
# LIBRARIES & SETUP
# ==============================================================================
library(dplyr)
library(xgboost)
library(glmnet)
library(readr)

# ==============================================================================
# 1. ADVANCED FEATURE ENGINEERING (Relational + Demographics)
# ==============================================================================
engineer_relational_features <- function(df) {
  df <- df %>%
    mutate(
      # DELTAS (Absolute Differences)
      Diff_Price_1v2 = Price1 - Price2,
      Diff_Price_1v3 = Price1 - Price3,
      Diff_Price_1v4 = Price1 - Price4,
      Diff_Price_2v3 = Price2 - Price3,
      Diff_Price_2v4 = Price2 - Price4,
      Diff_Price_3v4 = Price3 - Price4,
      
      # RATIOS (Proportional Differences)
      Ratio_Price_1v2 = Price1 / (Price2 + 1e-5),
      Ratio_Price_1v3 = Price1 / (Price3 + 1e-5),
      Ratio_Price_1v4 = Price1 / (Price4 + 1e-5),
      Ratio_Price_2v3 = Price2 / (Price3 + 1e-5),
      Ratio_Price_2v4 = Price2 / (Price4 + 1e-5),
      Ratio_Price_3v4 = Price3 / (Price4 + 1e-5),
      
      # CONTEXTUAL AGGREGATES
      Mean_Price_Offered = (Price1 + Price2 + Price3 + Price4) / 4,
      
      Is_Alt1_Cheapest = as.numeric(Price1 == pmin(Price1, Price2, Price3, Price4)),
      Is_Alt2_Cheapest = as.numeric(Price2 == pmin(Price1, Price2, Price3, Price4)),
      Is_Alt3_Cheapest = as.numeric(Price3 == pmin(Price1, Price2, Price3, Price4)),
      Is_Alt4_Cheapest = as.numeric(Price4 == pmin(Price1, Price2, Price3, Price4)),
      
      # DEMOGRAPHIC CROSS-FEATURES (Price Sensitivity)
      Price1_to_Income = Price1 / (incomea + 1e-5),
      Price2_to_Income = Price2 / (incomea + 1e-5),
      Price3_to_Income = Price3 / (incomea + 1e-5),
      Price4_to_Income = Price4 / (incomea + 1e-5),
      
      Price1_to_Age = Price1 / (agea + 1e-5),
      Price2_to_Age = Price2 / (agea + 1e-5),
      Price3_to_Age = Price3 / (agea + 1e-5),
      Price4_to_Age = Price4 / (agea + 1e-5)
    )
  return(df)
}

print("Engineering advanced relational and demographic features...")
train_clean <- engineer_relational_features(train_clean)
test_clean <- engineer_relational_features(test_clean)

# ==============================================================================
# 2. PREPARE MATRICES FOR MODELING
# ==============================================================================
y_factor <- as.factor(train_clean$Choice)
y_num <- as.integer(y_factor) - 1 

# Create matrices
X_mm <- model.matrix(Choice ~ . - 1, data = train_clean)
X_test_mm <- model.matrix(~ . - 1, data = test_clean)

# Align columns to prevent "non-conformable arguments" error
missing_cols <- setdiff(colnames(X_mm), colnames(X_test_mm))
if(length(missing_cols) > 0) {
  missing_mat <- matrix(0, nrow = nrow(X_test_mm), ncol = length(missing_cols))
  colnames(missing_mat) <- missing_cols
  X_test_mm <- cbind(X_test_mm, missing_mat)
}
X_test_mm <- X_test_mm[, colnames(X_mm), drop = FALSE]

dtrain_all <- xgb.DMatrix(data = X_mm, label = y_num)
label_map <- levels(y_factor) 

# ==============================================================================
# 3. GLMNET (ELASTIC NET) CROSS-VALIDATION
# ==============================================================================
print("Starting CV for GLM (Elastic Net)...")
oof_glm <- matrix(NA, nrow(train_clean), length(label_map), dimnames = list(NULL, label_map))

for (f in 1:5) {
  val_idx <- which(fold_assignments == f)
  
  # Alpha = 0.5 (Elastic Net)
  cv_fit <- cv.glmnet(
    x = X_mm[-val_idx, , drop = FALSE], 
    y = y_factor[-val_idx], 
    family = "multinomial",
    alpha = 0.5,               
    type.measure = "deviance",
    nfolds = 5 
  )
  
  glm_preds_3d <- predict(cv_fit, newx = X_mm[val_idx, , drop = FALSE], s = "lambda.min", type = "response")
  oof_glm[val_idx, ] <- glm_preds_3d[, , 1]
}
glm_loss <- calculate_log_loss(train_clean$Choice, oof_glm)
print(paste0("OOF GLM (Elastic Net) alone log-loss: ", round(glm_loss, 5)))

# ==============================================================================
# 4. XGBOOST GRID SEARCH & CROSS-VALIDATION
# ==============================================================================
print("Running Grid Search for XGBoost on new feature set...")
xgb_grid <- expand.grid(
  max_depth = c(3, 4, 6),
  eta = c(0.02, 0.05),
  min_child_weight = c(5, 10)
)

best_xgb_loss <- Inf
best_xgb_params <- NULL
best_oof_xgb <- matrix(NA, nrow(train_clean), length(label_map), dimnames = list(NULL, label_map))
best_nrounds_final <- 0

for (i in 1:nrow(xgb_grid)) {
  params <- list(
    objective = "multi:softprob",
    num_class = length(label_map),
    eval_metric = "mlogloss",
    max_depth = xgb_grid$max_depth[i],
    eta = xgb_grid$eta[i],
    min_child_weight = xgb_grid$min_child_weight[i],
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  
  # Fast CV to find optimal rounds for these parameters
  xgb_cv <- xgb.cv(
    params = params,
    data = dtrain_all,
    nrounds = 1000,
    nfold = 5,
    early_stopping_rounds = 50,
    verbose = 0
  )
  
  min_loss <- min(xgb_cv$evaluation_log$test_mlogloss_mean)
  
  if (min_loss < best_xgb_loss) {
    best_xgb_loss <- min_loss
    best_xgb_params <- params
    
    # Fallback if best_iteration is NULL or empty
    if (is.null(xgb_cv$best_iteration) || length(xgb_cv$best_iteration) == 0) {
      best_nrounds_final <- which.min(xgb_cv$evaluation_log$test_mlogloss_mean)
    } else {
      best_nrounds_final <- xgb_cv$best_iteration
    }
  }
}

print(paste0("Best XGB alone log-loss: ", round(best_xgb_loss, 5)))
print(paste0("Best nrounds selected: ", best_nrounds_final))

# Generate proper OOF predictions for the best XGBoost model
print("Generating OOF predictions for best XGB model...")
for (f in 1:5) {
  val_idx <- which(fold_assignments == f)
  fold_xgb <- xgb.train(
    params = best_xgb_params, 
    data = xgb.DMatrix(data = X_mm[-val_idx, , drop = FALSE], label = y_num[-val_idx]), 
    nrounds = best_nrounds_final, 
    verbose = 0
  )
  xgb_preds <- predict(fold_xgb, xgb.DMatrix(data = X_mm[val_idx, , drop = FALSE]))
  
  if (!is.matrix(xgb_preds)) xgb_preds <- matrix(xgb_preds, ncol = length(label_map), byrow = TRUE)
  colnames(xgb_preds) <- label_map
  best_oof_xgb[val_idx, ] <- xgb_preds
}

# ==============================================================================
# 5. OPTIMIZE BLEND WEIGHTS
# ==============================================================================
blend_grid <- seq(0, 1, by = 0.01)
blend_losses <- sapply(blend_grid, function(w) {
  calculate_log_loss(train_clean$Choice, w * best_oof_xgb + (1 - w) * oof_glm)
})

best_w <- blend_grid[which.min(blend_losses)]
print(paste0("Best blend weight (XGB weight) = ", best_w, " | OOF Blended Log Loss = ", round(min(blend_losses), 5)))

# ==============================================================================
# 6. TRAIN FINAL MODELS & GENERATE SUBMISSION
# ==============================================================================
print("Retraining final models on full training data...")

final_glm <- cv.glmnet(
  x = X_mm, 
  y = y_factor, 
  family = "multinomial", 
  alpha = 0.5, 
  type.measure = "deviance"
)
final_glm_probs <- predict(final_glm, newx = X_test_mm, s = "lambda.min", type = "response")[, , 1]
colnames(final_glm_probs) <- label_map

final_xgb <- xgb.train(
  params = best_xgb_params, 
  data = dtrain_all, 
  nrounds = best_nrounds_final, 
  verbose = 0
)
final_xgb_pred <- predict(final_xgb, xgb.DMatrix(data = X_test_mm))
if (!is.matrix(final_xgb_pred)) final_xgb_pred <- matrix(final_xgb_pred, ncol = length(label_map), byrow = TRUE)
colnames(final_xgb_pred) <- label_map
final_xgb_probs <- final_xgb_pred[, label_map]

# Blend and normalize
final_probs <- best_w * final_xgb_probs + (1 - best_w) * final_glm_probs
final_probs <- final_probs / rowSums(final_probs)

sample_sub <- read_csv("sample_submission.csv")
submission_017 <- sample_sub %>%
  mutate(
    Ch1 = final_probs[, "Alternative_1"],
    Ch2 = final_probs[, "Alternative_2"],
    Ch3 = final_probs[, "Alternative_3"],
    Ch4 = final_probs[, "Alternative_4"]
  )

write_csv(submission_017, "submission_017_advanced_blend.csv")
print("submission_017_advanced_blend.csv has been generated!")