# ==============================================================================
# 018_final_blend.R
# RF + XGBoost + Elastic-Net multinomial logit, blended.
# Respondent-grouped CV used EVERYWHERE (tuning, OOF blend weights, final
# reporting) -- verified necessary because train has 1135 respondents x 19
# tasks each, and test contains 263 COMPLETELY DIFFERENT respondents (zero
# overlap). Row-based CV lets a model key off a respondent's repeated
# demographic combination almost like a disguised ID, which inflates CV
# scores in a way that does not transfer to the leaderboard.
# ==============================================================================

library(dplyr)
library(readr)
library(ranger)
library(xgboost)
library(glmnet)

set.seed(42)

# ==============================================================================
# 1. Load data
# ==============================================================================
train_raw  <- read_csv("../data/train.csv", show_col_types = FALSE)
test_raw   <- read_csv("../data/test.csv", show_col_types = FALSE)
sample_sub <- read_csv("../data/sample_submission.csv", show_col_types = FALSE)

# ==============================================================================
# 2. Preprocessing
#    - Alternative 4 is a constant, all-zero "opt out / baseline" option in
#      this dataset (verified: every CC4..Price4 column has exactly 1 unique
#      value across both train AND test). It carries zero signal as a raw
#      feature, and worse, any derived feature computed against it (e.g.
#      Price1/Price4) is either a degenerate constant or a trivial rescaling
#      of another feature. We drop the alt-4 attribute block entirely rather
#      than let it leak into feature engineering.
#    - Several demographic fields are encoded three different ways (text
#      label / integer index / numeric proxy, e.g. income, incomeind,
#      incomea). Keeping all three is pure redundancy and inflates the
#      one-hot dimension for no benefit. We keep exactly one representation
#      per variable: the numeric "a" proxy when available (continuous,
#      preserves ordering), otherwise the text label as a factor.
#    - Ch1..Ch4 are always dropped as features (both train and test), since
#      test.csv carries these columns as empty placeholders for the target
#      -- they must never end up in the feature matrix.
# ==============================================================================
alt4_cols <- c("CC4","GN4","NS4","BU4","FA4","LD4","BZ4","FC4","FP4","RP4","PP4",
               "KA4","SC4","TS4","NV4","MA4","LB4","AF4","HU4","Price4")
redundant_demo_cols <- c("milesind","miles","nightind","night","ageind","age",
                         "incomeind","income","yearind",
                         "segmentind","genderind","educind","regionind","Urbind","pparkind")

process_data <- function(df, is_train = TRUE) {
  out <- df
  if (is_train) {
    out <- out %>%
      mutate(
        Choice = case_when(
          Ch1 == 1 ~ "Alternative_1", Ch2 == 1 ~ "Alternative_2",
          Ch3 == 1 ~ "Alternative_3", Ch4 == 1 ~ "Alternative_4"
        ),
        Choice = as.factor(Choice)
      )
  }
  out <- out %>%
    select(-any_of(c("No", "Task", "Ch1", "Ch2", "Ch3", "Ch4", alt4_cols, redundant_demo_cols)))
  out
}

train_clean <- process_data(train_raw, is_train = TRUE)
test_clean  <- process_data(test_raw, is_train = FALSE)

# Case is kept temporarily (needed to build grouped folds) but is NEVER used
# as a model feature -- it's dropped right before building X_mm below.
case_ids <- train_raw$Case

# ==============================================================================
# 3. Relational feature engineering (alternatives 1-3 ONLY)
#    Grounded in choice theory: what matters for utility is how each option
#    compares to the others on offer, not just its raw attributes.
# ==============================================================================
engineer_features <- function(df) {
  df %>% mutate(
    Diff_Price_1v2 = Price1 - Price2,
    Diff_Price_1v3 = Price1 - Price3,
    Diff_Price_2v3 = Price2 - Price3,
    Ratio_Price_1v2 = Price1 / (Price2 + 1e-5),
    Ratio_Price_1v3 = Price1 / (Price3 + 1e-5),
    Ratio_Price_2v3 = Price2 / (Price3 + 1e-5),
    Mean_Price_1to3 = (Price1 + Price2 + Price3) / 3,
    Is_Alt1_Cheapest = as.numeric(Price1 <= pmin(Price2, Price3)),
    Is_Alt2_Cheapest = as.numeric(Price2 <= pmin(Price1, Price3)),
    Is_Alt3_Cheapest = as.numeric(Price3 <= pmin(Price1, Price2)),
    Price1_to_Income = Price1 / (incomea + 1e-5),
    Price2_to_Income = Price2 / (incomea + 1e-5),
    Price3_to_Income = Price3 / (incomea + 1e-5),
    Price1_to_Age = Price1 / (agea + 1e-5),
    Price2_to_Age = Price2 / (agea + 1e-5),
    Price3_to_Age = Price3 / (agea + 1e-5)
  )
}

train_clean <- engineer_features(train_clean)
test_clean  <- engineer_features(test_clean)

# ==============================================================================
# 4. Log-loss helper
# ==============================================================================
calculate_log_loss <- function(actual_factor, predicted_probs) {
  actual_matrix <- model.matrix(~ actual_factor - 1)
  eps <- 1e-15
  predicted_probs <- pmax(pmin(predicted_probs, 1 - eps), eps)
  -mean(rowSums(actual_matrix * log(predicted_probs)))
}

# ==============================================================================
# 5. Respondent-grouped 5-fold CV (the fold assignment used by EVERY model
#    below -- tuning, OOF predictions, and blend-weight search all read
#    this same variable, so the fix applies everywhere at once).
# ==============================================================================
set.seed(42)
unique_cases <- unique(case_ids)
case_fold_lookup <- setNames(sample(rep(1:5, length.out = length(unique_cases))), unique_cases)
fold_assignments <- unname(case_fold_lookup[as.character(case_ids)])
print("Using respondent-grouped 5-fold CV (each respondent's rows stay in one fold).")
print(table(fold_assignments))

label_map <- levels(train_clean$Choice)
y_factor  <- train_clean$Choice
y_num     <- as.integer(y_factor) - 1

train_model_df <- train_clean %>% select(-Case)  # RF uses the data frame directly
test_model_df  <- test_clean  %>% select(-Case)

# ==============================================================================
# 6. Build aligned model matrices for XGBoost / glmnet (train+test combined
#    before encoding so factor levels and columns are guaranteed to match,
#    and any leftover constant column is dropped safely).
# ==============================================================================
X_all <- bind_rows(
  train_model_df %>% select(-Choice) %>% mutate(.split = "train"),
  test_model_df %>% mutate(.split = "test")
)
feat_cols <- setdiff(names(X_all), ".split")
n_unique <- sapply(feat_cols, function(cn) length(unique(na.omit(X_all[[cn]]))))
const_cols <- names(n_unique)[n_unique < 2]
if (length(const_cols) > 0) {
  print(paste("Dropping constant column(s):", paste(const_cols, collapse = ", ")))
  X_all <- X_all %>% select(-all_of(const_cols))
}
X_all_mm  <- model.matrix(~ . - 1, data = X_all %>% select(-.split))
X_mm      <- X_all_mm[X_all$.split == "train", , drop = FALSE]
X_test_mm <- X_all_mm[X_all$.split == "test", , drop = FALSE]
print(paste0("Feature matrix: ", ncol(X_mm), " columns, ", nrow(X_mm), " train rows, ", nrow(X_test_mm), " test rows."))

# ==============================================================================
# 7. RF grid search (respondent-grouped CV)
# ==============================================================================
rf_grid <- expand.grid(
  mtry = c(6, 10, 14, floor(sqrt(ncol(train_model_df) - 1))),
  min.node.size = c(5, 10, 20, 50),
  cv_log_loss = NA
) %>% distinct(mtry, min.node.size, .keep_all = TRUE)

print(paste("RF grid search over", nrow(rf_grid), "combinations..."))
for (i in 1:nrow(rf_grid)) {
  fold_losses <- numeric(5)
  for (f in 1:5) {
    val_idx <- which(fold_assignments == f)
    rf_model <- ranger(
      formula = Choice ~ ., data = train_model_df[-val_idx, ], num.trees = 500,
      mtry = rf_grid$mtry[i], min.node.size = rf_grid$min.node.size[i],
      probability = TRUE, seed = 42
    )
    val_probs <- predict(rf_model, data = train_model_df[val_idx, ])$predictions[, label_map]
    fold_losses[f] <- calculate_log_loss(train_model_df$Choice[val_idx], val_probs)
  }
  rf_grid$cv_log_loss[i] <- mean(fold_losses)
  print(paste0("RF ", i, "/", nrow(rf_grid), " | mtry: ", rf_grid$mtry[i],
               " | node: ", rf_grid$min.node.size[i], " | loss: ", round(rf_grid$cv_log_loss[i], 5)))
}
best_rf <- rf_grid[which.min(rf_grid$cv_log_loss), ]
print("Best RF params:"); print(best_rf)

# ==============================================================================
# 8. XGBoost grid search (respondent-grouped CV, via the `folds` argument)
# ==============================================================================
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
dtrain_full <- xgb.DMatrix(data = X_mm, label = y_num)
grouped_fold_list <- lapply(1:5, function(f) which(fold_assignments == f))

for (i in 1:nrow(xgb_grid)) {
  params <- list(
    objective = "multi:softprob", num_class = length(label_map), eval_metric = "mlogloss",
    max_depth = xgb_grid$max_depth[i], eta = xgb_grid$eta[i],
    min_child_weight = xgb_grid$min_child_weight[i],
    subsample = xgb_grid$subsample[i], colsample_bytree = xgb_grid$colsample_bytree[i]
  )
  cv_res <- xgb.cv(
    params = params, data = dtrain_full, nrounds = 2000,
    folds = grouped_fold_list, early_stopping_rounds = 50, verbose = 0
  )
  # xgboost's R API moved `best_iteration` into `cv_res$early_stop$best_iteration`
  # in recent versions (was top-level `cv_res$best_iteration` in older ones).
  # Check both, then fall back to deriving it directly from the evaluation
  # log so this is robust to whichever version is installed.
  best_iter <- cv_res$early_stop$best_iteration
  if (is.null(best_iter)) best_iter <- cv_res$best_iteration
  if (is.null(best_iter) || length(best_iter) == 0) {
    best_iter <- which.min(cv_res$evaluation_log$test_mlogloss_mean)
  }
  best_score <- cv_res$evaluation_log$test_mlogloss_mean[best_iter]
  
  xgb_grid$cv_log_loss[i] <- best_score
  xgb_grid$best_nrounds[i] <- best_iter
  print(paste0("XGB ", i, "/", nrow(xgb_grid),
               " | depth: ", xgb_grid$max_depth[i], " | eta: ", xgb_grid$eta[i],
               " | min_child: ", xgb_grid$min_child_weight[i],
               " | best_nrounds: ", best_iter, " | loss: ", round(best_score, 5)))
}
best_xgb <- xgb_grid[which.min(xgb_grid$cv_log_loss), ]
print("Best XGB params:"); print(best_xgb)

xgb_params_best <- list(
  objective = "multi:softprob", num_class = length(label_map), eval_metric = "mlogloss",
  max_depth = best_xgb$max_depth, eta = best_xgb$eta,
  min_child_weight = best_xgb$min_child_weight,
  subsample = best_xgb$subsample, colsample_bytree = best_xgb$colsample_bytree
)

# Helper: recent xgboost versions already return a proper [nrow, nclass]
# matrix from predict() for multiclass objectives. Older versions return a
# flat vector needing a manual byrow reshape. Handle both.
safe_xgb_predict <- function(model, dmat) {
  pred <- predict(model, dmat)
  if (!is.matrix(pred)) pred <- matrix(pred, ncol = length(label_map), byrow = TRUE)
  colnames(pred) <- label_map
  pred[, label_map]
}

# ==============================================================================
# 9. Elastic-Net multinomial logit (glmnet), alpha tuned by grouped CV too.
#    A regularized LINEAR model tends to make different mistakes than tree
#    ensembles, which is exactly what makes it valuable in a blend -- it's
#    picked for diversity, not because it beats RF/XGB alone.
# ==============================================================================
alpha_grid <- c(0, 0.25, 0.5, 0.75, 1)
alpha_results <- data.frame(alpha = alpha_grid, cv_log_loss = NA)

print("Tuning glmnet alpha (respondent-grouped CV)...")
for (a_i in seq_along(alpha_grid)) {
  oof_glm_tmp <- matrix(NA, nrow(train_clean), length(label_map), dimnames = list(NULL, label_map))
  for (f in 1:5) {
    val_idx <- which(fold_assignments == f)
    cv_fit <- cv.glmnet(
      x = X_mm[-val_idx, , drop = FALSE], y = y_factor[-val_idx],
      family = "multinomial", alpha = alpha_grid[a_i], type.measure = "deviance",
      foldid = as.integer(factor(fold_assignments[-val_idx]))  # renumbered to contiguous 1..k -- glmnet requires all fold labels present
    )
    p <- predict(cv_fit, newx = X_mm[val_idx, , drop = FALSE], s = "lambda.min", type = "response")
    oof_glm_tmp[val_idx, ] <- p[, , 1]
  }
  alpha_results$cv_log_loss[a_i] <- calculate_log_loss(y_factor, oof_glm_tmp)
  print(paste0("alpha=", alpha_grid[a_i], " | loss: ", round(alpha_results$cv_log_loss[a_i], 5)))
}
best_alpha <- alpha_results$alpha[which.min(alpha_results$cv_log_loss)]
print(paste("Best glmnet alpha:", best_alpha))

# ==============================================================================
# 10. Compute OOF predictions for all THREE models under identical folds
# ==============================================================================
oof_rf  <- matrix(NA, nrow(train_clean), length(label_map), dimnames = list(NULL, label_map))
oof_xgb <- matrix(NA, nrow(train_clean), length(label_map), dimnames = list(NULL, label_map))
oof_glm <- matrix(NA, nrow(train_clean), length(label_map), dimnames = list(NULL, label_map))

print("Computing final OOF predictions for RF, XGB, and GLM...")
for (f in 1:5) {
  val_idx <- which(fold_assignments == f)
  
  rf_model <- ranger(
    formula = Choice ~ ., data = train_model_df[-val_idx, ], num.trees = 1000,
    mtry = best_rf$mtry, min.node.size = best_rf$min.node.size,
    probability = TRUE, seed = 42
  )
  oof_rf[val_idx, ] <- predict(rf_model, data = train_model_df[val_idx, ])$predictions[, label_map]
  
  dtr  <- xgb.DMatrix(data = X_mm[-val_idx, ], label = y_num[-val_idx])
  dval <- xgb.DMatrix(data = X_mm[val_idx, ])
  xgb_model <- xgb.train(params = xgb_params_best, data = dtr, nrounds = best_xgb$best_nrounds, verbose = 0)
  oof_xgb[val_idx, ] <- safe_xgb_predict(xgb_model, dval)
  
  cv_fit <- cv.glmnet(
    x = X_mm[-val_idx, , drop = FALSE], y = y_factor[-val_idx],
    family = "multinomial", alpha = best_alpha, type.measure = "deviance",
    foldid = as.integer(factor(fold_assignments[-val_idx]))  # renumbered to contiguous 1..k
  )
  p <- predict(cv_fit, newx = X_mm[val_idx, , drop = FALSE], s = "lambda.min", type = "response")
  oof_glm[val_idx, ] <- p[, , 1]
  
  print(paste0("Fold ", f, "/5 done."))
}

print(paste0("OOF RF alone:  ", round(calculate_log_loss(y_factor, oof_rf), 5)))
print(paste0("OOF XGB alone: ", round(calculate_log_loss(y_factor, oof_xgb), 5)))
print(paste0("OOF GLM alone: ", round(calculate_log_loss(y_factor, oof_glm), 5)))

# ==============================================================================
# 11. Optimize THREE-way blend weights on a simplex grid (w_rf + w_xgb + w_glm = 1)
# ==============================================================================
step <- 0.05
best_blend <- list(loss = Inf, w_rf = NA, w_xgb = NA, w_glm = NA)
for (w_rf in seq(0, 1, by = step)) {
  for (w_xgb in seq(0, 1 - w_rf, by = step)) {
    w_glm <- 1 - w_rf - w_xgb
    blended <- w_rf * oof_rf + w_xgb * oof_xgb + w_glm * oof_glm
    loss <- calculate_log_loss(y_factor, blended)
    if (loss < best_blend$loss) {
      best_blend <- list(loss = loss, w_rf = w_rf, w_xgb = w_xgb, w_glm = w_glm)
    }
  }
}
print("---------------------------------------------------")
print(paste0("Best blend: w_rf=", best_blend$w_rf, " w_xgb=", best_blend$w_xgb,
             " w_glm=", best_blend$w_glm, " | OOF blended log loss = ", round(best_blend$loss, 5)))
print("---------------------------------------------------")

# ==============================================================================
# 12. Train final models on 100% of data and blend for the submission
# ==============================================================================
print("Retraining final models on full training data...")

final_rf <- ranger(
  formula = Choice ~ ., data = train_model_df, num.trees = 1500,
  mtry = best_rf$mtry, min.node.size = best_rf$min.node.size,
  probability = TRUE, seed = 42
)
final_rf_probs <- predict(final_rf, data = test_model_df)$predictions[, label_map]

dtrain_all <- xgb.DMatrix(data = X_mm, label = y_num)
final_xgb <- xgb.train(params = xgb_params_best, data = dtrain_all, nrounds = best_xgb$best_nrounds, verbose = 0)
final_xgb_probs <- safe_xgb_predict(final_xgb, xgb.DMatrix(data = X_test_mm))

final_glm_fit <- cv.glmnet(
  x = X_mm, y = y_factor, family = "multinomial", alpha = best_alpha,
  type.measure = "deviance", foldid = fold_assignments
)
final_glm_probs <- predict(final_glm_fit, newx = X_test_mm, s = "lambda.min", type = "response")[, , 1]
colnames(final_glm_probs) <- label_map
final_glm_probs <- final_glm_probs[, label_map]

final_probs <- best_blend$w_rf * final_rf_probs + best_blend$w_xgb * final_xgb_probs + best_blend$w_glm * final_glm_probs
final_probs <- final_probs / rowSums(final_probs)  # defensive renormalization

submission_018 <- sample_sub %>%
  mutate(
    Ch1 = final_probs[, "Alternative_1"],
    Ch2 = final_probs[, "Alternative_2"],
    Ch3 = final_probs[, "Alternative_3"],
    Ch4 = final_probs[, "Alternative_4"]
  ) %>
  select(all_of(names(sample_sub)))

write_csv(submission_018, "../results/submission_018_final_blend.csv")
print("submission_018_final_blend.csv has been generated!")