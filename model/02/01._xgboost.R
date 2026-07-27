# METRIC FUNCTIONS
compute_logloss <- function(P, choice){
    # P: predicted choices
    # choice: actual choices
    logloss <- 0
    for (i in 1:(nrow(choice))) {
        logloss <- logloss + choice$Ch1[i]*log(P[i,1]) + choice$Ch2[i]*log(P[i,2]) + choice$Ch3[i]*log(P[i,3]) + choice$Ch4[i]*log(P[i,4])
    }
    logloss <- logloss / nrow(choice)
    logloss <- -1 * logloss
    return(logloss)
}

compute_accuracy <- function(P, choice){
    # P: predicted choices
    # choice: actual choices
    # Determine the predicted choice by taking the column index of the max probability in each row
    predicted_choice <- apply(P, 1, which.max)
    
    # Determine the actual choice by taking the column index of the max probability in each row
    actual_choice <- apply(choice, 1, which.max)
    
    # Compute the confusion matrix
    confusion_matrix <- table(Predicted = predicted_choice, Actual = actual_choice)
    
    (sum(diag(confusion_matrix)))/(sum((confusion_matrix)))
}


# ==================================================================================
# [1] XGBOOST
# ==================================================================================
library(xgboost)
library(caTools)
library(dplyr)
library(caret)

# DATA EXTRACTION
df_train <- read.csv("train.csv")
df_train$Choice <- ifelse(df_train$Ch1 == 1, 1, ifelse(df_train$Ch2 == 1, 2, ifelse(df_train$Ch3 == 1, 3, 4)))
choice <- subset(df_train, select = c(Ch1, Ch2, Ch3, Ch4))
subset_train <- subset(df_train, select = -c(Task, Ch1, Ch2, Ch3, Ch4, educ, gender, region, segment, ppark, night, miles, Case, No, CC4, GN4, NS4, BU4, FA4, LD4, BZ4, FC4, FP4, RP4, PP4, KA4, SC4, TS4, NV4, MA4, LB4, AF4, HU4, Price4, Urb, income, age))

xgb_params <- list(
  booster = "gbtree",
  eta = 0.07,
  max_depth = 5,
  gamma = 4,
  subsample = 1,
  colsample_bytree = 1,
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = 4
)

# K-FOLD CROSS VALIDATION MODEL TRAINING
set.seed(12)
k <- 5
case_ids <- unique(df_train$Case)
case_assignment <- sample(rep(1:k, length.out = length(case_ids)))
case_folds <- split(seq_along(case_ids), case_assignment)
folds <- lapply(case_folds, function(idx) which(df_train$Case %in% case_ids[idx]))

# TRAIN PREDICTIONS
# Making predictions for the whole of train.csv
oof_preds_xgb <- matrix(NA, nrow = nrow(subset_train), ncol = 4)

for (i in 1:k) {
  train_indices <- unlist(folds[-i])
  val_indices <- folds[[i]]
  
  train_data <- subset_train[train_indices, ]
  val_data <- subset_train[val_indices, ]
  y_train <- as.numeric(train_data$Choice) - 1
  X_train <- train_data %>% select(-Choice)
  X_val <- val_data %>% select(-Choice)
  
  xgb_train <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
  
  fold_model <- xgb.train(
    params = xgb_params,
    data = xgb_train,
    nrounds = 150,
    verbose = 0
  )

# Predict ONLY on this fold's held-out rows, using a model that never saw them
  fold_preds <- predict(fold_model, as.matrix(X_val), reshape = TRUE)
  oof_preds_xgb[val_indices, ] <- fold_preds
    fold_logloss <- compute_logloss(as.data.frame(fold_preds), choice[val_indices, ])
    cat("Fold", i, "Validation Log Loss:", fold_logloss, "\n")
}

# oof_preds_xgb now has one honest out-of-fold prediction per training row
stopifnot(all(!is.na(oof_preds_xgb)))
preds_train_xgb <- as.data.frame(oof_preds_xgb)
colnames(preds_train_xgb) <- c("Ch1","Ch2","Ch3","Ch4")
preds_train_xgb$No <- df_train$No
write.csv(preds_train_xgb, "TrainingTheEnsemble_XGB.csv", row.names = FALSE)
                
overall_oof_logloss <- compute_logloss(preds_train_xgb[, c("Ch1","Ch2","Ch3","Ch4")], choice)
cat("Overall out-of-fold log loss:", overall_oof_logloss, "\n")

# THEN, separately, train your final submission model on ALL of subset_train
final_xgb_model <- xgb.train(
  params = xgb_params,
  data = xgb.DMatrix(data = as.matrix(subset_train %>% select(-Choice)),
                      label = as.numeric(subset_train$Choice) - 1),
  nrounds = 150,
  verbose = 0
)


# SUBMISSION PREDICTIONS 
df_test <- read.csv("test.csv")
subset_test <- subset(df_test, select = -c(Task,Ch1,Ch2,Ch3,Ch4,educ,gender,region,segment,
                                            ppark,night,miles,Case,CC4,GN4,NS4,BU4,FA4,
                                            LD4,BZ4,FC4,FP4,RP4,PP4,KA4,SC4,TS4,NV4,MA4,
                                            LB4,AF4,HU4,Price4,Urb,income,age))
X_test <- subset_test  # No column already excluded above if present; drop it if not
if ("No" %in% colnames(X_test)) X_test <- X_test %>% select(-No)
stopifnot(identical(colnames(subset_train %>% select(-Choice)), colnames(X_test)))

xgb_preds <- predict(final_xgb_model, as.matrix(X_test), reshape = TRUE)
xgb_preds <- as.data.frame(xgb_preds)
colnames(xgb_preds) <- c("Ch1","Ch2","Ch3","Ch4")

submission_prediction_XGB <- subset(df_test, select = c(No))
submission_prediction_XGB <- cbind(submission_prediction_XGB, xgb_preds)

write.csv(submission_prediction_XGB, "InputEnsemble_XGB.csv", row.names = FALSE)
