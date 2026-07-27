# ==================================================================================
# [3] RANDOM FOREST
# ==================================================================================
library(randomForest)
library(caret)
library(ggplot2)
library(tidyverse)

safety <- read.csv("train.csv")
safety$Choice <- ifelse(safety$Ch1 == 1, 1, 
                        ifelse(safety$Ch2 == 1, 2, 
                               ifelse(safety$Ch3 == 1, 3, 
                                      ifelse(safety$Ch4 == 1, 4, NA))))
safety_full <- safety
                
safety_test <- read.csv("test.csv")
safety_test$Choice <- 0
safety_test$Ch1 <- 0
safety_test$Ch2 <- 0
safety_test$Ch3 <- 0
safety_test$Ch4 <- 0

actual_probs <- safety[, c("Ch1","Ch2","Ch3","Ch4")]

# remove unnecessary columns
safety <- subset(safety, select=-c(No,Case,CC4,GN4,NS4,BU4,FA4,LD4,BZ4,FC4,FP4,RP4,
                                   PP4,KA4,SC4,TS4,NV4,MA4,LB4,AF4,HU4,Price4,Task,
                                   segment,year,miles,night,ppark,gender,age,educ,
                                   region,Urb,income,agea,nighta,incomea,milesa,Ch1,Ch2,Ch3,Ch4))

safety_test <- subset(safety_test, select=-c(No,Case,CC4,GN4,NS4,BU4,FA4,LD4,BZ4,FC4,FP4,RP4,
                                             PP4,KA4,SC4,TS4,NV4,MA4,LB4,AF4,HU4,Price4,Task,
                                             segment,year,miles,night,ppark,gender,age,educ,
                                             region,Urb,income,agea,nighta,incomea,milesa,Ch1,Ch2,Ch3,Ch4))

# DATA SPLITTING
set.seed(42)
k <- 5
case_ids <- unique(safety_full$Case)
case_assignment <- sample(rep(1:k, length.out = length(case_ids)))
case_folds <- split(seq_along(case_ids), case_assignment)
folds <- lapply(case_folds, function(idx) which(safety_full$Case %in% case_ids[idx]))

oof_preds_rf <- matrix(NA, nrow = nrow(safety), ncol = 4)
for (i in 1:k) {
  train_idx <- unlist(folds[-i])
  val_idx   <- folds[[i]]

  fold_train <- safety[train_idx, ]
  fold_val   <- safety[val_idx, ]

  fold_model <- randomForest(as.factor(Choice) ~ ., data = fold_train,
                              mtry = 18, importance = FALSE, ntree = 500)

  fold_preds <- predict(fold_model, fold_val, type = "prob")
  oof_preds_rf[val_idx, ] <- fold_preds

  fold_logloss <- compute_logloss(as.data.frame(fold_preds), actual_probs[val_idx, ])
  cat("RF Fold", i, "Validation Log Loss:", fold_logloss, "\n")
}

stopifnot(all(!is.na(oof_preds_rf)))
preds_train_RF <- as.data.frame(oof_preds_rf)
colnames(preds_train_RF) <- c("Ch1","Ch2","Ch3","Ch4")
preds_train_RF$No <- safety_full$No

overall_oof_logloss_rf <- compute_logloss(preds_train_RF[, c("Ch1","Ch2","Ch3","Ch4")], actual_probs)
cat("RF overall out-of-fold log loss:", overall_oof_logloss_rf, "\n")

write.csv(preds_train_RF, "TrainingTheEnsemble_RF.csv", row.names = FALSE)

# Final RF trained on ALL data for submission (unchanged)
final_RF <- randomForest(as.factor(Choice) ~ ., data = safety, mtry=18, importance=TRUE, ntree=500)
RF_pred <- predict(final_RF, safety_test, type="prob")
RF_pred <- as.data.frame(RF_pred)
colnames(RF_pred) <- c("Ch1","Ch2","Ch3","Ch4")

safety_test_raw <- read.csv("test.csv")   # re-read for the No column
submission_prediction_RF <- subset(safety_test_raw, select=c(No))
submission_prediction_RF <- cbind(submission_prediction_RF, RF_pred)
write.csv(submission_prediction_RF, file = "InputEnsemble_RF.csv", row.names = FALSE)
