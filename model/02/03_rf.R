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
seed <- 42
set.seed(seed)
case_ids <- unique(safety_full$Case)
val_cases <- sample(case_ids, size = round(0.2 * length(case_ids)))

train_mask <- !(safety_full$Case %in% val_cases)
val_mask   <- safety_full$Case %in% val_cases

trainingSet <- safety[train_mask, ]
testSet     <- safety[val_mask, ]
trainingprobs <- actual_probs[train_mask, ]
testprobs     <- actual_probs[val_mask, ]

# MODEL TRAINING
set.seed(seed)
RF <- randomForest(as.factor(Choice) ~ ., data = trainingSet, mtry=18, importance=TRUE, ntree = 500) 

# MAKE PREDICTIONS
val_pred <- predict(RF, testSet, type="prob")
train_pred <- predict(RF, trainingSet, type="prob")
final_RF <- randomForest(as.factor(Choice) ~ ., data = safety, mtry=18, importance=TRUE, ntree=500)
RF_pred <- predict(final_RF, safety_test, type="prob")

RF_pred <- as.data.frame(RF_pred)
colnames(RF_pred) <- c("Ch1","Ch2","Ch3","Ch4")

                safety_test_raw <- read.csv("test.csv")   # re-read for the No column
submission_prediction_RF <- subset(safety_test_raw, select=c(No))
submission_prediction_RF <- cbind(submission_prediction_RF, RF_pred)

write.csv(submission_prediction_RF, file = "InputEnsemble_RF.csv", row.names = FALSE)

# MODEL EVALUATION
compute_logloss(train_pred,trainingprobs)
compute_accuracy(train_pred,trainingprobs)
compute_logloss(val_pred,testprobs)
compute_accuracy(val_pred,testprobs)

# PREPARE OUTPUT
colnames(val_pred) <- c("Ch1", "Ch2", "Ch3", "Ch4")
colnames(train_pred) <- c("Ch1", "Ch2", "Ch3", "Ch4")
#colnames(RF_pred) <- c("Ch1", "Ch2", "Ch3", "Ch4")
val_pred_df <- as.data.frame(val_pred)
colnames(val_pred_df) <- c("Ch1", "Ch2", "Ch3", "Ch4")
val_pred_df$No <- safety_full$No[val_mask]

write.csv(val_pred_df, file = "TrainingTheEnsemble_RF.csv", row.names = FALSE)







# ==================================================================================
# [4] ENSEMBLE TRAINING & OPTIMISATION
# ==================================================================================
rm(list=ls())
# METRIC FUNCTIONS
compute_logloss <- function(P, choice) {
    # P: predicted choices
    # choice: actual choices
    
    # Calculate the element-wise product of choice and log(P)
    logP <- log(P)
    elementwise_product <- choice * logP
    
    # Sum all the element-wise products
    logloss <- sum(elementwise_product)
    
    # Calculate the average and apply the negative sign
    logloss <- -1 * (logloss / nrow(P))
    
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
    
    accuracy <- (sum(diag(confusion_matrix)))/(sum((confusion_matrix)))
    
    return (accuracy)
}

# DATA EXTRACTION
XGB <- read.csv("TrainingTheEnsemble_XGB.csv")
MNL <- read.csv("TrainingTheEnsemble_MNL.csv")
RF  <- read.csv("TrainingTheEnsemble_RF.csv")

raw_train <- read.csv("train.csv")
raw_train$Choice <- ifelse(raw_train$Ch1==1,1, ifelse(raw_train$Ch2==1,2, ifelse(raw_train$Ch3==1,3,4)))

# Rename each model's Ch columns before merging so they don't collide
colnames(XGB) <- c("Ch1_xgb","Ch2_xgb","Ch3_xgb","Ch4_xgb","No")
colnames(MNL) <- c("Ch1_mnl","Ch2_mnl","Ch3_mnl","Ch4_mnl","No")
colnames(RF)  <- c("Ch1_rf","Ch2_rf","Ch3_rf","Ch4_rf","No")

true_labels <- subset(raw_train, select = c(No, Ch1, Ch2, Ch3, Ch4))

# Inner join keeps only rows where ALL THREE models have honest out-of-fold predictions
ensemble_data <- Reduce(function(x, y) merge(x, y, by = "No"),
                         list(XGB, MNL, RF, true_labels))

cat("Rows available for ensemble (all 3 models overlap):", nrow(ensemble_data), "\n")
                           
set.seed(1234)
ensemble_cases <- unique(subset(raw_train, No %in% ensemble_data$No)$Case)
ensemble_data <- merge(ensemble_data, subset(raw_train, select = c(No, Case)), by = "No")

val_cases_ensemble <- sample(unique(ensemble_data$Case), size = round(0.2 * length(unique(ensemble_data$Case))))

train_ensemble <- subset(ensemble_data, !(Case %in% val_cases_ensemble))
test_ensemble  <- subset(ensemble_data, Case %in% val_cases_ensemble)

train_ensemble_choice <- subset(train_ensemble, select = c(Ch1, Ch2, Ch3, Ch4))
test_ensemble_choice  <- subset(test_ensemble, select = c(Ch1, Ch2, Ch3, Ch4))




# SOFT VOTING
soft_voting <- function(data, weights = c(0.5, 0.3, 0.2)){
    predictions <- data.frame(
      Ch1 = data$Ch1_xgb*weights[1] + data$Ch1_mnl*weights[2] + data$Ch1_rf*weights[3],
      Ch2 = data$Ch2_xgb*weights[1] + data$Ch2_mnl*weights[2] + data$Ch2_rf*weights[3],
      Ch3 = data$Ch3_xgb*weights[1] + data$Ch3_mnl*weights[2] + data$Ch3_rf*weights[3],
      Ch4 = data$Ch4_xgb*weights[1] + data$Ch4_mnl*weights[2] + data$Ch4_rf*weights[3]
    )
    row_sums <- rowSums(predictions)
    predictions <- predictions / row_sums
    return(predictions)
}

# SAMPLE IMPLEMENTATION
predictions <- soft_voting(train_ensemble, weights = c(0.5, 0.3, 0.2))
logloss <- compute_logloss(predictions, train_ensemble_choice)
logloss
accuracy <- compute_accuracy(predictions, train_ensemble_choice)
accuracy

df_results <- data.frame(weight1 = numeric(), 
                         weight2 = numeric(),
                         weight3 = numeric(),
                         train_logloss = numeric(), test_logloss = numeric(),
                         train_accuracy = numeric(), test_accuracy = numeric())

# SIMULATION
iteration <- 0
for (weight1 in seq(0, 0.99, by=0.01)){
    for (weight2 in seq(0, 1-weight1, by=0.01)){
        weight3 <- 1 - weight1 - weight2

        train_predictions <- soft_voting(train_ensemble, weights = c(weight1, weight2, weight3))
        train_logloss <- compute_logloss(train_predictions, train_ensemble_choice)
        train_accuracy <- compute_accuracy(train_predictions, train_ensemble_choice)

        test_predictions <- soft_voting(test_ensemble, weights = c(weight1, weight2, weight3))
        test_logloss <- compute_logloss(test_predictions, test_ensemble_choice)
        test_accuracy <- compute_accuracy(test_predictions, test_ensemble_choice)
        
        df_results <- rbind(df_results, data.frame(train_logloss = train_logloss, test_logloss = test_logloss,
                                                   train_accuracy = train_accuracy, test_accuracy = test_accuracy, 
                                                   weight1 = weight1, weight2 = weight2, weight3 = weight3))
        
        iteration <- iteration + 1
        cat("Iteration ", iteration, "\n")
    }
}

plot(df_results$train_logloss, df_results$test_logloss)
plot(df_results$train_logloss, df_results$train_accuracy)
plot(df_results$test_logloss, df_results$test_accuracy)
plot(df_results$train_accuracy, df_results$train_logloss)
plot(df_results$weight1, df_results$train_logloss)
plot(df_results$weight2, df_results$train_logloss)
plot(df_results$weight3, df_results$train_logloss)


# Pick weights that minimise TEST log loss objectively, rather than eyeballing plots
best_row <- df_results[which.min(df_results$test_logloss), ]
weight_XGB <- best_row$weight1
weight_MNL <- best_row$weight2
weight_RF  <- best_row$weight3
cat("Best weights - XGB:", weight_XGB, " MNL:", weight_MNL, " RF:", weight_RF, "\n")

train_predictions <- soft_voting(train_ensemble, weights = c(weight_XGB, weight_MNL, weight_RF))
test_predictions  <- soft_voting(test_ensemble,  weights = c(weight_XGB, weight_MNL, weight_RF))

train_logloss <- compute_logloss(train_predictions, train_ensemble_choice)
train_accuracy <- compute_accuracy(train_predictions, train_ensemble_choice)
test_logloss <- compute_logloss(test_predictions, test_ensemble_choice)
test_accuracy <- compute_accuracy(test_predictions, test_ensemble_choice)
train_logloss
test_logloss
train_accuracy
test_accuracy


# ==============================================================================
# MAKING SUBMISSION PREDICTION
# ==============================================================================
XGB_probabilities <- read.csv("InputEnsemble_XGB.csv") 
MNL_probabilities <- read.csv("InputEnsemble_MNL.csv") 
RF_probabilities <- read.csv("InputEnsemble_RF.csv") 

test <- read.csv("test.csv")
submission_13 <- subset(test, select=c(No))
soft_voting_submission <- function(pred1, pred2, pred3, weights){
    predictions <- data.frame(
      Ch1 = pred1$Ch1*weights[1] + pred2$Ch1*weights[2] + pred3$Ch1*weights[3],
      Ch2 = pred1$Ch2*weights[1] + pred2$Ch2*weights[2] + pred3$Ch2*weights[3],
      Ch3 = pred1$Ch3*weights[1] + pred2$Ch3*weights[2] + pred3$Ch3*weights[3],
      Ch4 = pred1$Ch4*weights[1] + pred2$Ch4*weights[2] + pred3$Ch4*weights[3]
    )
    predictions / rowSums(predictions)
}
                
submission_probabilities <- soft_voting_submission(XGB_probabilities, MNL_probabilities, RF_probabilities, 
                             weights=c(weight_XGB, weight_MNL, weight_RF))
submission_13$Ch1 <- submission_probabilities$Ch1
submission_13$Ch2 <- submission_probabilities$Ch2
submission_13$Ch3 <- submission_probabilities$Ch3
submission_13$Ch4 <- submission_probabilities$Ch4

write.csv(submission_13, "0802_submission13_ENSEMBLE_1.csv", row.names = FALSE)
