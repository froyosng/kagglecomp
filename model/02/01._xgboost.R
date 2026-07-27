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


# ==================================================================================
# [2] MULTINOMIAL LOGIT MODEL WITH ONE-HOT ENCODING
# ==================================================================================
library(mlogit)
library(caret)

# DATA EXTRACTION 
train_raw <- read.csv("train.csv")
test_raw <- read.csv("test.csv")
train_raw$Choice <- ifelse(train_raw$Ch1 == 1, 1, ifelse(train_raw$Ch2 == 1, 2, ifelse(train_raw$Ch3 == 1, 3, 4)))
test_raw$Choice <- sample(c(1, 2, 3, 4), nrow(test_raw), replace = TRUE, prob = c(0.25, 0.25, 0.25, 0.25))

id_train <- subset(train_raw, select = c(Case, Task, No, Ch1, Ch2, Ch3, Ch4))
id_test <- subset(test_raw, select = c(Case, Task, No, Ch1, Ch2, Ch3, Ch4))

df_train <- subset(train_raw, select = -c(segment,segmentind,year,yearind,milesa,miles,milesind,nighta,night,
                                          nightind,ppark,pparkind,gender,genderind,age,ageind,agea,educ,
                                          educind,region,regionind,Urb,Urbind,income,incomeind,incomea,
                                          Ch1,Ch2,Ch3,Ch4))
df_test <- subset(test_raw, select = -c(segment,segmentind,year,yearind,milesa,miles,milesind,nighta,night,
                                        nightind,ppark,pparkind,gender,genderind,age,ageind,agea,educ,
                                        educind,region,regionind,Urb,Urbind,income,incomeind,incomea,
                                        Ch1,Ch2,Ch3,Ch4))

# ----------------------------------------------------------------------------------
# FEATURE ENGINEERING (ONE HOT ENCODING)
# PART 1. ONE HOT ENCODING SAFETY PACKAGE FEATURES

# 1. Converting the car package features into factors / categorical variables
car_features <- c('CC1','GN1','NS1','BU1','FA1','LD1','BZ1','FC1','FP1','RP1','PP1','KA1','SC1','TS1','NV1','MA1','LB1','AF1','HU1','Price1',
                  'CC2','GN2','NS2','BU2','FA2','LD2','BZ2','FC2','FP2','RP2','PP2','KA2','SC2','TS2','NV2','MA2','LB2','AF2','HU2','Price2',
                  'CC3','GN3','NS3','BU3','FA3','LD3','BZ3','FC3','FP3','RP3','PP3','KA3','SC3','TS3','NV3','MA3','LB3','AF3','HU3','Price3',
                  'CC4','GN4','NS4','BU4','FA4','LD4','BZ4','FC4','FP4','RP4','PP4','KA4','SC4','TS4','NV4','MA4','LB4','AF4','HU4','Price4')
for (car_feature in car_features){
    df_train[, car_feature] <- factor(df_train[, car_feature])
    df_test[, car_feature] <- factor(df_test[, car_feature])
}

# # 2. Check and remove factors with only one level
for (car_feature in car_features) {
    if (length(unique(df_train[, car_feature])) <= 1) {
        df_train[, car_feature] <- NULL
        df_test[, car_feature] <- NULL
    }
}

# 3. Create dummy variables
dummy <- dummyVars(" ~ .", data = df_train)
train_data <- data.frame(predict(dummy, newdata = df_train))
test_data <- data.frame(predict(dummy, newdata = df_test))

# 4. Swap column names from FeatureAlt.Level to FeatureLevel.Alt
swap_xy <- function(col_name) {
    sub("([A-Za-z]+)([0-9]+)\\.([0-9]+)", "\\1\\3.\\2", col_name)
}

train_data_tochange <- subset(train_data,select = -c(Choice,Case,No,Task))
colnames(train_data_tochange) <- sapply(colnames(train_data_tochange), swap_xy)

test_data_tochange <- subset(test_data,select = -c(Choice,Case,No,Task))
colnames(test_data_tochange) <- sapply(colnames(test_data_tochange), swap_xy)

# 5. Add the columns for the 4th alternative.
# Step 1: Extract the PrefixY part from current column names
prefix_y <- sub("([A-Za-z]+[0-9]+)\\.[0-9]+", "\\1", colnames(train_data_tochange)[1:91])

# Step 2: Create new column names by appending ".4"
new_colnames <- paste0(prefix_y, ".4")

# Step 3: Add new columns with these names to the data frame
# Initialize the new columns with NA or any other value as required
for (new_col in new_colnames) {
    train_data_tochange[[new_col]] <- 0
    test_data_tochange[[new_col]] <- 0
}

# 6. Remove 1 indicator variable for each category to reduce model variability
train_features <- subset(train_data_tochange, select = -c(AF0.1, AF0.2, AF0.3, AF0.4,
                                                          BU0.1, BU0.2, BU0.3, BU0.4,
                                                          BZ0.1, BZ0.2, BZ0.3, BZ0.4,
                                                          CC0.1, CC0.2, CC0.3, CC0.4, 
                                                          FA0.1, FA0.2, FA0.3, FA0.4,
                                                          FC0.1, FC0.2, FC0.3, FC0.4,
                                                          FP0.1, FP0.2, FP0.3, FP0.4,
                                                          GN0.1, GN0.2, GN0.3, GN0.4,
                                                          HU0.1, HU0.2, HU0.3, HU0.4,
                                                          KA0.1, KA0.2, KA0.3, KA0.4,
                                                          LB0.1, LB0.2, LB0.3, LB0.4,
                                                          LD0.1, LD0.2, LD0.3, LD0.4,
                                                          MA0.1, MA0.2, MA0.3, MA0.4,
                                                          NS0.1, NS0.2, NS0.3, NS0.4,
                                                          NV0.1, NV0.2, NV0.3, NV0.4,
                                                          PP0.1, PP0.2, PP0.3, PP0.4,
                                                          RP0.1, RP0.2, RP0.3, RP0.4,
                                                          SC0.1, SC0.2, SC0.3, SC0.4,
                                                          TS0.1, TS0.2, TS0.3, TS0.4,
                                                          Price1.1, Price1.2, Price1.3, Price1.4))

test_features <- subset(test_data_tochange, select = -c(AF0.1, AF0.2, AF0.3, AF0.4,
                                                        BU0.1, BU0.2, BU0.3, BU0.4,
                                                        BZ0.1, BZ0.2, BZ0.3, BZ0.4,
                                                        CC0.1, CC0.2, CC0.3, CC0.4, 
                                                        FA0.1, FA0.2, FA0.3, FA0.4,
                                                        FC0.1, FC0.2, FC0.3, FC0.4,
                                                        FP0.1, FP0.2, FP0.3, FP0.4,
                                                        GN0.1, GN0.2, GN0.3, GN0.4,
                                                        HU0.1, HU0.2, HU0.3, HU0.4,
                                                        KA0.1, KA0.2, KA0.3, KA0.4,
                                                        LB0.1, LB0.2, LB0.3, LB0.4,
                                                        LD0.1, LD0.2, LD0.3, LD0.4,
                                                        MA0.1, MA0.2, MA0.3, MA0.4,
                                                        NS0.1, NS0.2, NS0.3, NS0.4,
                                                        NV0.1, NV0.2, NV0.3, NV0.4,
                                                        PP0.1, PP0.2, PP0.3, PP0.4,
                                                        RP0.1, RP0.2, RP0.3, RP0.4,
                                                        SC0.1, SC0.2, SC0.3, SC0.4,
                                                        TS0.1, TS0.2, TS0.3, TS0.4,
                                                        Price1.1, Price1.2, Price1.3, Price1.4))
# 7. Add Case, No, Task, Choice columns back 
train_features$No <- seq(1, nrow(train_features))
train <- merge(id_train, train_features, on='No')
train["Choice"] <- train_data["Choice"]

test_features$No <- seq(21566, 21566+nrow(test_features)-1)
test <- merge(id_test, test_features, on='No')
test["Choice"] <- test_data["Choice"]

# 8. Clearing the environment 
rm(train_data, train_data_tochange, train_features,
   test_data, test_data_tochange, test_features,
   dummy)

# ------------------------------------------------------------------------
# Part 2 (1 HOT ENCODE HUMAN FEATURES)
# 1. Select all human features to encode

train_human_to_encode <- subset(train_raw, select = c(No,segmentind, yearind, milesind, nightind,
                                                      pparkind, genderind, ageind, educind,
                                                      regionind, Urbind, incomeind))

test_human_to_encode <- subset(test_raw, select = c(No,segmentind, yearind, milesind, nightind,
                                                    pparkind, genderind, ageind, educind,
                                                    regionind, Urbind, incomeind))

# 2. Convert all human features to factors
human_features <- c('segmentind', 'yearind', 'milesind', 'nightind',
                    'pparkind', 'genderind', 'ageind', 'educind',
                    'regionind', 'Urbind', 'incomeind')

for (human_feature in human_features){
    train_human_to_encode[, human_feature] <- factor(train_human_to_encode[, human_feature])
    test_human_to_encode[, human_feature] <- factor(test_human_to_encode[, human_feature])
}

# 3. Check and remove factors with only one level
for (human_feature in human_features) {
    if (length(unique(train_human_to_encode[, human_feature])) <= 1) {
        train_human_to_encode[, human_feature] <- NULL
        test_human_to_encode[, human_feature] <- NULL
    }
}                                

# 4. Create dummy variables
dummy <- dummyVars(" ~ .", data = train_human_to_encode)
train_human <- data.frame(predict(dummy, newdata = train_human_to_encode))
test_human <- data.frame(predict(dummy, newdata = test_human_to_encode))

# 5. Remove 1 indicator variable for each category to reduce model variability
train_human <- subset(train_human, select = -c(segmentind.1, yearind.1, milesind.1, nightind.1,
                                               pparkind.1, genderind.1, ageind.1, educind.1,
                                               regionind.1, Urbind.1, incomeind.1))

test_human <- subset(test_human, select = -c(segmentind.1, yearind.1, milesind.1, nightind.1,
                                             pparkind.1, genderind.1, ageind.1, educind.1,
                                             regionind.1, Urbind.1, incomeind.1))
# 6. Merge with Part 1.

train <- merge(train, train_human, on='No')
test <- merge(test, test_human, on='No')

# ------------------------------------------------------------------------
# Merge data with incomea, nighta, agea
human_train <- subset(train_raw, select = c(agea,nighta,incomea,No))
human_test <- subset(test_raw,select = c(agea,nighta,incomea,No))

train <- merge(train, human_train, on='No')
test <- merge(test, human_test, on = 'No')

rm(test_human,test_human_to_encode,train_human,train_human_to_encode,dummy,human_test,human_train)

# ------------------------------------------------------------------------
# PART 3 MODEL BUILDING
# Formatting data for mlogit()

set.seed(12)
val_cases_mnl <- sample(unique(train_raw$Case), size = round(0.2 * length(unique(train_raw$Case))))

S_train <- dfidx(subset(train, !(Case %in% val_cases_mnl)), shape="wide", choice="Choice", sep=".",
                 varying = c(8:291), idx = c("No", "Case"))
S_val <- dfidx(subset(train, Case %in% val_cases_mnl), shape="wide", choice="Choice", sep=".",
               varying = c(8:291), idx = c("No", "Case"))
                


S_test <- dfidx(test, shape="wide", choice="Choice", sep=".",
                varying = c(8:291), idx = c("No", "Case"))

S_everything <- dfidx(train, shape="wide", choice="Choice", sep=".", 
                      varying = c(8:291), idx = c("No", "Case"))

MNL_2 <- mlogit(Choice~AF1+AF2+BU1+BU2+BU3+BU4+BU5+BZ1+BZ2+BZ3+CC1+CC2+CC3+
                    FA1+FC1+FP1+FP2+FP3+HU1+KA1+KA2+LB3+LD1+LD2+MA1+MA3+NS1+
                    NS4+NV1+NV2+PP1+PP2+RP1+SC1+SC2+SC3+TS1+TS2+Price2+Price3+
                    Price4+Price5+Price6+Price7+Price8+Price9+Price10+Price11+Price12-1,
                    data=S_train)

# Making Predictions to train the Ensemble
preds_val_MNL <- predict(MNL_2, newdata = S_val)
preds_val_MNL <- as.data.frame(preds_val_MNL)
colnames(preds_val_MNL) <- c("Ch1","Ch2","Ch3","Ch4")
preds_val_MNL$No <- subset(train_raw, Case %in% val_cases_mnl)$No

write.csv(preds_val_MNL, "TrainingTheEnsemble_MNL.csv", row.names = FALSE)

# Making Predictions for the submission
submission_prediction_MNL <- subset(test_raw, select=c(No))
MNL_preds <- predict(MNL_2, newdata = S_test)
MNL_preds <- as.data.frame(MNL_preds)
colnames(MNL_preds) <- c("Ch1","Ch2","Ch3","Ch4")
submission_prediction_MNL$Ch1 <- MNL_preds$Ch1
submission_prediction_MNL$Ch2 <- MNL_preds$Ch2
submission_prediction_MNL$Ch3 <- MNL_preds$Ch3
submission_prediction_MNL$Ch4 <- MNL_preds$Ch4

write.csv(submission_prediction_MNL, "InputEnsemble_MNL.csv", row.names = FALSE)



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
