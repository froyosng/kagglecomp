
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

choice <- subset(train, select = c(Ch1, Ch2, Ch3, Ch4))
set.seed(12)
k <- 5
case_ids <- unique(train$Case)
case_assignment <- sample(rep(1:k, length.out = length(case_ids)))
case_folds <- split(seq_along(case_ids), case_assignment)
folds <- lapply(case_folds, function(idx) which(train$Case %in% case_ids[idx]))

oof_preds_mnl <- matrix(NA, nrow = nrow(train), ncol = 4)

for (i in 1:k) {
  train_idx <- unlist(folds[-i])
  val_idx   <- folds[[i]]

  fold_train <- train[train_idx, ]
  fold_val   <- train[val_idx, ]

  S_fold_train <- dfidx(fold_train, shape = "wide", choice = "Choice", sep = ".",
                         varying = c(8:291), idx = c("No", "Case"))
  S_fold_val   <- dfidx(fold_val, shape = "wide", choice = "Choice", sep = ".",
                         varying = c(8:291), idx = c("No", "Case"))

  fold_model <- mlogit(Choice~AF1+AF2+BU1+BU2+BU3+BU4+BU5+BZ1+BZ2+BZ3+CC1+CC2+CC3+
                          FA1+FC1+FP1+FP2+FP3+HU1+KA1+KA2+LB3+LD1+LD2+MA1+MA3+NS1+
                          NS4+NV1+NV2+PP1+PP2+RP1+SC1+SC2+SC3+TS1+TS2+Price2+Price3+
                          Price4+Price5+Price6+Price7+Price8+Price9+Price10+Price11+Price12-1,
                        data = S_fold_train)

  fold_preds <- predict(fold_model, newdata = S_fold_val)
  oof_preds_mnl[val_idx, ] <- as.matrix(fold_preds)

  fold_logloss <- compute_logloss(as.data.frame(fold_preds), choice[val_idx, ])
  cat("MNL Fold", i, "Validation Log Loss:", fold_logloss, "\n")
}

stopifnot(all(!is.na(oof_preds_mnl)))
preds_train_MNL <- as.data.frame(oof_preds_mnl)
colnames(preds_train_MNL) <- c("Ch1","Ch2","Ch3","Ch4")
preds_train_MNL$No <- train$No

overall_oof_logloss_mnl <- compute_logloss(preds_train_MNL[, c("Ch1","Ch2","Ch3","Ch4")], choice)
cat("MNL overall out-of-fold log loss:", overall_oof_logloss_mnl, "\n")

write.csv(preds_train_MNL, "TrainingTheEnsemble_MNL.csv", row.names = FALSE)

# THEN fit final MNL on ALL training data for the actual submission
S_test <- dfidx(test, shape="wide", choice="Choice", sep=".",
                varying = c(8:291), idx = c("No", "Case"))

S_everything <- dfidx(train, shape="wide", choice="Choice", sep=".",
                      varying = c(8:291), idx = c("No", "Case"))

MNL_2 <- mlogit(Choice~AF1+AF2+BU1+BU2+BU3+BU4+BU5+BZ1+BZ2+BZ3+CC1+CC2+CC3+
                    FA1+FC1+FP1+FP2+FP3+HU1+KA1+KA2+LB3+LD1+LD2+MA1+MA3+NS1+
                    NS4+NV1+NV2+PP1+PP2+RP1+SC1+SC2+SC3+TS1+TS2+Price2+Price3+
                    Price4+Price5+Price6+Price7+Price8+Price9+Price10+Price11+Price12-1,
                    data=S_everything)


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
