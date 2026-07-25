library(xgboost)
library(caTools)
library(dplyr)
library(caret)

safety <- read.csv("train.csv")

# DATA SPLITTING
safety$Choice <- ifelse(safety$Ch1==1, 1, 
                 ifelse(safety$Ch2==1, 2, 
                 ifelse(safety$Ch3==1, 3, 4)))
train_raw <- subset(safety, Task <= 12)
test_raw <- subset(safety, Task > 12)
test_choices <- subset(test_raw, select = c(Ch1,Ch2,Ch3,Ch4)) 

train <- subset(train_raw, select=-c(Task,Ch1,Ch2,Ch3,Ch4,educ,gender,region,segment,
                                     ppark,night,miles,Case,No,CC4,GN4,NS4,BU4,FA4,
                                     LD4,BZ4,FC4,FP4,RP4,PP4,KA4,SC4,TS4,NV4,MA4,
                                     LB4,AF4,HU4,Price4,Urb,income,age))
test <- subset(test_raw, select=-c(Task,Ch1,Ch2,Ch3,Ch4,educ,gender,region,segment,
                                     ppark,night,miles,Case,No,CC4,GN4,NS4,BU4,FA4,
                                     LD4,BZ4,FC4,FP4,RP4,PP4,KA4,SC4,TS4,NV4,MA4,
                                     LB4,AF4,HU4,Price4,Urb,income,age))
y_train <- as.numeric(train$Choice) -1
y_test <- as.integer(test$Choice) -1
X_train <- train %>% select(-Choice)
X_test <- test %>% select(-Choice)


# MODEL TRAINING
xgb_train <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
xgb_test <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)
xgb_params <- list(
    booster = "gbtree",
    eta = 0.01,
    max_depth = 8,
    gamma = 4,
    subsample = 0.75,
    colsample_bytree = 1,
    objective = "multi:softprob",
    eval_metric = "mlogloss",
    num_class = 4
)

watchlist <- list(train = xgb_train, test = xgb_test)
xgb_model <- xgb.train(
    params = xgb_params,
    data = xgb_train,
    nrounds = 5000,
    watchlist = watchlist, 
    #uncomment watchlist if you dont need metric printings
    verbose = 1
)
xgb_model

# PRINT METRICS
eval_log <- xgb_model$evaluation_log
print(eval_log)
last_eval_score <- eval_log[nrow(eval_log), ]
print(last_eval_score)
final_train_logloss <- last_eval_score$train_mlogloss
final_test_logloss <- last_eval_score$test_mlogloss
print(paste("Final Training Log Loss: ", final_train_logloss))
print(paste("Final Testing Log Loss: ", final_test_logloss))

# MODEL TESTING
xgb_preds <- predict(xgb_model, as.matrix(X_test), reshape = TRUE)
xgb_preds <- as.data.frame(xgb_preds)
colnames(xgb_preds) <- c("1","2","3","4")

xgb_preds$PredictedClass <- apply(xgb_preds, 1, function(y) colnames(xgb_preds)[which.max(y)])
xgb_preds$ActualClass <- test_raw[,"Choice"]
xgb_preds

logloss <- 0
for (i in 1:nrow(test_choices)) {
    logloss <- logloss + test_choices$Ch1[i]*log(xgb_preds$"1"[i]) 
    logloss <- logloss + test_choices$Ch2[i]*log(xgb_preds$"2"[i]) 
    logloss <- logloss + test_choices$Ch3[i]*log(xgb_preds$"3"[i]) 
    logloss <- logloss + test_choices$Ch4[i]*log(xgb_preds$"4"[i])
}
logloss <- logloss / nrow(test_choices)
logloss <- -1 * logloss
logloss

compute_logloss <- function(test_set, testpredict_df) {
    # Create one-hot encoding for each choice on-the-fly
    Ch1 <- as.integer(test_set$Choice == 0)
    Ch2 <- as.integer(test_set$Choice == 1)
    Ch3 <- as.integer(test_set$Choice == 2)
    Ch4 <- as.integer(test_set$Choice == 3)
    
    # Calculate logloss using these one-hot encoded variables
    result <- -1/nrow(test_set) * sum(Ch1 * log(testpredict_df$Ch1+.Machine$double.eps) +
                                          Ch2 * log(testpredict_df$Ch2+.Machine$double.eps) +
                                          Ch3 * log(testpredict_df$Ch3+.Machine$double.eps) +
                                          Ch4 * log(testpredict_df$Ch4+.Machine$double.eps))
    return(result)
}


compute_logloss(y_test, )








# PREDICTING AND TESTING FOR SUBMISSION
df_test <- read.csv("test.csv")
df_test$Choice <- sample(c(1, 2, 3, 4), nrow(df_test), replace = TRUE, prob = c(0.25, 0.25, 0.25, 0.25))
subset_test <- subset(df_test,select=-c(Task,Ch1,Ch2,Ch3,Ch4,educ,gender,region,segment,ppark,night,miles,Case,CC4,GN4,NS4,BU4,FA4,LD4,BZ4,FC4,FP4,RP4,PP4,KA4,SC4,TS4,NV4,MA4,LB4,AF4,HU4,Price4,Urb,income,age))
X_test <- subset_test %>% select(-c(Choice,No))
xgb_preds <- predict(xgb_model, as.matrix(X_test), reshape = TRUE)
xgb_preds <- as.data.frame(xgb_preds)
colnames(xgb_preds) <- c("1","2","3","4")

xgb_preds
