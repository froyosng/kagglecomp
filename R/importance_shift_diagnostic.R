library(tidyverse)
source("R/log_loss.R")

oof <- readRDS("data_processed/oof_ensemble_v10.rds")
oof_mlogit <- oof$oof_mlogit
oof_truth <- oof$oof_truth
fold_of_case <- oof$fold_of_case

train <- read.csv("csv files/train.csv")
test <- read.csv("csv files/test.csv")

resp_vars <- c("segmentind","yearind","milesind","milesa","nightind","nighta",
               "pparkind","genderind","ageind","agea","educind",
               "regionind","Urbind","incomeind","incomea")

resp_train <- train %>% dplyr::distinct(Case, .keep_all = TRUE)
resp_test  <- test  %>% dplyr::distinct(Case, .keep_all = TRUE)

adv_train <- resp_train[, resp_vars]; adv_train$is_test <- 0L
adv_test  <- resp_test[, resp_vars];  adv_test$is_test  <- 1L
adv <- rbind(adv_train, adv_test)

# Fit the adversarial classifier on ALL data (this is just for diagnosing shift
# and constructing weights, not for the choice model itself, so no leakage
# concern the way it would be for the actual predictive task)
clf <- glm(is_test ~ ., data = adv, family = binomial())
p_test_given_train_resp <- predict(clf, newdata = adv_train, type = "response")
w_importance <- pmin(p_test_given_train_resp / (1 - p_test_given_train_resp), 20)  # capped, avoid extreme weights
names(w_importance) <- resp_train$Case

cat("Importance weight summary (capped at 20):\n")
print(summary(w_importance))

# Diagnostic (no refitting needed): does the EXISTING model's OOF performance
# get worse when we upweight test-like (high income, etc.) training
# respondents in the evaluation? This tells us whether the current model is
# specifically weaker on the sub-population that resembles test, without
# needing to retrain anything.
row_weight <- w_importance[as.character(train$Case)]
row_ll <- -log(pmax(oof_mlogit[cbind(seq_len(nrow(train)), max.col(oof_truth))], 1e-15))

cat("\nUnweighted mean OOF log loss (mlogit):", round(mean(row_ll), 6), "\n")
cat("Importance-weighted mean OOF log loss (mlogit, weighted toward test-like respondents):",
    round(weighted.mean(row_ll, row_weight), 6), "\n")
cat("(if this is notably higher than the unweighted number, the model really is\n")
cat(" weaker specifically on the kind of respondent test skews toward)\n\n")

# Split by income tercile as a simple, interpretable version of the same check
income_tercile <- cut(train$incomea, quantile(train$incomea, c(0, 1/3, 2/3, 1), na.rm = TRUE),
                       include.lowest = TRUE, labels = c("low", "mid", "high"))
cat("Mean OOF log loss by training income tercile:\n")
print(round(tapply(row_ll, income_tercile, mean), 6))
cat("\n(test respondents skew toward the 'high' tercile -- if that tercile's log loss\n")
cat(" is already higher even within TRAINING, that's a real, actionable weak spot)\n")
