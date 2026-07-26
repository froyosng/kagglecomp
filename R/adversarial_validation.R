library(tidyverse)

train <- read.csv("csv files/train.csv")
test <- read.csv("csv files/test.csv")

resp_vars <- c("segmentind","yearind","milesind","milesa","nightind","nighta",
               "pparkind","genderind","ageind","agea","educind",
               "regionind","Urbind","incomeind","incomea")

resp_train <- train %>% distinct(Case, .keep_all = TRUE) %>% select(all_of(resp_vars))
resp_test  <- test  %>% distinct(Case, .keep_all = TRUE) %>% select(all_of(resp_vars))

resp_train$is_test <- 0L
resp_test$is_test  <- 1L
adv <- bind_rows(resp_train, resp_test)

cat("n train respondents:", nrow(resp_train), " n test respondents:", nrow(resp_test), "\n\n")

# 5-fold CV AUC for a simple logistic classifier distinguishing train vs test
# respondents from covariates alone. AUC ~ 0.5 => no detectable covariate
# shift; AUC notably > 0.5 => the test panel's covariate mix genuinely differs.
set.seed(123)
n <- nrow(adv)
folds <- sample(rep(1:5, length.out = n))
oof_pred <- numeric(n)

auc <- function(pred, label) {
  r <- rank(pred)
  n1 <- sum(label == 1); n0 <- sum(label == 0)
  (sum(r[label == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

for (k in 1:5) {
  fit <- glm(is_test ~ ., data = adv[folds != k, ], family = binomial())
  oof_pred[folds == k] <- predict(fit, newdata = adv[folds == k, ], type = "response")
}
cat("Adversarial validation AUC (logistic, 5-fold CV):", round(auc(oof_pred, adv$is_test), 4), "\n")
cat("(0.5 = indistinguishable / no shift, 1.0 = perfectly separable)\n\n")

full_fit <- glm(is_test ~ ., data = adv, family = binomial())
cat("Full-data logistic fit coefficients (which covariates differ most, if any):\n")
print(round(summary(full_fit)$coefficients[, c("Estimate", "Pr(>|z|)")], 4))

cat("\n--- Univariate train vs test comparison (means) ---\n")
comp <- bind_rows(
  resp_train %>% summarise(across(all_of(resp_vars), ~ mean(.x, na.rm = TRUE))) %>% mutate(set = "train"),
  resp_test  %>% summarise(across(all_of(resp_vars), ~ mean(.x, na.rm = TRUE))) %>% mutate(set = "test")
)
print(as.data.frame(t(comp)))
