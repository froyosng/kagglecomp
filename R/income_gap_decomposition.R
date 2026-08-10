library(tidyverse)
source("R/log_loss.R")

oof <- readRDS("data_processed/oof_ensemble_v10.rds")
oof_mlogit <- oof$oof_mlogit
oof_truth <- oof$oof_truth

train <- read.csv("csv files/train.csv")
test <- read.csv("csv files/test.csv")

resp_vars <- c("segmentind","yearind","milesind","milesa","nightind","nighta",
               "pparkind","genderind","ageind","agea","educind",
               "regionind","Urbind","incomeind","incomea")
resp_train <- train %>% distinct(Case, .keep_all = TRUE)
resp_test  <- test  %>% distinct(Case, .keep_all = TRUE)

# --- Rebuild the adversarial classifier (same as before) for importance weights ---
adv_train <- resp_train[, resp_vars]; adv_train$is_test <- 0L
adv_test  <- resp_test[, resp_vars];  adv_test$is_test  <- 1L
adv <- rbind(adv_train, adv_test)
clf <- glm(is_test ~ ., data = adv, family = binomial())
p_test_given_train_resp <- predict(clf, newdata = adv_train, type = "response")
w_importance <- pmin(p_test_given_train_resp / (1 - p_test_given_train_resp), 20)
names(w_importance) <- resp_train$Case

row_weight <- w_importance[as.character(train$Case)]
row_ll <- -log(pmax(oof_mlogit[cbind(seq_len(nrow(train)), max.col(oof_truth))], 1e-15))

ll_unweighted <- mean(row_ll)
ll_weighted <- weighted.mean(row_ll, row_weight)

cat("=== Gap decomposition ===\n")
cat("Unweighted CV log loss (mimics train distribution):", round(ll_unweighted, 6), "\n")
cat("Importance-weighted CV log loss (mimics test's covariate distribution):", round(ll_weighted, 6), "\n")
gap_explained <- ll_weighted - ll_unweighted
total_gap <- 1.202 - 1.145094  # ensemble_v11 CV-to-public gap (logit-only weights used as a proxy; same direction)
cat("Portion of the confirmed shift's effect on CV:", round(gap_explained, 6), "\n")
cat("As a fraction of ensemble_v11's total CV-to-public gap (0.057):",
    round(100 * gap_explained / total_gap, 1), "%\n\n")

# --- Fine-grained per-incomeind-BRACKET breakdown (not tercile) ---
income_bracket <- train$incomeind
bracket_tab <- data.frame(
  bracket = sort(unique(income_bracket)),
  n_train_rows = as.numeric(table(income_bracket)[as.character(sort(unique(income_bracket)))])
)
bracket_ll <- tapply(row_ll, income_bracket, mean)
bracket_tab$mean_oof_logloss <- bracket_ll[as.character(bracket_tab$bracket)]

# test-side representation of each bracket, for comparison
test_bracket_share <- prop.table(table(resp_test$incomeind))
train_bracket_share <- prop.table(table(resp_train$incomeind))
bracket_tab$train_resp_share_pct <- round(100 * train_bracket_share[as.character(bracket_tab$bracket)], 2)
bracket_tab$test_resp_share_pct <- round(100 * test_bracket_share[as.character(bracket_tab$bracket)], 2)
bracket_tab$test_resp_share_pct[is.na(bracket_tab$test_resp_share_pct)] <- 0

cat("=== Per-income-bracket OOF log loss vs train/test representation ===\n")
print(bracket_tab, row.names = FALSE)

cat("\n=== Specifically: the brackets ChatGPT/Claude flagged as over-represented in test ===\n")
for (b in c(14, 21, 22, 24, 26, 27, 28)) {
  if (b %in% bracket_tab$bracket) {
    row <- bracket_tab[bracket_tab$bracket == b, ]
    cat("Bracket", b, ": train_share=", row$train_resp_share_pct, "% test_share=", row$test_resp_share_pct,
        "% OOF logloss=", round(row$mean_oof_logloss, 4), " (overall mean:", round(ll_unweighted, 4), ")\n")
  }
}
