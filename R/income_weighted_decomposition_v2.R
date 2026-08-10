library(tidyverse)
source("R/log_loss.R")

oof <- readRDS("data_processed/oof_ensemble_v10.rds")
oof_mlogit <- oof$oof_mlogit
oof_truth <- oof$oof_truth

train <- read.csv("csv files/train.csv")
test <- read.csv("csv files/test.csv")
resp_train <- train %>% distinct(Case, .keep_all = TRUE)
resp_test  <- test  %>% distinct(Case, .keep_all = TRUE)

# Respondent-level mean OOF log loss across each respondent's 19 tasks
row_ll <- -log(pmax(oof_mlogit[cbind(seq_len(nrow(train)), max.col(oof_truth))], 1e-15))
resp_ll <- tapply(row_ll, train$Case, mean)
resp_ll <- resp_ll[as.character(resp_train$Case)]

# Clean univariate density-ratio weight: P_test(bracket)/P_train(bracket),
# computed at the RESPONDENT level (not conflated with other covariates the
# way the earlier multivariate adversarial-classifier weight was).
train_share <- prop.table(table(resp_train$incomeind))
test_share <- prop.table(table(resp_test$incomeind))
all_brackets <- union(names(train_share), names(test_share))
test_share_full <- setNames(rep(0, length(all_brackets)), all_brackets)
test_share_full[names(test_share)] <- test_share
w <- as.numeric(test_share_full[as.character(resp_train$incomeind)]) /
     as.numeric(train_share[as.character(resp_train$incomeind)])

L_ordinary <- mean(resp_ll)
L_weighted <- sum(w * resp_ll) / sum(w)
delta <- L_weighted - L_ordinary
n_eff <- sum(w)^2 / sum(w^2)

cat("=== Respondent-level income-bracket density-ratio decomposition ===\n")
cat("L_ordinary (unweighted):", round(L_ordinary, 6), "\n")
cat("L_weighted (test income-bracket distribution):", round(L_weighted, 6), "\n")
cat("Delta (income-shift contribution):", round(delta, 6), "\n")
cat("Effective sample size of the weighting: n_eff =", round(n_eff, 1), "of", length(w), "respondents\n")
cat("(low n_eff means the weighted estimate is dominated by a handful of respondents)\n\n")

# Bootstrap respondents for CIs on L_weighted and delta
set.seed(2027)
B <- 1000
n <- length(resp_ll)
boot_Lw <- numeric(B); boot_delta <- numeric(B)
for (b in 1:B) {
  idx <- sample(seq_len(n), n, replace = TRUE)
  Lo_b <- mean(resp_ll[idx])
  Lw_b <- sum(w[idx] * resp_ll[idx]) / sum(w[idx])
  boot_Lw[b] <- Lw_b
  boot_delta[b] <- Lw_b - Lo_b
}
cat("=== Bootstrap CIs (1000 resamples of respondents) ===\n")
cat("L_weighted: mean =", round(mean(boot_Lw), 6), " 95% CI [", round(quantile(boot_Lw, 0.025), 6), ",", round(quantile(boot_Lw, 0.975), 6), "]\n")
cat("Delta: mean =", round(mean(boot_delta), 6), " 95% CI [", round(quantile(boot_delta, 0.025), 6), ",", round(quantile(boot_delta, 0.975), 6), "]\n\n")

# Per-bracket bootstrap CI for the two flagged brackets specifically
cat("=== Per-bracket bootstrap CI (respondent-level, flagged brackets) ===\n")
for (b in c(14, 28)) {
  in_bracket <- resp_train$incomeind == b
  n_b <- sum(in_bracket)
  vals <- resp_ll[in_bracket]
  boot_means <- replicate(1000, mean(sample(vals, n_b, replace = TRUE)))
  cat("Bracket", b, "(n =", n_b, "respondents): mean OOF loss =", round(mean(vals), 4),
      " 95% CI [", round(quantile(boot_means, 0.025), 4), ",", round(quantile(boot_means, 0.975), 4), "]\n")
}
cat("Overall mean (all respondents):", round(L_ordinary, 4), "\n")
