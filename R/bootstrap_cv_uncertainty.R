source("R/log_loss.R")

oof <- readRDS("data_processed/oof_ensemble_v10.rds")
oof_mlogit <- oof$oof_mlogit   # m8trpg alone
oof_xgb <- oof$oof_xgb
oof_truth <- oof$oof_truth

train <- read.csv("csv files/train.csv")
cases <- unique(train$Case)

w <- 0.80
oof_ensemble <- w * oof_mlogit + (1 - w) * oof_xgb
oof_ensemble <- oof_ensemble / rowSums(oof_ensemble)

cat("Point estimates: mlogit alone =", round(log_loss(oof_truth, oof_mlogit), 6),
    " | ensemble =", round(log_loss(oof_truth, oof_ensemble), 6), "\n\n")

# Respondent-level bootstrap: resample the 1135 respondents WITH replacement,
# pull all their (task) rows (with repeats), recompute pooled log loss for
# both models on each resample. This answers "how much would our CV number
# move around just from which respondents happened to be in the panel" --
# the noise floor neither review's point estimate accounts for.
set.seed(2026)
B <- 500
ll_mlogit_boot <- numeric(B)
ll_ensemble_boot <- numeric(B)

case_to_rows <- split(seq_along(train$Case), train$Case)

for (b in 1:B) {
  boot_cases <- sample(cases, length(cases), replace = TRUE)
  idx <- unlist(case_to_rows[as.character(boot_cases)])
  ll_mlogit_boot[b] <- log_loss(oof_truth[idx, ], oof_mlogit[idx, ])
  ll_ensemble_boot[b] <- log_loss(oof_truth[idx, ], oof_ensemble[idx, ])
}

diff_boot <- ll_mlogit_boot - ll_ensemble_boot  # positive => ensemble beats mlogit alone

cat("=== Bootstrap distribution (", B, "resamples of respondents) ===\n")
cat("mlogit alone:   mean =", round(mean(ll_mlogit_boot), 6), " SD =", round(sd(ll_mlogit_boot), 6),
    " 95% CI [", round(quantile(ll_mlogit_boot, 0.025), 6), ",", round(quantile(ll_mlogit_boot, 0.975), 6), "]\n")
cat("ensemble:       mean =", round(mean(ll_ensemble_boot), 6), " SD =", round(sd(ll_ensemble_boot), 6),
    " 95% CI [", round(quantile(ll_ensemble_boot, 0.025), 6), ",", round(quantile(ll_ensemble_boot, 0.975), 6), "]\n\n")

cat("=== Paired difference (mlogit alone - ensemble); positive = ensemble wins ===\n")
cat("mean diff:", round(mean(diff_boot), 6), " SD:", round(sd(diff_boot), 6), "\n")
cat("95% CI: [", round(quantile(diff_boot, 0.025), 6), ",", round(quantile(diff_boot, 0.975), 6), "]\n")
cat("Fraction of bootstrap resamples where ensemble beats mlogit alone:",
    round(mean(diff_boot > 0), 4), "\n\n")

cat("=== Interpretation ===\n")
cat("Point-estimate CV gap between models this session has been discussed at\n")
cat("the 0.003-0.006 scale. The bootstrap SD above is the actual noise floor\n")
cat("for a single-number CV estimate on this 1135-respondent panel.\n")
