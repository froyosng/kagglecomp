# Candidate B verdict: blend self-train's mlogit OOF into the SAME fixed ensemble
# recipe as the current best (0.80/0.20 mlogit/xgb, then the existing fold-
# cross-fitted MLP weights, none re-optimized), then run the respondent-clustered
# 100,000-replicate paired bootstrap against the current best's crossfit OOF.

source("R/codex_transductive_common.R")

train <- read.csv("csv files/train.csv")
truth <- as.matrix(train[, paste0("Ch", 1:4)])

saved_base <- readRDS("data_processed/oof_ensemble_v10.rds")
saved_mlp <- readRDS("data_processed/codex_behavioral_round/mlp_oof.rds")
mlp_weights <- read.csv("data_processed/codex_behavioral_round/mlp_cv_weights.csv",
                         stringsAsFactors = FALSE)
selftrain <- readRDS("data_processed/codex_transductive/selftrain_oof.rds")

candidate <- saved_mlp$candidates[[1]]
mlp <- saved_mlp$oof[[candidate]]
oof_xgb <- saved_base$oof_xgb
fold_of_row <- unname(saved_base$fold_of_case[as.character(train$Case)])
stopifnot(identical(fold_of_row, selftrain$fold_of_row))

v11_current <- 0.8 * saved_base$oof_mlogit + 0.2 * oof_xgb
crossfit_current <- matrix(NA_real_, nrow(train), 4L)
for (fold in 1:5) {
  rows <- fold_of_row == fold
  w <- mlp_weights$mlp_weight[mlp_weights$fold == fold]
  crossfit_current[rows, ] <- (1 - w) * v11_current[rows, ] + w * mlp[rows, ]
}
stopifnot(abs(log_loss_matrix(truth, crossfit_current) - 1.14378944178118) < 1e-10)
cat("Current-best crossfit reproduced:", sprintf("%.11f", log_loss_matrix(truth, crossfit_current)), "\n")

v11_selftrain <- 0.8 * selftrain$oof_mlogit_selftrain + 0.2 * oof_xgb
crossfit_selftrain <- matrix(NA_real_, nrow(train), 4L)
for (fold in 1:5) {
  rows <- fold_of_row == fold
  w <- mlp_weights$mlp_weight[mlp_weights$fold == fold]
  crossfit_selftrain[rows, ] <- (1 - w) * v11_selftrain[rows, ] + w * mlp[rows, ]
}
cat("Candidate B (selftrain) crossfit logloss:", sprintf("%.11f", log_loss_matrix(truth, crossfit_selftrain)), "\n")
cat("Point gain (current_best - selftrain, positive = selftrain better):",
    sprintf("%.6f", log_loss_matrix(truth, crossfit_current) - log_loss_matrix(truth, crossfit_selftrain)), "\n")

case_gain <- unname(tapply(
  row_log_loss(truth, crossfit_current) - row_log_loss(truth, crossfit_selftrain),
  train$Case, mean
))
point_gain <- mean(case_gain)
boot <- bootstrap_case_means(case_gain)

family_size <- 2L
alpha_family <- 0.05 / family_size

result <- data.frame(
  candidate = "selftrain_B",
  current_best_logloss = log_loss_matrix(truth, crossfit_current),
  candidate_logloss = log_loss_matrix(truth, crossfit_selftrain),
  point_gain = point_gain,
  bootstrap_mean = mean(boot),
  bootstrap_sd = sd(boot),
  lower_95 = unname(quantile(boot, 0.025)),
  upper_95 = unname(quantile(boot, 0.975)),
  lower_99 = unname(quantile(boot, 0.005)),
  upper_99 = unname(quantile(boot, 0.995)),
  lower_bonf_95 = unname(quantile(boot, alpha_family / 2)),
  upper_bonf_95 = unname(quantile(boot, 1 - alpha_family / 2)),
  win_rate = mean(boot > 0),
  n_boot = length(boot),
  clears_ordinary_95 = unname(quantile(boot, 0.025)) > 0
)

print(result, digits = 8)

dir.create("data_processed/codex_transductive", recursive = TRUE, showWarnings = FALSE)
write.csv(result, "data_processed/codex_transductive/selftrain_precision.csv", row.names = FALSE)
saveRDS(list(result = result, case_gain = case_gain, boot = boot,
             crossfit_current = crossfit_current, crossfit_selftrain = crossfit_selftrain),
        "data_processed/codex_transductive/selftrain_precision.rds")
