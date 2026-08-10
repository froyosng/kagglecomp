# Sanity check: reproduce the officially-logged "current best" (ensemble_v11 + MLP)
# crossfit OOF log loss from the copied artifacts, before building anything new on
# top of them. Mirrors R/codex_mlp_precision.R's baseline reconstruction exactly.

log_loss_matrix <- function(truth, pred) {
  pred <- pmin(pmax(pred, 1e-15), 1 - 1e-15)
  -mean(rowSums(truth * log(pred)))
}

train <- read.csv("csv files/train.csv")
truth <- as.matrix(train[, paste0("Ch", 1:4)])

saved_base <- readRDS("data_processed/oof_ensemble_v10.rds")
saved_mlp <- readRDS("data_processed/codex_behavioral_round/mlp_oof.rds")
weights <- read.csv("data_processed/codex_behavioral_round/mlp_cv_weights.csv",
                     stringsAsFactors = FALSE)

candidate <- saved_mlp$candidates[[1]]
mlp <- saved_mlp$oof[[candidate]]
baseline <- saved_base$oof_mlogit
xgb <- saved_base$oof_xgb
v11 <- 0.8 * baseline + 0.2 * xgb
fold_of_row <- unname(saved_base$fold_of_case[as.character(train$Case)])

cat("mlogit-only OOF logloss:", log_loss_matrix(truth, baseline), "(expect 1.1470212110518)\n")
cat("v11 (0.8 mlogit + 0.2 xgb) OOF logloss:", log_loss_matrix(truth, v11), "(expect 1.14509421298673)\n")
cat("mlp-only OOF logloss:", log_loss_matrix(truth, mlp), "(expect 1.19054334979533)\n")

crossfit <- matrix(NA_real_, nrow(train), 4L)
for (fold in 1:5) {
  rows <- fold_of_row == fold
  weight <- weights$mlp_weight[weights$fold == fold]
  crossfit[rows, ] <- (1 - weight) * v11[rows, ] + weight * mlp[rows, ]
}
cat("crossfit (current best) OOF logloss:", log_loss_matrix(truth, crossfit), "(expect 1.14378944178118)\n")

stopifnot(
  abs(log_loss_matrix(truth, baseline) - 1.1470212110518) < 1e-8,
  abs(log_loss_matrix(truth, v11) - 1.14509421298673) < 1e-10,
  abs(log_loss_matrix(truth, mlp) - 1.19054334979533) < 1e-10,
  abs(log_loss_matrix(truth, crossfit) - 1.14378944178118) < 1e-10
)
cat("ALL BASELINE CHECKS PASSED\n")

saveRDS(list(truth = truth, v11 = v11, mlp = mlp, crossfit = crossfit,
             fold_of_row = fold_of_row, fold_of_case = saved_base$fold_of_case,
             oof_mlogit = baseline, oof_xgb = xgb),
        "data_processed/codex_transductive/current_best_reconstruction.rds")
