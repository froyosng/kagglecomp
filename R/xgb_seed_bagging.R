## Seed-bagging test: does averaging the SAME xgboost config across many random
## seeds (pure variance reduction, no new model complexity) improve the
## ensemble, given the project has already confirmed that diverse-model
## variance reduction (mlogit+xgboost blend) genuinely helps generalization?
## Reuses the exact hyperparameters, feature set, and canonical fold split
## from R/cv_ensemble_v10.R so this is an apples-to-apples comparison against
## the already-logged single-seed xgboost (CV 1.178668) and ensemble_v11.

suppressPackageStartupMessages({
  library(tidyverse)
  library(xgboost)
})
source("R/log_loss.R")

attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
           "PP","KA","SC","TS","NV","MA","LB","AF","HU")
xgb_covariates <- c("segmentind","yearind","milesind","milesa","nightind","nighta",
                     "pparkind","genderind","ageind","agea","educind",
                     "regionind","Urbind","incomeind","incomea")

train <- read.csv("csv files/train.csv")
train$Choice <- max.col(train[, c("Ch1","Ch2","Ch3","Ch4")])

saved <- readRDS("data_processed/oof_ensemble_v10.rds")
oof_mlogit <- saved$oof_mlogit
oof_xgb_single <- saved$oof_xgb
fold_of_case <- saved$fold_of_case
truth <- saved$oof_truth
stopifnot(abs(log_loss(truth, oof_xgb_single) - 1.178668) < 1e-4)
stopifnot(abs(log_loss(truth, oof_mlogit) - 1.147021) < 1e-4)

train$fold <- fold_of_case[as.character(train$Case)]
stopifnot(!anyNA(train$fold))

xgb_feature_cols <- function(df) {
  attr_wide <- paste0(rep(attrs, each = 4), rep(1:4, times = length(attrs)))
  price_wide <- paste0("Price", 1:4)
  df[, c(attr_wide, price_wide, xgb_covariates)]
}

n_seeds <- 20L
seeds <- 1:n_seeds

per_seed_fold_pred <- vector("list", n_seeds)  # for a learning-curve check

for (k in 1:5) {
  tr_k <- train[train$fold != k, ]
  va_k <- train[train$fold == k, ]
  X_tr <- as.matrix(xgb_feature_cols(tr_k))
  X_va <- as.matrix(xgb_feature_cols(va_k))
  y_tr <- tr_k$Choice - 1
  xgb_tr <- xgb.DMatrix(data = X_tr, label = y_tr)
  xgb_va <- xgb.DMatrix(data = X_va)
  row_idx <- match(va_k$No, train$No)

  fold_seed_preds <- array(NA_real_, dim = c(nrow(va_k), 4, n_seeds))
  for (si in seq_along(seeds)) {
    params <- list(objective = "multi:softprob", num_class = 4,
                    eval_metric = "mlogloss",
                    eta = 0.1, max_depth = 4, subsample = 0.8,
                    colsample_bytree = 0.8, seed = seeds[si])
    mod <- xgb.train(params = params, data = xgb_tr, nrounds = 73, verbose = 0)
    pred <- predict(mod, xgb_va, reshape = TRUE)
    fold_seed_preds[, , si] <- pred
  }
  # Cumulative average over an increasing number of seeds, to see how many
  # seeds are actually needed (diminishing-returns check), plus the full
  # 20-seed average used for the headline comparison.
  cum_avg <- apply(fold_seed_preds, c(1, 2), function(v) cumsum(v) / seq_along(v))
  # cum_avg has dim (n_seeds, n_rows, 4) after apply's reordering; store per-n
  for (si in seq_along(seeds)) {
    if (is.null(per_seed_fold_pred[[si]])) {
      per_seed_fold_pred[[si]] <- matrix(NA_real_, nrow(train), 4)
    }
    per_seed_fold_pred[[si]][row_idx, ] <- cum_avg[si, , ]
  }
  cat(sprintf("fold %d done (%d seeds)\n", k, n_seeds))
}

## Build the learning curve: pooled OOF log loss using the first m seeds' average,
## for m = 1..20.
learning_curve <- data.frame(
  n_seeds = seeds,
  ordinary_logloss = vapply(seeds, function(m) log_loss(truth, per_seed_fold_pred[[m]]), numeric(1))
)
cat("\n=== Seed-count learning curve (pooled OOF xgboost alone) ===\n")
print(learning_curve, digits = 6)

bagged_xgb <- per_seed_fold_pred[[n_seeds]]
stopifnot(!anyNA(bagged_xgb))

cat(sprintf("\nSingle-seed xgboost (logged):      %.6f\n", log_loss(truth, oof_xgb_single)))
cat(sprintf("%d-seed bagged xgboost:             %.6f\n", n_seeds, log_loss(truth, bagged_xgb)))

## Fixed 0.80/0.20 blend (matching ensemble_v11's actual submitted weight)
v11_single <- 0.8 * oof_mlogit + 0.2 * oof_xgb_single
v11_bagged <- 0.8 * oof_mlogit + 0.2 * bagged_xgb
v11_loss <- log_loss(truth, v11_single)
stopifnot(abs(v11_loss - 1.145094) < 5e-6)
cat(sprintf("\nensemble_v11 (single-seed xgb, official):     %.6f\n", v11_loss))
cat(sprintf("ensemble_v11 with %d-seed bagged xgb (fixed 0.80): %.6f\n",
            n_seeds, log_loss(truth, v11_bagged)))

## Diagnostic: re-optimized blend weight with the bagged xgboost.
weights <- seq(0, 1, by = 0.01)
grid_loss <- vapply(weights, function(w) log_loss(truth, w * oof_mlogit + (1 - w) * bagged_xgb), numeric(1))
best_w <- weights[which.min(grid_loss)]
cat(sprintf("Diagnostic best blend weight with bagged xgb: %.2f -> %.6f\n", best_w, min(grid_loss)))

## Respondent-clustered bootstrap: bagged-blend vs official ensemble_v11.
row_loss <- function(pred, eps = 1e-15) {
  pred <- pmin(pmax(pred / rowSums(pred), eps), 1 - eps)
  -rowSums(truth * log(pred))
}
base_rl <- row_loss(v11_single)
cand_rl <- row_loss(v11_bagged)
resp_gain <- tapply(base_rl - cand_rl, train$Case, mean)
set.seed(4821)
B <- 2000L
boot <- replicate(B, mean(sample(resp_gain, length(resp_gain), replace = TRUE)))
cat(sprintf(
  "\nBagged-blend gain vs ensemble_v11: point %.6f; bootstrap mean %.6f; 95%% CI [%.6f, %.6f]; win rate %.3f\n",
  mean(resp_gain), mean(boot), quantile(boot, 0.025), quantile(boot, 0.975), mean(boot > 0)
))

## Also check: bagged xgboost ALONE vs single-seed xgboost alone (isolate the
## pure variance-reduction effect before any blending).
base_rl2 <- row_loss(oof_xgb_single)
cand_rl2 <- row_loss(bagged_xgb)
resp_gain2 <- tapply(base_rl2 - cand_rl2, train$Case, mean)
set.seed(4821)
boot2 <- replicate(B, mean(sample(resp_gain2, length(resp_gain2), replace = TRUE)))
cat(sprintf(
  "Bagged xgb ALONE vs single-seed xgb ALONE: point %.6f; 95%% CI [%.6f, %.6f]; win rate %.3f\n",
  mean(resp_gain2), quantile(boot2, 0.025), quantile(boot2, 0.975), mean(boot2 > 0)
))

write.csv(learning_curve, "data_processed/codex_shift/seed_bagging_learning_curve.csv", row.names = FALSE)
saveRDS(list(bagged_xgb = bagged_xgb, n_seeds = n_seeds, fold_of_case = fold_of_case),
        "data_processed/codex_shift/seed_bagging_oof.rds")
