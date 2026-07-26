library(tidyverse)
library(mlogit)
source("R/log_loss.R")

oof <- readRDS("data_processed/oof_ensemble_v10.rds")
oof_mlogit <- oof$oof_mlogit
oof_xgb <- oof$oof_xgb
oof_truth <- oof$oof_truth
fold_of_case <- oof$fold_of_case

train <- read.csv("csv files/train.csv")
fold_of_row <- fold_of_case[as.character(train$Case)]

eps <- 1e-9
log_p_mlogit <- log(pmax(oof_mlogit, eps))
log_p_xgb <- log(pmax(oof_xgb, eps))

n <- nrow(train)
alt_long <- expand.grid(row = 1:n, alt = 1:4)
alt_long$chosen <- as.integer(oof_truth[cbind(alt_long$row, alt_long$alt)] == 1)
alt_long$lp_mlogit <- log_p_mlogit[cbind(alt_long$row, alt_long$alt)]
alt_long$lp_xgb <- log_p_xgb[cbind(alt_long$row, alt_long$alt)]
alt_long$fold <- fold_of_row[alt_long$row]

# Proper stacking: a conditional logit ("logistic mixture of experts") using
# log(p_mlogit) and log(p_xgb) as alternative-varying covariates, respecting
# the true 4-way softmax choice likelihood (unlike a per-row binary GLM, which
# would repeat the same weakness the binary-xgboost experiment already showed
# -- see cleaning_log.md). More flexible than a single arithmetic blend weight:
# 2 free coefficients instead of 1, log-linear (geometric) pooling instead of
# linear pooling. Nested 5-fold CV: refit the meta-model on 4 folds' OOF preds,
# evaluate on the 5th fold's OOF preds, so the reported number isn't optimistic
# from the meta-model having "seen" its own eval data.
meta_pred <- matrix(NA_real_, n, 4)
for (k in 1:5) {
  fit_data <- alt_long[alt_long$fold != k, ]
  eval_data <- alt_long[alt_long$fold == k, ]

  mdat_fit <- dfidx(fit_data, idx = c("row", "alt"), choice = "chosen")
  meta_mod <- mlogit(chosen ~ lp_mlogit + lp_xgb | 0, data = mdat_fit)

  mdat_eval <- dfidx(eval_data, idx = c("row", "alt"), choice = "chosen")
  pred_k <- predict(meta_mod, newdata = mdat_eval)
  row_idx <- as.integer(rownames(pred_k))
  meta_pred[row_idx, ] <- pred_k

  cat("Fold", k, "meta coefs:", round(coef(meta_mod), 4), "\n")
}

ll_stack <- log_loss(oof_truth, meta_pred)
cat("\nNested-CV stacked (logistic mixture of experts) log loss:", round(ll_stack, 6), "\n")

weights <- seq(0, 1, by = 0.01)
ll_blend <- sapply(weights, function(w) {
  blend <- w * oof_mlogit + (1 - w) * oof_xgb
  blend <- blend / rowSums(blend)
  log_loss(oof_truth, blend)
})
cat("Best fixed linear blend (for comparison):", round(min(ll_blend), 6), "at w =", weights[which.min(ll_blend)], "\n")
