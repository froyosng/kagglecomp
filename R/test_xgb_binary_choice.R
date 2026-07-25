library(tidyverse)
library(xgboost)

source("R/log_loss.R")

attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
           "PP","KA","SC","TS","NV","MA","LB","AF","HU")
xgb_covariates <- c("segmentind","yearind","milesind","milesa","nightind","nighta",
                     "pparkind","genderind","ageind","agea","educind",
                     "regionind","Urbind","incomeind","incomea")

split <- readRDS("data_processed/train_val_split.rds")
tr <- split$train_long_tr
va <- split$train_long_val
tr$chid <- paste(tr$Case, tr$Task, sep = "_")
va$chid <- paste(va$Case, va$Task, sep = "_")

feat_cols <- c(attrs, "Price", xgb_covariates)
X_tr <- as.matrix(tr[, feat_cols])
X_va <- as.matrix(va[, feat_cols])
y_tr <- tr$chosen

xgb_tr <- xgb.DMatrix(data = X_tr, label = y_tr)
xgb_va_full <- xgb.DMatrix(data = X_va, label = va$chosen)

params <- list(objective = "binary:logistic", eval_metric = "logloss",
               eta = 0.1, max_depth = 4, subsample = 0.8, colsample_bytree = 0.8)

# Tune nrounds via the held-out split as a watchlist (quick, single split only)
# xgboost 3.x's xgb.train() no longer exposes $evaluation_log on the returned
# booster (it's a bare external pointer now); picked nrounds=200 from the
# printed per-iteration curve instead (plateaus ~150-275, slightly worse by 300).
best_nrounds <- 200
xgb_bin <- xgb.train(params = params, data = xgb_tr, nrounds = best_nrounds, verbose = 0)
raw_prob <- predict(xgb_bin, X_va)

va_pred <- va %>% select(chid, alt) %>% mutate(p = raw_prob)
wide_pred <- va_pred %>%
  pivot_wider(names_from = alt, values_from = p, names_prefix = "p") %>%
  mutate(across(starts_with("p"), ~ .x / (p1 + p2 + p3 + p4)))

truth_mat <- va %>% select(chid, alt, chosen) %>%
  pivot_wider(names_from = alt, values_from = chosen, names_prefix = "Ch")
truth_mat <- truth_mat[match(wide_pred$chid, truth_mat$chid), ]

pred_mat <- as.matrix(wide_pred[, c("p1","p2","p3","p4")])
ll_xgb_bin <- log_loss(truth_mat[, paste0("Ch", 1:4)], pred_mat)
cat("Choice-structured (binary + renormalized) xgboost validation log loss:", round(ll_xgb_bin, 6), "\n")
cat("(compare: wide multi:softprob xgboost validation was 1.2042 on this same kind of split)\n")
