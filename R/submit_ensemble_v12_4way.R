## Submission candidate: 4-way blend of mlogit_m8trpg + rank:ndcg xgboost +
## retuned xgboost + glmnet-Cox, using the GLOBAL (all-fold, in-sample)
## weights from Codex's ensemble search (ensemble_meta_cv.csv, pool
## "mlogit+rank_ndcg+retuned_xgb+cox_min": 0.6793/0.0964/0.1153/0.1090).
## CV-tested: global OOF 1.144029, honest fold-cross-fitted 1.144363, vs
## ensemble_v11's official 1.145094 (+0.0007 gain). Respondent-bootstrap 95%
## CI on that gain crosses zero ([-0.000214, 0.001775]) -- NOT CV-confirmed
## as a real improvement; submitted for empirical validation against the
## public leaderboard, not because it cleared the project's submission bar.
##
## Reuses already-verified helper functions from the Codex modeling scripts
## (sourced with CODEX_STAGE set to a value that matches neither "screen" nor
## "cv", so only their function definitions load -- their own screen/cv
## blocks do not execute).

suppressPackageStartupMessages({
  library(tidyverse)
  library(mlogit)
  library(dfidx)
  library(xgboost)
  library(glmnet)
  library(survival)
})

Sys.setenv(CODEX_STAGE = "submission_build")
source("R/codex_modeling_common.R")
source("R/codex_rank_xgb.R")
source("R/codex_xgb_retune.R")
source("R/codex_glmnet_cox_ensemble.R")

attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
           "PP","KA","SC","TS","NV","MA","LB","AF","HU")
scaler_vars <- c("incomea","agea","milesa","nighta","genderind","Urbind","educind")

train <- read.csv("csv files/train.csv")
test <- read.csv("csv files/test.csv")
train$Choice <- max.col(train[, c("Ch1","Ch2","Ch3","Ch4")])

pat <- paste0("^(", paste(c(attrs, "Price", "Ch"), collapse = "|"), ")([1-4])$")
to_long <- function(df) {
  df %>%
    pivot_longer(cols = matches(pat), names_to = c(".value", "alt"),
                 names_pattern = "^(.*)([1-4])$") %>%
    mutate(alt = as.integer(alt), chosen = as.integer(Ch == 1),
           chid = paste(Case, Task, sep = "_"),
           d2 = as.integer(alt == 2), d3 = as.integer(alt == 3))
}
train_long <- to_long(train)
test$Ch1 <- 1; test$Ch2 <- 0; test$Ch3 <- 0; test$Ch4 <- 0
test_long <- to_long(test)

stopifnot(identical(test$No, sort(test$No)))  # confirms row-order == ascending No

## --- Component A: mlogit m8trpg (same code as submit_ensemble_v11.R) ---
ctr <- sapply(train_long[scaler_vars], mean, na.rm = TRUE)
scl <- sapply(train_long[scaler_vars], sd, na.rm = TRUE)
scl[scl == 0] <- 1

make_features <- function(df, ctr, scl) {
  df <- as.data.frame(df)
  z <- sweep(sweep(df[scaler_vars], 2, ctr, "-"), 2, scl, "/")
  df$inside <- as.integer(df$alt != 4)
  df$Price_num <- as.numeric(df$Price)
  for (k in 2:12) df[[paste0("Pr_lvl", k)]] <- as.integer(df$Price_num == k)
  df$Task_c <- (as.numeric(df$Task) - 10) / 9
  df$P_income <- df$Price_num * z$incomea
  df$P_age <- df$Price_num * z$agea
  df$P_miles <- df$Price_num * z$milesa
  df$P_night <- df$Price_num * z$nighta
  df$In_income <- df$inside * z$incomea
  df$In_age <- df$inside * z$agea
  df$In_miles <- df$inside * z$milesa
  df$In_night <- df$inside * z$nighta
  df$In_gender <- df$inside * z$genderind
  df$In_urb <- df$inside * z$Urbind
  df$In_educ <- df$inside * z$educind
  for (s in 2:6) {
    df[[paste0("P_seg", s)]] <- df$Price_num * (df$segmentind == s)
    df[[paste0("In_seg", s)]] <- df$inside * (df$segmentind == s)
  }
  df$P_task <- df$Price_num * df$Task_c
  df$In_task <- df$inside * df$Task_c
  for (s in 2:5) {
    df[[paste0("P_region", s)]] <- df$Price_num * (df$regionind == s)
    df[[paste0("In_region", s)]] <- df$inside * (df$regionind == s)
    df[[paste0("P_ppark", s)]] <- df$Price_num * (df$pparkind == s)
    df[[paste0("In_ppark", s)]] <- df$inside * (df$pparkind == s)
  }
  task_stats <- df %>% group_by(chid) %>%
    summarise(price_min = min(Price_num[inside == 1], na.rm = TRUE),
              price_max = max(Price_num[inside == 1], na.rm = TRUE), .groups = "drop")
  df <- left_join(df, task_stats, by = "chid")
  df$is_cheapest <- as.integer(df$inside == 1 & df$Price_num == df$price_min)
  df$is_dearest <- as.integer(df$inside == 1 & df$Price_num == df$price_max)
  df$price_gap_min <- ifelse(df$inside == 1, df$Price_num - df$price_min, 0)
  df$price_gap_max <- ifelse(df$inside == 1, df$price_max - df$Price_num, 0)
  df
}

attr_terms <- paste0("factor(", attrs, ")")
price_terms <- paste0("Pr_lvl", 2:12)
int_terms <- c(
  "P_income", "P_age", "P_miles", "P_night",
  "In_income", "In_age", "In_miles", "In_night",
  "In_gender", "In_urb", "In_educ",
  paste0("P_seg", 2:6), paste0("In_seg", 2:6),
  "P_task", "In_task",
  paste0("P_region", 2:5), paste0("In_region", 2:5),
  paste0("P_ppark", 2:5), paste0("In_ppark", 2:5),
  "is_cheapest", "is_dearest", "price_gap_min", "price_gap_max"
)
fml <- as.formula(paste("chosen ~", paste(c(attr_terms, price_terms, "d2", "d3", int_terms), collapse = " + "), "| 0"))

train_feat <- make_features(train_long, ctr, scl)
test_feat <- make_features(test_long, ctr, scl)
mdat_tr <- dfidx(train_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
mdat_te <- dfidx(test_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
mod_full <- mlogit(fml, data = mdat_tr)
pred_mlogit_raw <- predict(mod_full, newdata = mdat_te)
test_map <- test_long %>% distinct(chid, No)
pred_mlogit <- pred_mlogit_raw[match(test_map$chid, rownames(pred_mlogit_raw)), ]
pred_mlogit <- pred_mlogit[match(test$No, test_map$No), ]
stopifnot(!anyNA(pred_mlogit))
cat("Component A (mlogit m8trpg) done.\n")

## --- Component B: rank:ndcg xgboost ---
## Temperature scale is NOT calibrated in-sample here: the ranker
## substantially overfits its own training data (300 boosting rounds), so an
## in-sample-calibrated scale would be contaminated by that overfitting (a
## trial run gave in-sample log loss 1.077, far below any genuine held-out
## number seen all day, and a scale of 1.375 vs. the CV round's honestly
## cross-fitted ~1.125). Reuse the already cross-validated scale instead:
## R/codex_rank_xgb.R's rank_cv.csv reports global_scale=1.125 and
## fold_scales 1.175/1.100/1.125/1.100/1.125 (mean exactly 1.125) for this
## exact config -- a properly held-out-calibrated number, not a guess.
rank_scale <- 1.125
train_long_sorted <- sort_long_tasks(train_long)
rank_model <- fit_ranker(
  train_long_sorted, rank_params("rank:ndcg", eta = 0.10, depth = 4), nrounds = 300
)
test_margin <- predict_ranker_margin(rank_model, test_long)
pred_rank_sorted <- softmax_margins(test_margin, rank_scale)
rank_task_order <- sort(unique(sort_long_tasks(test_long)$No))
pred_rank <- pred_rank_sorted[match(test$No, rank_task_order), ]
stopifnot(!anyNA(pred_rank))
cat("Component B (rank:ndcg xgboost) done.\n")

## --- Component C: retuned xgboost ---
retuned_spec <- data.frame(
  eta = 0.03, max_depth = 6L, min_child_weight = 8,
  subsample = 0.65, colsample_bytree = 0.80,
  reg_alpha = 0, reg_lambda = 5, gamma = 0, nrounds = 243L
)
retuned_model <- fit_multiclass(train, retuned_spec)
pred_retuned <- predict_multiclass(retuned_model, test)
stopifnot(!anyNA(pred_retuned), nrow(pred_retuned) == nrow(test))
cat("Component C (retuned xgboost) done.\n")

## --- Component D: glmnet stratified-Cox ---
attr_max <- as.list(vapply(
  rbind(train_long[, attrs], test_long[, attrs]), max, numeric(1)
))
cox_fit <- fit_cox_cv(train_long, attr_max, seed = 4821)
cat(sprintf("Cox lambda.min: %.6f; selected interactions: %d\n",
            cox_fit$cvfit$lambda.min,
            selected_interactions(cox_fit, cox_fit$cvfit$lambda.min)))
pred_cox_sorted <- predict_cox_choice(cox_fit, test_long, cox_fit$cvfit$lambda.min)
cox_task_order <- sort(unique(sort_long_tasks(test_long)$No))
pred_cox <- pred_cox_sorted[match(test$No, cox_task_order), ]
stopifnot(!anyNA(pred_cox))
cat("Component D (glmnet-Cox) done.\n")

## --- Blend with the global (all-fold, in-sample) weights from
## data_processed/codex/ensemble_meta_cv.csv, pool
## "mlogit+rank_ndcg+retuned_xgb+cox_min": 0.6793/0.0964/0.1153/0.1090 ---
w <- c(mlogit = 0.6793, rank_ndcg = 0.0964, retuned_xgb = 0.1153, cox_min = 0.1090)
stopifnot(abs(sum(w) - 1) < 1e-6)
blend <- w["mlogit"] * pred_mlogit + w["rank_ndcg"] * pred_rank +
  w["retuned_xgb"] * pred_retuned + w["cox_min"] * pred_cox
blend <- blend / rowSums(blend)

submission <- data.frame(No = test$No, Ch1 = blend[,1], Ch2 = blend[,2], Ch3 = blend[,3], Ch4 = blend[,4])
sample_sub <- read.csv("csv files/sample_submission.csv")
stopifnot(all(names(submission) == names(sample_sub)),
          all(submission$No == sample_sub$No), !anyNA(submission))
write.csv(submission, "submission_ensemble_v12_4way.csv", row.names = FALSE)
cat("Wrote submission_ensemble_v12_4way.csv\n")
cat("CV expectation: 1.144363 (fold-cross-fitted) / 1.144029 (global); ensemble_v11 CV was 1.145094\n")
