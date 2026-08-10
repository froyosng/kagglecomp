library(tidyverse)
library(mlogit)
library(xgboost)

source("R/log_loss.R")

attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
           "PP","KA","SC","TS","NV","MA","LB","AF","HU")
scaler_vars <- c("incomea","agea","milesa","nighta","genderind","Urbind","educind")
xgb_covariates <- c("segmentind","yearind","milesind","milesa","nightind","nighta",
                     "pparkind","genderind","ageind","agea","educind",
                     "regionind","Urbind","incomeind","incomea")

train <- read.csv("csv files/train.csv")
train$Choice <- max.col(train[, c("Ch1","Ch2","Ch3","Ch4")])

pat <- paste0("^(", paste(c(attrs, "Price", "Ch"), collapse = "|"), ")([1-4])$")
train_long <- train %>%
  pivot_longer(cols = matches(pat), names_to = c(".value", "alt"),
               names_pattern = "^(.*)([1-4])$") %>%
  mutate(alt = as.integer(alt), chosen = as.integer(Ch == 1),
         chid = paste(Case, Task, sep = "_"),
         d2 = as.integer(alt == 2), d3 = as.integer(alt == 3))

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

xgb_feature_cols <- function(df) {
  attr_wide <- paste0(rep(attrs, each = 4), rep(1:4, times = length(attrs)))
  price_wide <- paste0("Price", 1:4)
  df[, c(attr_wide, price_wide, xgb_covariates)]
}

set.seed(4821)
cases <- unique(train_long$Case)
folds <- sample(rep(1:5, length.out = length(cases)))
names(folds) <- cases
fold_of_case <- setNames(folds, cases)
train$fold <- fold_of_case[as.character(train$Case)]

oof_mlogit <- matrix(NA_real_, nrow(train), 4)
oof_xgb <- matrix(NA_real_, nrow(train), 4)
oof_truth <- as.matrix(train[, c("Ch1","Ch2","Ch3","Ch4")])

for (k in 1:5) {
  val_cases_k <- cases[folds == k]
  tr_k <- train_long %>% filter(!(Case %in% val_cases_k))
  va_k <- train_long %>% filter(Case %in% val_cases_k)

  ctr <- sapply(tr_k[scaler_vars], mean, na.rm = TRUE)
  scl <- sapply(tr_k[scaler_vars], sd, na.rm = TRUE)
  scl[scl == 0] <- 1

  tr_feat <- make_features(tr_k, ctr, scl)
  va_feat <- make_features(va_k, ctr, scl)

  mdat_tr <- dfidx(tr_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
  mdat_va <- dfidx(va_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
  mod_k <- mlogit(fml, data = mdat_tr)
  pred_mlogit_k <- predict(mod_k, newdata = mdat_va)

  # map dfidx rownames (chid) back to original train row numbers (No) for this fold's val rows
  va_map <- va_k %>% distinct(chid, No)
  row_idx <- match(va_map$No, train$No)
  pred_mlogit_ordered <- pred_mlogit_k[match(va_map$chid, rownames(pred_mlogit_k)), ]
  oof_mlogit[row_idx, ] <- pred_mlogit_ordered

  train_k_wide <- train[train$fold != k, ]
  val_k_wide <- train[train$fold == k, ]
  X_tr <- as.matrix(xgb_feature_cols(train_k_wide))
  X_va <- as.matrix(xgb_feature_cols(val_k_wide))
  y_tr <- train_k_wide$Choice - 1
  xgb_tr <- xgb.DMatrix(data = X_tr, label = y_tr)
  xgb_va <- xgb.DMatrix(data = X_va)
  xgb_params <- list(objective = "multi:softprob", num_class = 4, eval_metric = "mlogloss",
                      eta = 0.1, max_depth = 4, subsample = 0.8, colsample_bytree = 0.8)
  xgb_mod_k <- xgb.train(params = xgb_params, data = xgb_tr, nrounds = 73, verbose = 0)
  pred_xgb_k <- predict(xgb_mod_k, xgb_va, reshape = TRUE)
  xgb_row_idx <- match(val_k_wide$No, train$No)
  oof_xgb[xgb_row_idx, ] <- pred_xgb_k

  cat("Fold", k, "done. mlogit ll:", round(log_loss(oof_truth[row_idx, ], pred_mlogit_ordered), 6),
      " xgb ll:", round(log_loss(oof_truth[xgb_row_idx, ], pred_xgb_k), 6), "\n")
}

ll_mlogit <- log_loss(oof_truth, oof_mlogit)
ll_xgb <- log_loss(oof_truth, oof_xgb)
cat("\nPooled OOF mlogit (price-factor+context):", round(ll_mlogit, 6), "\n")
cat("Pooled OOF xgboost:", round(ll_xgb, 6), "\n")

weights <- seq(0, 1, by = 0.05)
ll_blend <- sapply(weights, function(w) {
  blend <- w * oof_mlogit + (1 - w) * oof_xgb
  blend <- blend / rowSums(blend)
  log_loss(oof_truth, blend)
})
best_w <- weights[which.min(ll_blend)]
cat("\nBlend weight (mlogit share) vs pooled OOF log loss:\n")
print(data.frame(w_mlogit = weights, ll = round(ll_blend, 6)))
cat("\nBest weight:", best_w, " best pooled CV log loss:", round(min(ll_blend), 6), "\n")

saveRDS(list(oof_mlogit = oof_mlogit, oof_xgb = oof_xgb, oof_truth = oof_truth,
             fold_of_case = fold_of_case),
        "data_processed/oof_ensemble_v10.rds")
