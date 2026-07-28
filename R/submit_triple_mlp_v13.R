## Submission candidate: m8trpg + Price x z(income) x z(mileage) (triple
## interaction) + xgboost + MLP, combining the two independently-validated
## improvements found this session (never blended together before).
##
## CV-tested (own analysis, not yet Codex-verified in a dedicated round):
## honest fold-cross-fitted 3-way blend scores 1.143328 vs the current best
## v11+MLP's 1.143789 (gain +0.000462, bootstrap 95% CI [-0.000754,
## +0.001642] -- crosses zero, NOT confirmed better than the current best)
## and vs plain v11's 1.145094 (gain +0.001767, CI [+0.000054, +0.003420] --
## excludes zero, barely). Paired public-LB-sized simulation anchored to the
## current best's REAL public score (1.201) implies a 95% range of
## [1.1979, 1.2033] with a 63.6% chance of beating the current best on the
## same draw. Reuses the already-fit MLP test predictions from
## data_processed/codex_behavioral_round/mlp_full_test_candidate.rds (no
## MLP refit) and fits the triple-interaction mlogit + xgboost fresh on all
## training data.

library(tidyverse)
library(mlogit)
library(xgboost)

attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
           "PP","KA","SC","TS","NV","MA","LB","AF","HU")
scaler_vars <- c("incomea","agea","milesa","nighta","genderind","Urbind","educind")
xgb_covariates <- c("segmentind","yearind","milesind","milesa","nightind","nighta",
                     "pparkind","genderind","ageind","agea","educind",
                     "regionind","Urbind","incomeind","incomea")

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
  df$P_income_miles <- df$Price_num * z$incomea * z$milesa
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
  "is_cheapest", "is_dearest", "price_gap_min", "price_gap_max",
  "P_income_miles"
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
cat("Component A (m8trpg + triple interaction) done.\n")

xgb_feature_cols <- function(df) {
  attr_wide <- paste0(rep(attrs, each = 4), rep(1:4, times = length(attrs)))
  price_wide <- paste0("Price", 1:4)
  df[, c(attr_wide, price_wide, xgb_covariates)]
}
X_tr <- as.matrix(xgb_feature_cols(train))
X_te <- as.matrix(xgb_feature_cols(test))
y_tr <- train$Choice - 1
xgb_tr <- xgb.DMatrix(data = X_tr, label = y_tr)
xgb_te <- xgb.DMatrix(data = X_te)
xgb_params <- list(objective = "multi:softprob", num_class = 4, eval_metric = "mlogloss",
                    eta = 0.1, max_depth = 4, subsample = 0.8, colsample_bytree = 0.8)
xgb_full <- xgb.train(params = xgb_params, data = xgb_tr, nrounds = 73, verbose = 0)
pred_xgb <- predict(xgb_full, xgb_te)
if (is.null(dim(pred_xgb))) pred_xgb <- matrix(pred_xgb, ncol = 4, byrow = TRUE)
cat("Component B (xgboost) done.\n")

mlp_candidate <- readRDS("data_processed/codex_behavioral_round/mlp_full_test_candidate.rds")
pred_mlp <- mlp_candidate$test_prediction
stopifnot(
  identical(dim(pred_mlp), c(nrow(test), 4L)),
  !anyNA(pred_mlp),
  max(abs(rowSums(pred_mlp) - 1)) < 1e-8
)
cat("Component C (MLP, reused from already-fit full-data candidate) done.\n")

## Average of the fold-cross-fitted weights found for this 3-way combination
## (mlogit 0.76/0.72/0.72/0.72/0.74, xgb 0.08/0.10/0.12/0.14/0.12,
## mlp 0.16/0.18/0.16/0.14/0.14) -- same principle as the original MLP
## candidate's 15% weight (average of its own fold-cross-fitted weights),
## not a re-optimized/overfit single value.
w <- c(mlogit = 0.732, xgb = 0.112, mlp = 0.156)
stopifnot(abs(sum(w) - 1) < 1e-6)
blend <- w["mlogit"] * pred_mlogit + w["xgb"] * pred_xgb + w["mlp"] * pred_mlp
blend <- blend / rowSums(blend)

submission <- data.frame(No = test$No, Ch1 = blend[,1], Ch2 = blend[,2], Ch3 = blend[,3], Ch4 = blend[,4])
sample_sub <- read.csv("csv files/sample_submission.csv")
stopifnot(all(names(submission) == names(sample_sub)),
          all(submission$No == sample_sub$No), !anyNA(submission),
          max(abs(rowSums(submission[,2:5]) - 1)) < 1e-8)
write.csv(submission, "submission_triple_mlp_v13.csv", row.names = FALSE)
cat("Wrote submission_triple_mlp_v13.csv\n")
cat("CV expectation: 1.143328 (honest fold-cross-fitted); current best (v11+MLP) CV was 1.143789, public 1.201\n")
