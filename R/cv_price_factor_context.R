library(tidyverse)
library(mlogit)

source("R/log_loss.R")

attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
           "PP","KA","SC","TS","NV","MA","LB","AF","HU")
scaler_vars <- c("incomea","agea","milesa","nighta","genderind","Urbind","educind")

train <- read.csv("csv files/train.csv")
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

set.seed(4821)
cases <- unique(train_long$Case)
folds <- sample(rep(1:5, length.out = length(cases)))
names(folds) <- cases

oof_ll_num <- 0
oof_n <- 0

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

  truth_mat <- va_feat %>%
    select(chid, alt, chosen) %>%
    pivot_wider(names_from = alt, values_from = chosen, names_prefix = "Ch")

  pred_va <- predict(mod_k, newdata = mdat_va)
  truth_mat <- truth_mat[match(rownames(pred_va), truth_mat$chid), ]

  ll_k <- log_loss(truth_mat[, paste0("Ch", 1:4)], pred_va)
  n_k <- nrow(pred_va)
  oof_ll_num <- oof_ll_num + ll_k * n_k
  oof_n <- oof_n + n_k
  cat("Fold", k, "log loss:", round(ll_k, 6), " n =", n_k, "\n")
}

cat("\nPooled 5-fold CV log loss:", round(oof_ll_num / oof_n, 6), "\n")
