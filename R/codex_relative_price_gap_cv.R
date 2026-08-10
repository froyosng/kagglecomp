# Canonical 5-fold CV confirmation for the dispersion-relative price-gap
# screen (codex_relative_price_gap_screen.R showed a small positive
# single-split gain, +0.000261).
#
# Pre-registration: codex_relative_price_gap_preregister.md
#
# Run:
#   source("R/codex_relative_price_gap_cv.R")

suppressPackageStartupMessages({
  library(mlogit)
  library(dfidx)
})

output_dir <- file.path("data_processed", "codex_relative_price_gap")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

attrs <- c(
  "CC", "GN", "NS", "BU", "FA", "LD", "BZ", "FC", "FP", "RP",
  "PP", "KA", "SC", "TS", "NV", "MA", "LB", "AF", "HU"
)
scaler_vars <- c("incomea", "agea", "milesa", "nighta", "genderind", "Urbind", "educind")

to_long <- function(wide) {
  pattern <- paste0("^(", paste(c(attrs, "Price", "Ch"), collapse = "|"), ")([1-4])$")
  fixed_names <- names(wide)[!grepl(pattern, names(wide))]
  parts <- lapply(1:4, function(alternative) {
    piece <- wide[, fixed_names, drop = FALSE]
    for (variable in c(attrs, "Price", "Ch")) piece[[variable]] <- wide[[paste0(variable, alternative)]]
    piece$alt <- alternative
    piece$chosen <- as.integer(piece$Ch == 1)
    piece$chid <- paste(wide$Case, wide$Task, sep = "_")
    piece
  })
  output <- do.call(rbind, parts)
  output[order(output$No, output$alt), , drop = FALSE]
}

make_features <- function(df, ctr, scl) {
  df <- as.data.frame(df)
  z <- sweep(sweep(df[scaler_vars], 2, ctr, "-"), 2, scl, "/")
  df$inside <- as.integer(df$alt != 4)
  df$Price_num <- as.numeric(df$Price)
  for (k in 2:12) df[[paste0("Pr_lvl", k)]] <- as.integer(df$Price_num == k)
  df$d2 <- as.integer(df$alt == 2)
  df$d3 <- as.integer(df$alt == 3)
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
  inside_price <- ifelse(df$inside == 1, df$Price_num, NA)
  pmin_by_chid <- tapply(inside_price, df$chid, min, na.rm = TRUE)
  pmax_by_chid <- tapply(inside_price, df$chid, max, na.rm = TRUE)
  df$price_min <- unname(pmin_by_chid[df$chid])
  df$price_max <- unname(pmax_by_chid[df$chid])
  df$price_spread <- df$price_max - df$price_min
  df$is_cheapest <- as.integer(df$inside == 1 & df$Price_num == df$price_min)
  df$is_dearest <- as.integer(df$inside == 1 & df$Price_num == df$price_max)
  df$price_gap_min <- ifelse(df$inside == 1, df$Price_num - df$price_min, 0)
  df$price_gap_max <- ifelse(df$inside == 1, df$price_max - df$Price_num, 0)
  df$price_gap_min_rel <- df$price_gap_min / pmax(df$price_spread, 1)
  df$price_gap_max_rel <- df$price_gap_max / pmax(df$price_spread, 1)
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
base_terms <- c(attr_terms, price_terms, "d2", "d3", int_terms)
fml_relative <- as.formula(paste(
  "chosen ~", paste(c(base_terms, "price_gap_min_rel", "price_gap_max_rel"), collapse = " + "), "| 0"
))

log_loss_matrix <- function(truth, prediction) {
  prediction <- pmax(prediction, 1e-15)
  prediction <- prediction / rowSums(prediction)
  -mean(rowSums(truth * log(prediction)))
}
row_log_loss <- function(truth, prediction) {
  prediction <- pmax(prediction, 1e-15)
  prediction <- prediction / rowSums(prediction)
  -rowSums(truth * log(prediction))
}

align_prediction <- function(raw_prediction, long, wide) {
  chid_map <- unique(long[, c("chid", "No")])
  prediction <- raw_prediction[match(chid_map$chid, rownames(raw_prediction)), , drop = FALSE]
  prediction[match(wide$No, chid_map$No), , drop = FALSE]
}

train <- read.csv(file.path("csv files", "train.csv"))
train <- train[order(train$No), , drop = FALSE]
rownames(train) <- NULL
truth_all <- as.matrix(train[, paste0("Ch", 1:4), drop = FALSE])
stopifnot(nrow(train) == 21565L, all(rowSums(truth_all) == 1L))

saved <- readRDS(file.path("data_processed", "oof_ensemble_v10.rds"))
fold_of_case <- saved$fold_of_case
row_fold <- unname(fold_of_case[as.character(train$Case)])
stopifnot(!anyNA(row_fold), identical(sort(unique(row_fold)), 1:5))

oof_mlogit_baseline <- saved$oof_mlogit # the exact existing m8trpg-family OOF cache
oof_relative <- matrix(NA_real_, nrow(train), 4L)

for (fold in 1:5) {
  cat(sprintf("Fold %d/5...\n", fold))
  train_rows <- which(row_fold != fold)
  val_rows <- which(row_fold == fold)
  train_wide <- train[train_rows, , drop = FALSE]
  val_wide <- train[val_rows, , drop = FALSE]
  stopifnot(length(intersect(unique(train_wide$Case), unique(val_wide$Case))) == 0L)

  train_long <- to_long(train_wide)
  val_long <- to_long(val_wide)
  ctr <- sapply(train_long[scaler_vars], mean, na.rm = TRUE)
  scl <- sapply(train_long[scaler_vars], sd, na.rm = TRUE)
  scl[scl == 0] <- 1
  train_feat <- make_features(train_long, ctr, scl)
  val_feat <- make_features(val_long, ctr, scl)

  mdat_tr <- dfidx(train_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
  mdat_val <- dfidx(val_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
  model <- mlogit(fml_relative, data = mdat_tr)
  prediction <- align_prediction(predict(model, newdata = mdat_val), val_long, val_wide)
  oof_relative[val_rows, ] <- prediction
  cat(sprintf("  fold %d val logloss: %.6f\n", fold, log_loss_matrix(truth_all[val_rows, , drop = FALSE], prediction)))
}
stopifnot(!anyNA(oof_relative))

baseline_loss <- log_loss_matrix(truth_all, oof_mlogit_baseline)
candidate_loss <- log_loss_matrix(truth_all, oof_relative)
cat(sprintf("\nPooled CV: baseline (m8trpg) %.6f, +relative price gap %.6f, gain %.6f\n",
            baseline_loss, candidate_loss, baseline_loss - candidate_loss))

## respondent-clustered bootstrap
row_gain <- row_log_loss(truth_all, oof_mlogit_baseline) - row_log_loss(truth_all, oof_relative)
case_gain <- tapply(row_gain, train$Case, mean)
case_gain <- case_gain[order(as.integer(names(case_gain)))]

set.seed(4821)
n_boot <- 100000L
n_case <- length(case_gain)
boot <- numeric(n_boot)
chunk <- 1000L
start <- 1L
while (start <= n_boot) {
  count <- min(chunk, n_boot - start + 1L)
  idx <- matrix(sample.int(n_case, n_case * count, replace = TRUE), n_case, count)
  boot[start:(start + count - 1L)] <- colMeans(matrix(case_gain[idx], n_case, count))
  start <- start + count
}
summary_row <- data.frame(
  experiment = "relative_price_gap_v1", stage = "canonical",
  baseline_logloss = baseline_loss, candidate_logloss = candidate_loss,
  point_gain = mean(case_gain), bootstrap_mean = mean(boot), bootstrap_sd = sd(boot),
  lower_95 = unname(quantile(boot, 0.025)), upper_95 = unname(quantile(boot, 0.975)),
  lower_99 = unname(quantile(boot, 0.005)), upper_99 = unname(quantile(boot, 0.995)),
  win_rate = mean(boot > 0), n_boot = n_boot
)
summary_row$canonical_pass <- summary_row$point_gain > 0 & summary_row$lower_95 > 0
summary_row$near_miss <- summary_row$point_gain > 0 & summary_row$lower_95 <= 0 & summary_row$lower_95 >= -0.00075
cat("\nCanonical result:\n")
print(summary_row, digits = 9)

write.csv(summary_row, file.path(output_dir, "canonical_summary.csv"), row.names = FALSE)
saveRDS(
  list(oof_relative = oof_relative, oof_baseline = oof_mlogit_baseline, fold_of_case = fold_of_case,
       case_gain = case_gain, bootstrap = boot, summary = summary_row),
  file.path(output_dir, "canonical_result.rds")
)
cat("\nDone.\n")
