# Repeated-CV escalation for the dispersion-relative price-gap candidate
# (canonical near-miss: gain +0.000139, CI [-0.000110, 0.000385], triggers
# escalation per the pre-registered near-miss band).
#
# Reuses the already-verified per-seed fold_map + m8trpg-alone OOF cache from
# R/codex_repeated_cv.R's output (data_processed/codex_repeat_cv/repeat_seed_
# <seed>.rds), the same artifacts codex_two_head_ensemble_v2.R already
# verified and reused, instead of re-deriving the baseline from scratch.
#
# Pre-registration: codex_relative_price_gap_preregister.md
#
# Run:
#   source("R/codex_relative_price_gap_repeated_cv.R")

suppressPackageStartupMessages({
  library(mlogit)
  library(dfidx)
})

output_dir <- file.path("data_processed", "codex_relative_price_gap")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

additional_seeds <- c(1907L, 2719L, 6151L, 8293L, 104729L)
canonical_seed <- 4821L

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

fit_relative_oof <- function(row_fold) {
  prediction <- matrix(NA_real_, nrow(train), 4L)
  for (fold in 1:5) {
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
    prediction[val_rows, ] <- align_prediction(predict(model, newdata = mdat_val), val_long, val_wide)
  }
  stopifnot(!anyNA(prediction))
  prediction
}

## canonical seed: reuse the already-computed result
canonical_result <- readRDS(file.path(output_dir, "canonical_result.rds"))
case_gain_matrix <- matrix(
  NA_real_, length(canonical_result$case_gain), length(c(canonical_seed, additional_seeds)),
  dimnames = list(names(canonical_result$case_gain), as.character(c(canonical_seed, additional_seeds)))
)
case_gain_matrix[, as.character(canonical_seed)] <- canonical_result$case_gain
repeat_rows <- list(`4821` = data.frame(
  seed = canonical_seed,
  baseline_logloss = log_loss_matrix(truth_all, canonical_result$oof_baseline),
  candidate_logloss = log_loss_matrix(truth_all, canonical_result$oof_relative),
  gain = log_loss_matrix(truth_all, canonical_result$oof_baseline) - log_loss_matrix(truth_all, canonical_result$oof_relative)
))

for (seed in additional_seeds) {
  cat(sprintf("\n######## repeated CV seed %d ########\n", seed))
  seed_name <- as.character(seed)
  repeat_cache <- readRDS(file.path("data_processed", "codex_repeat_cv", sprintf("repeat_seed_%d.rds", seed)))
  fold_map <- repeat_cache$fold_map
  row_fold <- unname(fold_map[as.character(train$Case)])
  stopifnot(!anyNA(row_fold), identical(sort(unique(row_fold)), 1:5))
  baseline_oof <- repeat_cache$components$mlogit
  stopifnot(identical(dim(baseline_oof), c(21565L, 4L)))

  relative_oof <- fit_relative_oof(row_fold)
  baseline_loss <- log_loss_matrix(truth_all, baseline_oof)
  candidate_loss <- log_loss_matrix(truth_all, relative_oof)
  case_gain <- tapply(
    row_log_loss(truth_all, baseline_oof) - row_log_loss(truth_all, relative_oof), train$Case, mean
  )
  case_gain <- case_gain[order(as.integer(names(case_gain)))]
  case_gain_matrix[, seed_name] <- case_gain[rownames(case_gain_matrix)]
  repeat_rows[[seed_name]] <- data.frame(
    seed = seed, baseline_logloss = baseline_loss, candidate_logloss = candidate_loss,
    gain = baseline_loss - candidate_loss
  )
  cat(sprintf("  seed %d: baseline %.6f, candidate %.6f, gain %.6f\n", seed, baseline_loss, candidate_loss, baseline_loss - candidate_loss))
  saveRDS(relative_oof, file.path(output_dir, sprintf("relative_oof_seed_%d.rds", seed)))
}

repeated_by_seed <- do.call(rbind, repeat_rows)
average_case_gain <- rowMeans(case_gain_matrix)

set.seed(4821)
n_boot <- 100000L
n_case <- length(average_case_gain)
boot <- numeric(n_boot)
chunk <- 1000L
start <- 1L
while (start <= n_boot) {
  count <- min(chunk, n_boot - start + 1L)
  idx <- matrix(sample.int(n_case, n_case * count, replace = TRUE), n_case, count)
  boot[start:(start + count - 1L)] <- colMeans(matrix(average_case_gain[idx], n_case, count))
  start <- start + count
}
repeated_summary <- data.frame(
  experiment = "relative_price_gap_v1", stage = "repeated_cv",
  point_gain = mean(average_case_gain), bootstrap_mean = mean(boot), bootstrap_sd = sd(boot),
  lower_95 = unname(quantile(boot, 0.025)), upper_95 = unname(quantile(boot, 0.975)),
  lower_99 = unname(quantile(boot, 0.005)), upper_99 = unname(quantile(boot, 0.995)),
  win_rate = mean(boot > 0), n_boot = n_boot,
  positive_repeats = sum(repeated_by_seed$gain > 0), n_repeats = nrow(repeated_by_seed)
)
repeated_summary$promote <- repeated_summary$point_gain > 0 & repeated_summary$lower_95 > 0 & repeated_summary$positive_repeats >= 5L

cat("\nRepeated-CV by seed:\n")
print(repeated_by_seed, digits = 9)
cat("\nRepeated-CV pooled result:\n")
print(repeated_summary, digits = 9)

write.csv(repeated_by_seed, file.path(output_dir, "repeated_cv_by_seed.csv"), row.names = FALSE)
write.csv(repeated_summary, file.path(output_dir, "repeated_cv_summary.csv"), row.names = FALSE)
saveRDS(
  list(repeated_by_seed = repeated_by_seed, repeated_summary = repeated_summary,
       case_gain_matrix = case_gain_matrix, average_case_gain = average_case_gain, bootstrap = boot),
  file.path(output_dir, "repeated_cv_result.rds")
)

verdict <- if (isTRUE(repeated_summary$promote)) {
  "PROMOTE relative price-gap features for a full-data build audit."
} else {
  "REJECT relative price-gap features; retain the exact set-context v14 submission."
}
writeLines(verdict, file.path(output_dir, "verdict.txt"))
cat("\n", verdict, "\n", sep = "")
