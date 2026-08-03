# Repeated-CV confirmation for the segment-shift targeted pruning candidate
# (canonical result: whole-population gain +0.000504, CI excludes zero;
# top-30%-test-like gain +0.001777, CI excludes zero more strongly).
#
# Reuses the already-verified per-seed fold_map + m8trpg-alone OOF cache
# from data_processed/codex_repeat_cv/repeat_seed_<seed>.rds, the same
# artifacts this session's other repeated-CV scripts already used.
#
# Pre-registration: codex_segment_shift_preregister.md
#
# Run:
#   source("R/codex_segment_shift_repeated_cv.R")

suppressPackageStartupMessages({
  library(mlogit)
  library(dfidx)
})

output_dir <- file.path("data_processed", "codex_segment_shift")
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
  df$is_cheapest <- as.integer(df$inside == 1 & df$Price_num == df$price_min)
  df$is_dearest <- as.integer(df$inside == 1 & df$Price_num == df$price_max)
  df$price_gap_min <- ifelse(df$inside == 1, df$Price_num - df$price_min, 0)
  df$price_gap_max <- ifelse(df$inside == 1, df$price_max - df$Price_num, 0)
  df
}

attr_terms <- paste0("factor(", attrs, ")")
price_terms <- paste0("Pr_lvl", 2:12)
int_terms_pruned <- c(
  "P_income", "P_age", "P_miles", "P_night",
  "In_income", "In_age", "In_miles", "In_night",
  "In_gender", "In_urb", "In_educ",
  paste0("P_seg", 2:6), "In_seg2", "In_seg4", "In_seg6",
  "P_task", "In_task",
  paste0("P_region", 2:5), paste0("In_region", 2:5),
  paste0("P_ppark", 2:5), paste0("In_ppark", 2:5),
  "is_cheapest", "is_dearest", "price_gap_min", "price_gap_max"
)
fml_pruned <- as.formula(paste(
  "chosen ~", paste(c(attr_terms, price_terms, "d2", "d3", int_terms_pruned), collapse = " + "), "| 0"
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
bootstrap_case_means <- function(case_gain, replicates, seed, chunk_size = 1000L) {
  case_gain <- as.numeric(case_gain)
  set.seed(seed)
  n_case <- length(case_gain)
  output <- numeric(replicates)
  start <- 1L
  while (start <= replicates) {
    count <- min(chunk_size, replicates - start + 1L)
    idx <- matrix(sample.int(n_case, n_case * count, replace = TRUE), n_case, count)
    output[start:(start + count - 1L)] <- colMeans(matrix(case_gain[idx], n_case, count))
    start <- start + count
  }
  output
}
summarize_gain <- function(case_gain, replicates, seed) {
  boot <- bootstrap_case_means(case_gain, replicates, seed)
  data.frame(
    n_respondents = length(case_gain), point_gain = mean(case_gain), bootstrap_mean = mean(boot),
    bootstrap_sd = sd(boot), lower_95 = unname(quantile(boot, 0.025)), upper_95 = unname(quantile(boot, 0.975)),
    win_rate = mean(boot > 0), n_boot = replicates
  )
}

train <- read.csv(file.path("csv files", "train.csv"))
train <- train[order(train$No), , drop = FALSE]
rownames(train) <- NULL
truth_all <- as.matrix(train[, paste0("Ch", 1:4), drop = FALSE])
stopifnot(nrow(train) == 21565L, all(rowSums(truth_all) == 1L))

fit_pruned_oof <- function(row_fold) {
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
    model <- mlogit(fml_pruned, data = mdat_tr)
    prediction[val_rows, ] <- align_prediction(predict(model, newdata = mdat_val), val_long, val_wide)
  }
  stopifnot(!anyNA(prediction))
  prediction
}

## canonical seed: reuse the already-computed result
canonical <- readRDS(file.path(output_dir, "result.rds"))
propensity <- canonical$propensity # fixed, does not depend on CV fold structure

case_levels <- sort(unique(train$Case))
seed_names <- as.character(c(canonical_seed, additional_seeds))
case_gain_matrix <- matrix(
  NA_real_, length(case_levels), length(seed_names),
  dimnames = list(as.character(case_levels), seed_names)
)
case_gain_matrix[, as.character(canonical_seed)] <- canonical$case_gain[as.character(case_levels)]
repeat_rows <- list(`4821` = data.frame(
  seed = canonical_seed,
  baseline_logloss = log_loss_matrix(truth_all, canonical$oof_baseline),
  candidate_logloss = log_loss_matrix(truth_all, canonical$oof_pruned),
  gain = log_loss_matrix(truth_all, canonical$oof_baseline) - log_loss_matrix(truth_all, canonical$oof_pruned)
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

  pruned_oof <- fit_pruned_oof(row_fold)
  baseline_loss <- log_loss_matrix(truth_all, baseline_oof)
  candidate_loss <- log_loss_matrix(truth_all, pruned_oof)
  case_gain <- tapply(row_log_loss(truth_all, baseline_oof) - row_log_loss(truth_all, pruned_oof), train$Case, mean)
  case_gain <- case_gain[order(as.integer(names(case_gain)))]
  case_gain_matrix[, seed_name] <- case_gain[rownames(case_gain_matrix)]
  repeat_rows[[seed_name]] <- data.frame(
    seed = seed, baseline_logloss = baseline_loss, candidate_logloss = candidate_loss,
    gain = baseline_loss - candidate_loss
  )
  cat(sprintf("  seed %d: baseline %.6f, candidate %.6f, gain %.6f\n", seed, baseline_loss, candidate_loss, baseline_loss - candidate_loss))
  saveRDS(pruned_oof, file.path(output_dir, sprintf("pruned_oof_seed_%d.rds", seed)))
}

repeated_by_seed <- do.call(rbind, repeat_rows)
average_case_gain <- rowMeans(case_gain_matrix)

ordering <- order(propensity[rownames(case_gain_matrix)], decreasing = TRUE)
top_30 <- ordering[seq_len(ceiling(0.30 * length(ordering)))]
top_50 <- ordering[seq_len(ceiling(0.50 * length(ordering)))]

n_boot <- 100000L
pooled_all <- cbind(data.frame(population = "all_respondents"), summarize_gain(average_case_gain, n_boot, 4821))
pooled_top50 <- cbind(data.frame(population = "top_50pct_test_like"), summarize_gain(average_case_gain[top_50], n_boot, 4822))
pooled_top30 <- cbind(data.frame(population = "top_30pct_test_like"), summarize_gain(average_case_gain[top_30], n_boot, 4823))
pooled_result <- rbind(pooled_all, pooled_top50, pooled_top30)
pooled_result$positive_repeats <- sum(repeated_by_seed$gain > 0)
pooled_result$n_repeats <- nrow(repeated_by_seed)
pooled_result$promote <- pooled_result$point_gain > 0 & pooled_result$lower_95 > 0 & pooled_result$positive_repeats >= 5L

cat("\nRepeated-CV by seed:\n")
print(repeated_by_seed, digits = 9)
cat("\nRepeated-CV pooled result by population:\n")
print(pooled_result, digits = 6)

write.csv(repeated_by_seed, file.path(output_dir, "repeated_cv_by_seed.csv"), row.names = FALSE)
write.csv(pooled_result, file.path(output_dir, "repeated_cv_population_comparison.csv"), row.names = FALSE)
saveRDS(
  list(repeated_by_seed = repeated_by_seed, pooled_result = pooled_result,
       case_gain_matrix = case_gain_matrix, average_case_gain = average_case_gain),
  file.path(output_dir, "repeated_cv_result.rds")
)

verdict <- if (isTRUE(pooled_result$promote[pooled_result$population == "all_respondents"])) {
  "PROMOTE segment-shift pruning (In_seg3, In_seg5 dropped) for a full-data build audit."
} else {
  "REJECT segment-shift pruning on the whole population despite canonical pass; see per-population detail."
}
writeLines(verdict, file.path(output_dir, "verdict.txt"))
cat("\n", verdict, "\n", sep = "")
