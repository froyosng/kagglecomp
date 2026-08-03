# Targeted test: drop In_seg3/In_seg5 (already non-significant, p=0.76/0.40,
# opt-out-margin segment deviations) given the newly-confirmed segment
# train(~9%)-to-test(~69%) weight amplification for exactly those two
# segments. Evaluated on both the ordinary CV population and the
# test-like-reweighted top-30% population, per the pre-registration.
#
# Pre-registration: codex_segment_shift_preregister.md
#
# Run:
#   source("R/codex_segment_shift_cv.R")

suppressPackageStartupMessages({
  library(mlogit)
  library(dfidx)
})

output_dir <- file.path("data_processed", "codex_segment_shift")
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
  paste0("P_seg", 2:6), "In_seg2", "In_seg4", "In_seg6", # In_seg3, In_seg5 dropped
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
test <- read.csv(file.path("csv files", "test.csv"))
stopifnot(nrow(train) == 21565L, all(rowSums(truth_all) == 1L))

saved <- readRDS(file.path("data_processed", "oof_ensemble_v10.rds"))
fold_of_case <- saved$fold_of_case
row_fold <- unname(fold_of_case[as.character(train$Case)])
stopifnot(!anyNA(row_fold), identical(sort(unique(row_fold)), 1:5))
oof_baseline <- saved$oof_mlogit

## ---- fit the pruned model via the exact same canonical 5-fold partition ----
oof_pruned <- matrix(NA_real_, nrow(train), 4L)
for (fold in 1:5) {
  cat(sprintf("Fold %d/5 (pruned model: In_seg3, In_seg5 dropped)...\n", fold))
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
  prediction <- align_prediction(predict(model, newdata = mdat_val), val_long, val_wide)
  oof_pruned[val_rows, ] <- prediction
  cat(sprintf("  fold %d val logloss: %.6f\n", fold, log_loss_matrix(truth_all[val_rows, , drop = FALSE], prediction)))
}
stopifnot(!anyNA(oof_pruned))

baseline_loss <- log_loss_matrix(truth_all, oof_baseline)
pruned_loss <- log_loss_matrix(truth_all, oof_pruned)
cat(sprintf("\nOrdinary pooled CV: baseline (m8trpg) %.6f, pruned (no In_seg3/In_seg5) %.6f, gain %.6f\n",
            baseline_loss, pruned_loss, baseline_loss - pruned_loss))

## ---- test-like propensity (adversarial-validation classifier, no choice outcome used) ----
domain_covariates <- c(
  "segmentind", "yearind", "milesind", "milesa", "nightind", "nighta",
  "pparkind", "genderind", "ageind", "agea", "educind", "regionind",
  "Urbind", "incomeind", "incomea"
)
train_resp <- train[!duplicated(train$Case), c("Case", domain_covariates), drop = FALSE]
test_resp <- test[!duplicated(test$Case), c("Case", domain_covariates), drop = FALSE]
train_resp$segmentind <- factor(train_resp$segmentind, levels = 1:6)
test_resp$segmentind <- factor(test_resp$segmentind, levels = 1:6)
train_resp$regionind <- factor(train_resp$regionind)
test_resp$regionind <- factor(test_resp$regionind)
train_resp$pparkind <- factor(train_resp$pparkind)
test_resp$pparkind <- factor(test_resp$pparkind)
train_resp$yearind <- factor(train_resp$yearind)
test_resp$yearind <- factor(test_resp$yearind)
train_domain <- train_resp[, domain_covariates, drop = FALSE]
test_domain <- test_resp[, domain_covariates, drop = FALSE]
train_domain$is_test <- 0L
test_domain$is_test <- 1L
domain <- rbind(train_domain, test_domain)
classifier <- glm(is_test ~ ., data = domain, family = binomial())
propensity <- predict(classifier, newdata = train_resp[, domain_covariates, drop = FALSE], type = "response")
propensity <- pmin(pmax(as.numeric(propensity), 1e-8), 1 - 1e-8)
names(propensity) <- as.character(train_resp$Case)

## ---- per-respondent gain, evaluated on (a) all respondents, (b) top 30% test-like ----
case_gain <- tapply(row_log_loss(truth_all, oof_baseline) - row_log_loss(truth_all, oof_pruned), train$Case, mean)
case_gain <- case_gain[order(as.integer(names(case_gain)))]
propensity <- propensity[names(case_gain)]
stopifnot(!anyNA(propensity))

ordering <- order(propensity, decreasing = TRUE)
top_30 <- ordering[seq_len(ceiling(0.30 * length(ordering)))]
top_50 <- ordering[seq_len(ceiling(0.50 * length(ordering)))]

n_boot <- 100000L
summary_all <- cbind(data.frame(population = "all_respondents"), summarize_gain(case_gain, n_boot, 4821))
summary_top50 <- cbind(data.frame(population = "top_50pct_test_like"), summarize_gain(case_gain[top_50], n_boot, 4822))
summary_top30 <- cbind(data.frame(population = "top_30pct_test_like"), summarize_gain(case_gain[top_30], n_boot, 4823))
result <- rbind(summary_all, summary_top50, summary_top30)
cat("\nGain (baseline m8trpg - pruned) by population:\n")
print(result, digits = 6)

## sanity: confirm segment composition of the top-30% test-like slice
top30_cases <- as.integer(names(case_gain)[top_30])
seg_by_case <- train$segmentind[match(top30_cases, train$Case)]
cat("\nSegment composition of the top-30%-test-like training respondents:\n")
print(table(seg_by_case))

write.csv(result, file.path(output_dir, "population_comparison.csv"), row.names = FALSE)
saveRDS(
  list(oof_baseline = oof_baseline, oof_pruned = oof_pruned, case_gain = case_gain,
       propensity = propensity, result = result, top30_segment_composition = table(seg_by_case)),
  file.path(output_dir, "result.rds")
)
cat("\nDone.\n")
