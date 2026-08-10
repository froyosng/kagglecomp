# Canonical 5-fold CV for the one semantic-taste pairing that cleanly beat
# its mismatched placebo on the screen: parking situation x Parallel Park
# Aids (PP) fitted contribution. night% x NV failed its placebo outright;
# miles x (CC+LD) beat its placebo only ambiguously (~2x, with the placebo
# alone capturing over half the effect) and is not pursued further.
#
# gamma selected via nested nested nested inner-fold grid search (same
# nested-penalty-selection pattern as codex_alt_link.R / codex_task_
# temperature.R), applied as a residual correction on top of
# segment_shift_v15's own fold-fitted predictions.
#
# Pre-registration: codex_semantic_taste_preregister.md
#
# Run:
#   source("R/codex_semantic_taste_cv.R")

suppressPackageStartupMessages({
  library(mlogit)
  library(dfidx)
})

output_dir <- file.path("data_processed", "codex_semantic_taste")
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
int_terms <- c(
  "P_income", "P_age", "P_miles", "P_night",
  "In_income", "In_age", "In_miles", "In_night",
  "In_gender", "In_urb", "In_educ",
  paste0("P_seg", 2:6), "In_seg2", "In_seg4", "In_seg6",
  "P_task", "In_task",
  paste0("P_region", 2:5), paste0("In_region", 2:5),
  paste0("P_ppark", 2:5), paste0("In_ppark", 2:5),
  "is_cheapest", "is_dearest", "price_gap_min", "price_gap_max"
)
fml_v15 <- as.formula(paste("chosen ~", paste(c(attr_terms, price_terms, "d2", "d3", int_terms), collapse = " + "), "| 0"))

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
attribute_max_level <- function(wide) {
  vapply(attrs, function(a) max(unlist(wide[paste0(a, 1:3)])), numeric(1))
}
coefficient_lookup <- function(model_coef, attribute, max_level) {
  values <- numeric(max_level + 1L)
  for (level in seq_len(max_level)) {
    name <- paste0("factor(", attribute, ")", level)
    values[level + 1L] <- if (name %in% names(model_coef)) unname(model_coef[[name]]) else 0
  }
  values
}
attribute_contribution_matrix <- function(wide, attribute, lookup) {
  sapply(1:3, function(alt) lookup[wide[[paste0(attribute, alt)]] + 1L])
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
    point_gain = mean(case_gain), bootstrap_mean = mean(boot), bootstrap_sd = sd(boot),
    lower_95 = unname(quantile(boot, 0.025)), upper_95 = unname(quantile(boot, 0.975)),
    win_rate = mean(boot > 0), n_boot = replicates
  )
}

apply_correction <- function(base_pred, z_ppark, C_PP, gamma) {
  n <- nrow(base_pred)
  correction <- matrix(0, n, 4L)
  correction[, 1:3] <- gamma * z_ppark * C_PP
  u <- log(pmax(base_pred, 1e-15)) + correction
  u <- u - apply(u, 1, max)
  ez <- exp(u)
  ez / rowSums(ez)
}

fit_fold_model_and_features <- function(train_wide, holdout_wide) {
  train_long <- to_long(train_wide)
  holdout_long <- to_long(holdout_wide)
  ctr <- sapply(train_long[scaler_vars], mean, na.rm = TRUE)
  scl <- sapply(train_long[scaler_vars], sd, na.rm = TRUE)
  scl[scl == 0] <- 1
  train_feat <- make_features(train_long, ctr, scl)
  holdout_feat <- make_features(holdout_long, ctr, scl)
  mdat_tr <- dfidx(train_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
  mdat_ho <- dfidx(holdout_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
  model <- mlogit(fml_v15, data = mdat_tr)
  base_pred <- align_prediction(predict(model, newdata = mdat_ho), holdout_long, holdout_wide)
  max_level <- attribute_max_level(rbind(train_wide[paste0(rep(attrs, each = 3), 1:3)], holdout_wide[paste0(rep(attrs, each = 3), 1:3)]))
  lookup_pp <- coefficient_lookup(coef(model), "PP", max_level[["PP"]])
  C_PP <- attribute_contribution_matrix(holdout_wide, "PP", lookup_pp)
  ppark_ctr <- mean(train_wide$pparkind); ppark_scl <- sd(train_wide$pparkind)
  z_ppark <- (holdout_wide$pparkind - ppark_ctr) / ppark_scl
  list(base_pred = base_pred, z_ppark = z_ppark, C_PP = C_PP)
}

gamma_grid <- seq(-1, 1, by = 0.1)

run_nested_outer <- function(train, fold_map, outer_fold) {
  row_fold <- unname(fold_map[as.character(train$Case)])
  outer_rows <- which(row_fold == outer_fold)
  train_rows <- which(row_fold != outer_fold)
  inner_folds <- setdiff(1:5, outer_fold)

  penalty_losses <- numeric(length(gamma_grid))
  for (gamma_index in seq_along(gamma_grid)) {
    gamma <- gamma_grid[[gamma_index]]
    inner_prediction <- matrix(NA_real_, nrow(train), 4L)
    for (inner_fold in inner_folds) {
      fit_rows <- which(row_fold != outer_fold & row_fold != inner_fold)
      validation_rows <- which(row_fold == inner_fold)
      pieces <- fit_fold_model_and_features(train[fit_rows, , drop = FALSE], train[validation_rows, , drop = FALSE])
      inner_prediction[validation_rows, ] <- apply_correction(pieces$base_pred, pieces$z_ppark, pieces$C_PP, gamma)
    }
    tuning_rows <- setdiff(train_rows, which(row_fold == outer_fold))
    penalty_losses[[gamma_index]] <- log_loss_matrix(
      as.matrix(train[tuning_rows, paste0("Ch", 1:4)]), inner_prediction[tuning_rows, , drop = FALSE]
    )
  }
  best_gamma <- gamma_grid[[which.min(penalty_losses)]]
  pieces <- fit_fold_model_and_features(train[train_rows, , drop = FALSE], train[outer_rows, , drop = FALSE])
  prediction <- apply_correction(pieces$base_pred, pieces$z_ppark, pieces$C_PP, best_gamma)
  list(outer_rows = outer_rows, prediction = prediction, best_gamma = best_gamma, baseline = pieces$base_pred)
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

oof_candidate <- matrix(NA_real_, nrow(train), 4L)
oof_baseline <- matrix(NA_real_, nrow(train), 4L)
gammas <- numeric(5L)
for (fold in 1:5) {
  cat(sprintf("Outer fold %d/5...\n", fold))
  result <- run_nested_outer(train, fold_of_case, fold)
  oof_candidate[result$outer_rows, ] <- result$prediction
  oof_baseline[result$outer_rows, ] <- result$baseline
  gammas[[fold]] <- result$best_gamma
  cat(sprintf("  selected gamma = %.2f, fold logloss (baseline %.6f, candidate %.6f)\n",
              result$best_gamma,
              log_loss_matrix(truth_all[result$outer_rows, , drop = FALSE], result$baseline),
              log_loss_matrix(truth_all[result$outer_rows, , drop = FALSE], result$prediction)))
}
stopifnot(!anyNA(oof_candidate), !anyNA(oof_baseline))

baseline_loss <- log_loss_matrix(truth_all, oof_baseline)
candidate_loss <- log_loss_matrix(truth_all, oof_candidate)
cat(sprintf("\nPooled canonical CV: baseline (segment_shift_v15 mlogit) %.6f, +parking x PP %.6f, gain %.6f\n",
            baseline_loss, candidate_loss, baseline_loss - candidate_loss))
cat("Selected gammas per fold:", paste(gammas, collapse = ", "), "\n")

case_gain <- tapply(row_log_loss(truth_all, oof_baseline) - row_log_loss(truth_all, oof_candidate), train$Case, mean)
summary_row <- summarize_gain(case_gain, 100000L, 4821)
summary_row$canonical_pass <- summary_row$point_gain > 0 & summary_row$lower_95 > 0
summary_row$near_miss <- summary_row$point_gain > 0 & summary_row$lower_95 <= 0 & summary_row$lower_95 >= -0.00075
cat("\nCanonical result:\n")
print(summary_row, digits = 9)

write.csv(summary_row, file.path(output_dir, "canonical_summary.csv"), row.names = FALSE)
saveRDS(
  list(oof_baseline = oof_baseline, oof_candidate = oof_candidate, gammas = gammas,
       case_gain = case_gain, summary = summary_row),
  file.path(output_dir, "canonical_result.rds")
)
