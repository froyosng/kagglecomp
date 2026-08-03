# Screen: choice-set-dependent attribute focusing/salience -- a one-
# parameter reweighting of m8trpg's own fitted per-attribute utility
# contributions by their cross-alternative range within each task.
#
# Pre-registration: codex_salience_preregister.md
#
# Run:
#   source("R/codex_salience_screen.R")

suppressPackageStartupMessages({
  library(mlogit)
  library(dfidx)
})

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
  paste0("P_seg", 2:6), paste0("In_seg", 2:6),
  "P_task", "In_task",
  paste0("P_region", 2:5), paste0("In_region", 2:5),
  paste0("P_ppark", 2:5), paste0("In_ppark", 2:5),
  "is_cheapest", "is_dearest", "price_gap_min", "price_gap_max"
)
fml_base <- as.formula(paste("chosen ~", paste(c(attr_terms, price_terms, "d2", "d3", int_terms), collapse = " + "), "| 0"))

log_loss <- function(truth, pred) {
  pred <- pmax(pred, 1e-15)
  pred <- pred / rowSums(pred)
  -mean(rowSums(truth * log(pred)))
}
align_prediction <- function(raw_prediction, long, wide) {
  chid_map <- unique(long[, c("chid", "No")])
  prediction <- raw_prediction[match(chid_map$chid, rownames(raw_prediction)), , drop = FALSE]
  prediction[match(wide$No, chid_map$No), , drop = FALSE]
}

## ---- attribute-level lookup: coefficient contribution for each level of
## each attribute, reference level (0) contributes 0 ----
attribute_max_level <- function(wide) {
  vapply(attrs, function(a) max(unlist(wide[paste0(a, 1:3)])), numeric(1))
}

coefficient_lookup <- function(model_coef, attribute, max_level) {
  values <- numeric(max_level + 1L) # index 1 = level 0 (reference, always 0)
  for (level in seq_len(max_level)) {
    name <- paste0("factor(", attribute, ")", level)
    values[level + 1L] <- if (name %in% names(model_coef)) unname(model_coef[[name]]) else 0
  }
  values
}

## Build, for one attribute, the n_task x 3 matrix of fitted contributions
## for inside alternatives 1, 2, 3.
attribute_contribution_matrix <- function(wide, attribute, lookup) {
  sapply(1:3, function(alt) lookup[wide[[paste0(attribute, alt)]] + 1L])
}

split <- readRDS("data_processed/train_val_split.rds")
fitting <- split$train_wide_tr
validation <- split$train_wide_val
stopifnot(length(intersect(unique(fitting$Case), unique(validation$Case))) == 0L)

train_long <- to_long(fitting)
val_long <- to_long(validation)
truth_matrix <- align_prediction(
  matrix(as.integer(val_long$chosen), ncol = 4, byrow = TRUE,
         dimnames = list(unique(val_long$chid), NULL)),
  val_long, validation
)
stopifnot(all(rowSums(truth_matrix) == 1))

ctr <- sapply(train_long[scaler_vars], mean, na.rm = TRUE)
scl <- sapply(train_long[scaler_vars], sd, na.rm = TRUE)
scl[scl == 0] <- 1
train_feat <- make_features(train_long, ctr, scl)
val_feat <- make_features(val_long, ctr, scl)

mdat_tr <- dfidx(train_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
mdat_val <- dfidx(val_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")

cat("Fitting m8trpg (fold-fitted attribute coefficients)...\n")
model <- mlogit(fml_base, data = mdat_tr)
model_coef <- coef(model)

pred_fitting <- align_prediction(predict(model, newdata = mdat_tr), train_long, fitting)
pred_val <- align_prediction(predict(model, newdata = mdat_val), val_long, validation)
baseline_val_loss <- log_loss(truth_matrix, pred_val)
cat(sprintf("  base m8trpg val logloss: %.6f (should match the earlier screen's 1.159681)\n", baseline_val_loss))

max_level <- attribute_max_level(rbind(fitting[paste0(rep(attrs, each = 3), 1:3)], validation[paste0(rep(attrs, each = 3), 1:3)]))

## Per-attribute contribution matrices (n_task x 3) for fitting and validation
C_fit <- array(0, dim = c(nrow(fitting), 3L, length(attrs)))
C_val <- array(0, dim = c(nrow(validation), 3L, length(attrs)))
for (index in seq_along(attrs)) {
  attribute <- attrs[[index]]
  lookup <- coefficient_lookup(model_coef, attribute, max_level[[attribute]])
  C_fit[, , index] <- attribute_contribution_matrix(fitting, attribute, lookup)
  C_val[, , index] <- attribute_contribution_matrix(validation, attribute, lookup)
}

## d_ntm = range across alts 1-3 for attribute m; standardized per-attribute
## using fitting-task statistics only.
D_fit <- apply(C_fit, c(1, 3), function(v) max(v) - min(v))
D_val <- apply(C_val, c(1, 3), function(v) max(v) - min(v))
d_centre <- colMeans(D_fit)
d_scale <- apply(D_fit, 2, sd)
d_scale[d_scale == 0] <- 1
Z_fit <- sweep(sweep(D_fit, 2, d_centre, "-"), 2, d_scale, "/")
Z_val <- sweep(sweep(D_val, 2, d_centre, "-"), 2, d_scale, "/")

compute_prediction <- function(base_pred, C, Z, lambda) {
  n <- nrow(Z)
  w_raw <- exp(lambda * Z) # n x 19
  w <- w_raw / rowMeans(w_raw) # sum_m w = 19 exactly
  correction <- matrix(0, n, 4L)
  for (index in seq_along(attrs)) {
    correction[, 1:3] <- correction[, 1:3] + (w[, index] - 1) * C[, , index]
  }
  u_original <- log(pmax(base_pred, 1e-15))
  u_corrected <- u_original + correction
  u_corrected <- u_corrected - apply(u_corrected, 1, max)
  ez <- exp(u_corrected)
  ez / rowSums(ez)
}

lambda_grid <- seq(0, 5, by = 0.25)
losses <- vapply(lambda_grid, function(lambda) {
  pred <- compute_prediction(pred_val, C_val, Z_val, lambda)
  log_loss(truth_matrix, pred)
}, numeric(1))
cat("\nLambda grid (validation logloss):\n")
print(data.frame(lambda = lambda_grid, val_logloss = losses))
best_index <- which.min(losses)
cat(sprintf(
  "\nBest lambda: %.2f, val logloss %.6f, gain vs lambda=0: %.6f\n",
  lambda_grid[[best_index]], losses[[best_index]], losses[[1]] - losses[[best_index]]
))

## also check negative lambda for completeness (falsification: real focusing
## theory predicts lambda > 0; a negative optimum would refute the mechanism)
lambda_grid_neg <- seq(-5, 0, by = 0.25)
losses_neg <- vapply(lambda_grid_neg, function(lambda) {
  pred <- compute_prediction(pred_val, C_val, Z_val, lambda)
  log_loss(truth_matrix, pred)
}, numeric(1))
cat("\nNegative-lambda grid (validation logloss):\n")
print(data.frame(lambda = lambda_grid_neg, val_logloss = losses_neg))

saveRDS(
  list(model_coef = model_coef, C_fit = C_fit, C_val = C_val, Z_fit = Z_fit, Z_val = Z_val,
       lambda_grid = lambda_grid, losses = losses, lambda_grid_neg = lambda_grid_neg, losses_neg = losses_neg,
       baseline_val_loss = baseline_val_loss, pred_val = pred_val, truth_matrix = truth_matrix),
  file.path("data_processed", "codex_salience_screen_result.rds")
)
