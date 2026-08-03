# Screen: partial-profile alignability weighting -- a one-parameter
# reweighting of m8trpg's own fitted per-attribute utility contributions
# according to whether each attribute is uniquely displayed (shown by only
# 1 of 3 inside alternatives) vs. alignable (shown by 2 or 3), using only
# the already-established "level 0 = not shown" structural fact (no
# dependence on the external source's claimed code-to-feature mapping).
#
# Pre-registration: codex_alignability_preregister.md
#
# Run:
#   source("R/codex_alignability_screen.R")

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

## note: this screen uses the segment_shift_v15 formula (In_seg3/In_seg5
## already dropped) as its base, since that is now the standing best.
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
## k_ntm: how many of the 3 inside alternatives display attribute m (level != 0)
attribute_shown_count <- function(wide, attribute) {
  levels_mat <- sapply(1:3, function(alt) wide[[paste0(attribute, alt)]])
  rowSums(levels_mat != 0)
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

cat("Fitting segment_shift_v15 formula (fold-fitted coefficients)...\n")
model <- mlogit(fml_v15, data = mdat_tr)
model_coef <- coef(model)
pred_val <- align_prediction(predict(model, newdata = mdat_val), val_long, validation)
baseline_val_loss <- log_loss(truth_matrix, pred_val)
cat(sprintf("  base val logloss: %.6f\n", baseline_val_loss))

max_level <- attribute_max_level(rbind(fitting[paste0(rep(attrs, each = 3), 1:3)], validation[paste0(rep(attrs, each = 3), 1:3)]))

## per-attribute contribution matrices and shown-counts (validation set,
## since gamma is applied at prediction time as a residual correction)
C_val <- array(0, dim = c(nrow(validation), 3L, length(attrs)))
unique_val <- matrix(0, nrow(validation), length(attrs))
for (index in seq_along(attrs)) {
  attribute <- attrs[[index]]
  lookup <- coefficient_lookup(model_coef, attribute, max_level[[attribute]])
  C_val[, , index] <- attribute_contribution_matrix(validation, attribute, lookup)
  k <- attribute_shown_count(validation, attribute)
  unique_val[, index] <- as.numeric(k == 1)
}

n_unique_per_task <- rowSums(unique_val)
cat("\nDistribution of # uniquely-displayed attributes per task (validation):\n")
print(table(n_unique_per_task))

compute_prediction <- function(base_pred, C, unique_mask, gamma) {
  n <- dim(C)[1]
  correction <- matrix(0, n, 4L)
  for (index in seq_along(attrs)) {
    correction[, 1:3] <- correction[, 1:3] + gamma * unique_mask[, index] * C[, , index]
  }
  u_original <- log(pmax(base_pred, 1e-15))
  u_corrected <- u_original + correction
  u_corrected <- u_corrected - apply(u_corrected, 1, max)
  ez <- exp(u_corrected)
  ez / rowSums(ez)
}

gamma_grid <- seq(-2, 2, by = 0.1)
losses <- vapply(gamma_grid, function(gamma) {
  pred <- compute_prediction(pred_val, C_val, unique_val, gamma)
  log_loss(truth_matrix, pred)
}, numeric(1))
result_table <- data.frame(gamma = gamma_grid, val_logloss = losses)
cat("\nGamma grid (validation logloss):\n")
print(result_table)
best_index <- which.min(losses)
cat(sprintf(
  "\nBest gamma: %.2f, val logloss %.6f, gain vs gamma=0: %.6f\n",
  gamma_grid[[best_index]], losses[[best_index]], losses[gamma_grid == 0] - losses[[best_index]]
))

saveRDS(
  list(model_coef = model_coef, gamma_grid = gamma_grid, losses = losses, baseline_val_loss = baseline_val_loss),
  file.path("data_processed", "codex_alignability_screen_result.rds")
)
