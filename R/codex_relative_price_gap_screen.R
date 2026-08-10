# Screen: dispersion-relative price-gap features (price_gap_min/max divided
# by the task's own price spread) added to the exact m8trpg formula.
#
# Pre-registration: codex_relative_price_gap_preregister.md
#
# Run:
#   source("R/codex_relative_price_gap_screen.R")

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
  df$price_spread <- df$price_max - df$price_min
  df$is_cheapest <- as.integer(df$inside == 1 & df$Price_num == df$price_min)
  df$is_dearest <- as.integer(df$inside == 1 & df$Price_num == df$price_max)
  df$price_gap_min <- ifelse(df$inside == 1, df$Price_num - df$price_min, 0)
  df$price_gap_max <- ifelse(df$inside == 1, df$price_max - df$Price_num, 0)
  # New this experiment: the same gaps, rescaled by the task's own price
  # spread (floored at 1 to avoid 0/0 on the rare all-tied-price task, where
  # price_gap_min/max are already exactly 0 so the floor changes nothing).
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
fml_base <- as.formula(paste("chosen ~", paste(base_terms, collapse = " + "), "| 0"))
fml_relative <- as.formula(paste(
  "chosen ~", paste(c(base_terms, "price_gap_min_rel", "price_gap_max_rel"), collapse = " + "), "| 0"
))

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

cat("price_gap_min_rel summary (training):\n")
print(summary(train_feat$price_gap_min_rel[train_feat$inside == 1]))
cat("price_gap_max_rel summary (training):\n")
print(summary(train_feat$price_gap_max_rel[train_feat$inside == 1]))

mdat_tr <- dfidx(train_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
mdat_val <- dfidx(val_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")

cat("\nFitting base m8trpg...\n")
model_base <- mlogit(fml_base, data = mdat_tr)
pred_base <- align_prediction(predict(model_base, newdata = mdat_val), val_long, validation)
loss_base <- log_loss(truth_matrix, pred_base)
cat(sprintf("  base m8trpg val logloss: %.6f\n", loss_base))

cat("Fitting m8trpg + price_gap_min_rel + price_gap_max_rel...\n")
model_rel <- mlogit(fml_relative, data = mdat_tr)
pred_rel <- align_prediction(predict(model_rel, newdata = mdat_val), val_long, validation)
loss_rel <- log_loss(truth_matrix, pred_rel)
cat(sprintf("  +relative price gap val logloss: %.6f\n", loss_rel))

cat(sprintf("\nScreen gain (base - relative): %.6f\n", loss_base - loss_rel))
print(summary(model_rel)$CoefTable[c("price_gap_min", "price_gap_max", "price_gap_min_rel", "price_gap_max_rel"), ])

saveRDS(
  list(model_base = model_base, model_rel = model_rel, loss_base = loss_base, loss_rel = loss_rel),
  file.path("data_processed", "codex_relative_price_gap_screen_result.rds")
)
