# Smoke tests (code correctness only, no peeking at any real candidate result):
# 1. quantile_match() behaves as expected on synthetic data (monotonic, correct
#    percentile alignment, sane extrapolation behaviour).
# 2. to_long() reproduces a correct long-format reshape (checked structurally: row
#    counts, chosen sums to 1 per task, alt4 attributes all zero).
# 3. make_features_m8trpg() with recode_fn = NULL, fit on a single train/holdout
#    split, reproduces mlogit coefficients/predictions sane (no NA, rowSums 1).
# This script does not touch data_processed/codex_transductive outputs used for the
# real verdict.

source("R/codex_transductive_common.R")
library(mlogit)

cat("=== Test 1: quantile_match on synthetic data ===\n")
set.seed(1)
source_vals <- rexp(500, rate = 1 / 50000)   # skewed like income
target_vals <- rexp(200, rate = 1 / 90000)   # shifted/skewed like test income

# a value at the source median should map to roughly the target median
src_median <- median(source_vals)
mapped_median <- quantile_match(src_median, source_vals, target_vals)
tgt_median <- median(target_vals)
cat("source median:", src_median, " mapped:", mapped_median, " target median:", tgt_median, "\n")
stopifnot(abs(mapped_median - tgt_median) < 0.15 * tgt_median)

# monotonicity: mapping a sorted grid must be non-decreasing
grid <- sort(source_vals)
mapped_grid <- quantile_match(grid, source_vals, target_vals)
stopifnot(all(diff(mapped_grid) >= -1e-9))
cat("monotonic over sorted source grid: OK (", sum(diff(mapped_grid) < 0), "violations )\n")

# extrapolation: a value far beyond the source max should clamp near target max
far_above <- max(source_vals) * 100
mapped_far <- quantile_match(far_above, source_vals, target_vals)
cat("far-above value mapped to:", mapped_far, " target max:", max(target_vals), "\n")
stopifnot(mapped_far <= max(target_vals) + 1e-6, mapped_far >= quantile(target_vals, 0.9))

# hand-computed tiny example
tiny_source <- c(10, 20, 30, 40)
tiny_target <- c(100, 200, 300, 400)
m <- quantile_match(c(10, 40, 25), tiny_source, tiny_target)
cat("tiny example mapped:", m, "\n")
stopifnot(abs(m[1] - 100) < 60, abs(m[2] - 400) < 60)  # ends map near target ends
stopifnot(m[3] > m[1], m[3] < m[2])  # interior point stays interior and ordered
cat("Test 1 PASSED\n\n")

cat("=== Test 2: to_long structural checks ===\n")
train <- read.csv("csv files/train.csv")
train_long <- to_long(train)
stopifnot(nrow(train_long) == nrow(train) * 4L)
chosen_sums <- tapply(train_long$chosen, train_long$chid, sum)
stopifnot(all(chosen_sums == 1L))
alt4 <- train_long[train_long$alt == 4, ]
stopifnot(all(alt4$Price == 0))
for (a in attrs) stopifnot(all(alt4[[a]] == 0))
cat("to_long: rows =", nrow(train_long), "; every task sums to 1 chosen; alt4 all-zero attrs. PASSED\n\n")

cat("=== Test 3: make_features_m8trpg(recode_fn=NULL) fits and predicts sanely on a subset ===\n")
set.seed(4821)
cases <- unique(train_long$Case)
subset_cases <- sample(cases, 150)
val_cases <- subset_cases[1:30]
tr_cases <- setdiff(subset_cases, val_cases)

scaler_vars <- c("incomea","agea","milesa","nighta","genderind","Urbind","educind")
tr_k <- train_long[train_long$Case %in% tr_cases, ]
va_k <- train_long[train_long$Case %in% val_cases, ]
std_ctr <- sapply(tr_k[scaler_vars], mean, na.rm = TRUE)
std_scl <- sapply(tr_k[scaler_vars], sd, na.rm = TRUE)
std_scl[std_scl == 0] <- 1

tr_feat <- make_features_m8trpg(tr_k, std_ctr, std_scl, recode_fn = NULL)
va_feat <- make_features_m8trpg(va_k, std_ctr, std_scl, recode_fn = NULL)

mdat_tr <- dfidx(tr_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
mdat_va <- dfidx(va_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
mod_k <- mlogit(m8trpg_formula(), data = mdat_tr)
pred_k <- predict(mod_k, newdata = mdat_va)
stopifnot(!anyNA(pred_k), all(abs(rowSums(pred_k) - 1) < 1e-8), all(pred_k > 0))
cat("Subset fit converged; predictions well-formed (rowSums==1, no NA). PASSED\n\n")

cat("=== Test 4: recode_fn actually changes the design (sanity, not a real result) ===\n")
fake_recode <- function(varname, x) x * 0 + 999999  # absurd constant, just to prove it's wired in
tr_feat2 <- make_features_m8trpg(tr_k, std_ctr, std_scl, recode_fn = fake_recode)
stopifnot(!isTRUE(all.equal(tr_feat$P_income, tr_feat2$P_income)))
cat("recode_fn hook confirmed wired into P_income/In_income etc. PASSED\n\n")

cat("ALL SMOKE TESTS PASSED\n")
