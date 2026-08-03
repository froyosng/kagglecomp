# Diagnostic (not a candidate model): is v14's per-task/per-respondent loss
# predictable, out-of-sample, from test-time-available covariates and
# choice-set design content? Goes beyond the existing slice-based flatness
# check (cleaning_log.md, 2026-07-26 calibration diagnostic) by fitting an
# honest nested-cross-fitted regularized regression rather than checking a
# handful of pre-chosen slices, and decomposes loss into an opt-out
# component and a conditional-bundle-discrimination component per the
# adversarial-modelling brief's item #4.
#
# Run:
#   source("R/codex_difficulty_diagnostic.R")

suppressPackageStartupMessages({
  library(glmnet)
})

options(stringsAsFactors = FALSE)

output_dir <- file.path("data_processed", "codex_difficulty_diagnostic")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

canonical_seed <- 4821L
v14_dir <- file.path("data_processed", "codex_set_context_network")

validate_probability <- function(prediction, n_rows = NULL) {
  prediction <- as.matrix(prediction)
  if (!is.null(n_rows)) stopifnot(identical(dim(prediction), c(as.integer(n_rows), 4L)))
  prediction <- pmax(prediction, 1e-15)
  prediction / rowSums(prediction)
}

v14 <- readRDS(file.path(v14_dir, "canonical_result.rds"))
fold_map <- v14$fold_map
prediction <- validate_probability(v14$candidate_prediction)

train <- read.csv(file.path("csv files", "train.csv"))
train <- train[order(train$No), , drop = FALSE]
rownames(train) <- NULL
truth <- as.matrix(train[, paste0("Ch", 1:4), drop = FALSE])
stopifnot(nrow(train) == 21565L, all(rowSums(truth) == 1L))

row_fold <- unname(fold_map[as.character(train$Case)])
stopifnot(!anyNA(row_fold))

## ---- decompose row loss into opt-out and conditional-bundle components ----

q_hat <- prediction[, 4L]
chosen_optout <- truth[, 4L] == 1L
optout_loss <- ifelse(chosen_optout, -log(q_hat), -log(1 - q_hat))

inside_mass <- rowSums(prediction[, 1:3, drop = FALSE])
conditional <- prediction[, 1:3, drop = FALSE] / inside_mass
bundle_choice <- max.col(truth[, 1:3, drop = FALSE], ties.method = "first")
bundle_loss <- rep(NA_real_, nrow(train))
inside_rows <- which(!chosen_optout)
bundle_loss[inside_rows] <- -log(conditional[cbind(inside_rows, bundle_choice[inside_rows])])

total_loss <- -rowSums(truth * log(pmax(prediction, 1e-15)))
stopifnot(max(abs(total_loss - (optout_loss))[chosen_optout]) < 1e-8)

## ---- task-difficulty features, computable at test time (no truth used) ----

sorted_conditional <- t(apply(conditional, 1L, function(row) sort(row, decreasing = TRUE)))
gap_top2 <- sorted_conditional[, 1L] - sorted_conditional[, 2L]

attrs <- c(
  "CC", "GN", "NS", "BU", "FA", "LD", "BZ", "FC", "FP", "RP",
  "PP", "KA", "SC", "TS", "NV", "MA", "LB", "AF", "HU"
)
varying <- matrix(FALSE, nrow(train), length(attrs))
for (index in seq_along(attrs)) {
  attribute <- attrs[[index]]
  columns <- as.matrix(train[, paste0(attribute, 1:3), drop = FALSE])
  varying[, index] <- !(columns[, 1L] == columns[, 2L] & columns[, 2L] == columns[, 3L])
}
n_attrs_varying <- rowSums(varying)

price <- as.matrix(train[, paste0("Price", 1:3), drop = FALSE])
price_min <- apply(price, 1L, min)
price_max <- apply(price, 1L, max)
price_mean <- rowMeans(price)
price_cv <- (price_max - price_min) / price_mean
Task_c <- (as.numeric(train$Task) - 10) / 9

respondent_covariates <- c(
  "incomea", "agea", "milesa", "nighta", "genderind", "Urbind", "educind"
)
factor_covariates <- c("segmentind", "regionind", "pparkind", "yearind")

design <- cbind(
  data.frame(
    Task_c = Task_c, q_hat = q_hat, gap_top2 = gap_top2,
    price_cv = price_cv, price_spread = price_max - price_min,
    n_attrs_varying = n_attrs_varying
  ),
  train[, respondent_covariates, drop = FALSE]
)
for (factor_var in factor_covariates) {
  dummies <- model.matrix(as.formula(paste0("~ 0 + factor(", factor_var, ")")), data = train)
  colnames(dummies) <- paste0(factor_var, "_", seq_len(ncol(dummies)))
  design <- cbind(design, dummies)
}
x_full <- as.matrix(design)
storage.mode(x_full) <- "double"

## ---- nested-cross-fitted ridge regression: predict loss out of sample ----

fit_predict_oof <- function(x, y, valid_rows_only = NULL) {
  rows <- if (is.null(valid_rows_only)) seq_len(nrow(x)) else valid_rows_only
  oof <- rep(NA_real_, nrow(x))
  for (outer_fold in 1:5) {
    outer_rows <- intersect(rows, which(row_fold == outer_fold))
    train_rows <- intersect(rows, which(row_fold != outer_fold))
    if (length(outer_rows) == 0L) next
    inner_labels <- setdiff(1:5, outer_fold)
    foldid <- match(row_fold[train_rows], inner_labels)
    fitted <- cv.glmnet(
      x = x[train_rows, , drop = FALSE], y = y[train_rows],
      family = "gaussian", alpha = 0, foldid = foldid, standardize = TRUE
    )
    oof[outer_rows] <- as.numeric(predict(
      fitted, newx = x[outer_rows, , drop = FALSE], s = "lambda.min"
    ))
  }
  oof
}

cat("Fitting nested-cross-fitted predictor for TOTAL row loss...\n")
oof_total <- fit_predict_oof(x_full, total_loss)
cat("Fitting nested-cross-fitted predictor for OPT-OUT-margin loss...\n")
oof_optout <- fit_predict_oof(x_full, optout_loss)
cat("Fitting nested-cross-fitted predictor for CONDITIONAL-BUNDLE loss (inside rows only)...\n")
oof_bundle <- fit_predict_oof(x_full, bundle_loss, valid_rows_only = inside_rows)

r_squared <- function(actual, predicted) {
  ok <- is.finite(actual) & is.finite(predicted)
  1 - sum((actual[ok] - predicted[ok])^2) / sum((actual[ok] - mean(actual[ok]))^2)
}
correlation <- function(actual, predicted) {
  ok <- is.finite(actual) & is.finite(predicted)
  cor(actual[ok], predicted[ok])
}

summary_table <- data.frame(
  target = c("total_loss", "optout_loss", "bundle_loss"),
  n = c(nrow(train), nrow(train), length(inside_rows)),
  oof_r2 = c(
    r_squared(total_loss, oof_total),
    r_squared(optout_loss, oof_optout),
    r_squared(bundle_loss, oof_bundle)
  ),
  oof_correlation = c(
    correlation(total_loss, oof_total),
    correlation(optout_loss, oof_optout),
    correlation(bundle_loss, oof_bundle)
  )
)
cat("\nOut-of-fold predictability of loss from test-time-available features:\n")
print(summary_table, digits = 6)

## ---- respondent-level aggregation and specific slice checks ----

respondent_loss <- tapply(total_loss, train$Case, mean)
respondent_optout_rate <- tapply(chosen_optout, train$Case, mean)
respondent_q <- tapply(q_hat, train$Case, mean)

top_decile <- respondent_loss >= quantile(respondent_loss, 0.90)
cat(sprintf(
  "\nTop-decile-loss respondents (n=%d): mean actual opt-out rate %.3f vs. overall %.3f; mean predicted q %.3f vs. overall %.3f\n",
  sum(top_decile), mean(respondent_optout_rate[top_decile]), mean(respondent_optout_rate),
  mean(respondent_q[top_decile]), mean(respondent_q)
))

## Decile table on the OOF-predicted total-loss score: if predictable, high
## predicted-difficulty deciles should show monotonically higher actual loss.
decile <- cut(oof_total, quantile(oof_total, seq(0, 1, 0.1)), include.lowest = TRUE, labels = FALSE)
decile_table <- data.frame(
  decile = 1:10,
  mean_predicted = tapply(oof_total, decile, mean),
  mean_actual = tapply(total_loss, decile, mean),
  n = tapply(total_loss, decile, length)
)
cat("\nDecile table (predicted difficulty vs. actual loss, out-of-fold):\n")
print(decile_table, digits = 4)

write.csv(summary_table, file.path(output_dir, "predictability_summary.csv"), row.names = FALSE)
write.csv(decile_table, file.path(output_dir, "decile_table.csv"), row.names = FALSE)
saveRDS(
  list(
    oof_total = oof_total, oof_optout = oof_optout, oof_bundle = oof_bundle,
    total_loss = total_loss, optout_loss = optout_loss, bundle_loss = bundle_loss,
    respondent_loss = respondent_loss, respondent_optout_rate = respondent_optout_rate,
    respondent_q = respondent_q, summary_table = summary_table, decile_table = decile_table
  ),
  file.path(output_dir, "diagnostic_result.rds")
)
cat("\nDiagnostic complete. Results in", output_dir, "\n")
