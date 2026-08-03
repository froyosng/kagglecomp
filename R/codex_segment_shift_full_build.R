# Full-data build for the PROMOTED segment-shift pruning candidate
# (In_seg3, In_seg5 dropped from the m8trpg mlogit component; everything
# else -- xgboost, shallow MLP, set-context network, all blend weights --
# reused EXACTLY as already deployed in v14, not refit, to isolate the one
# real change and avoid introducing any non-reproducibility risk from
# refitting xgboost/torch components).
#
# Never submits to Kaggle. Requires running TWICE: the first run creates a
# reference artifact, the second run must reproduce it to within tolerance
# before writing the candidate CSV -- the same discipline
# codex_set_context_candidate_submission.R already established.
#
# Pre-registration: codex_segment_shift_preregister.md
# CV confirmation: codex_segment_shift_findings.md
#
# Run (twice):
#   source("R/codex_segment_shift_full_build.R")

suppressPackageStartupMessages({
  library(mlogit)
  library(dfidx)
})

output_dir <- file.path("data_processed", "codex_segment_shift_full_build")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
reproduction_tolerance <- 1e-6
probability_columns <- paste0("Ch", 1:4)

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

validate_probability <- function(prediction) {
  prediction <- pmax(as.matrix(prediction), 1e-15)
  prediction / rowSums(prediction)
}
align_prediction <- function(raw_prediction, long, wide) {
  chid_map <- unique(long[, c("chid", "No")])
  prediction <- raw_prediction[match(chid_map$chid, rownames(raw_prediction)), , drop = FALSE]
  prediction[match(wide$No, chid_map$No), , drop = FALSE]
}

## ---- inputs (all already-verified, already-deployed artifacts; nothing
## other than the mlogit component is refit) ----
required_files <- c(
  file.path("csv files", "train.csv"),
  file.path("csv files", "test.csv"),
  file.path("csv files", "sample_submission.csv"),
  "submission_mlogit_m8trpg_only.csv",
  "submission_ensemble_v11_pricegap.csv",
  "submission_set_context_v14_candidate.csv",
  file.path("data_processed", "codex_behavioral_round", "mlp_full_test_candidate.rds"),
  file.path("data_processed", "codex_set_context_full_build", "full_build_latest.rds"),
  file.path("data_processed", "codex_set_context_full_build", "full_build_reproduction_check.csv")
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop("Missing required file(s):\n  ", paste(missing_files, collapse = "\n  "))
}
sc_repro <- read.csv(file.path("data_processed", "codex_set_context_full_build", "full_build_reproduction_check.csv"))
stopifnot(identical(sc_repro$status[[1L]], "REPRODUCED_PASS"), isTRUE(sc_repro$pass[[1L]]))

train <- read.csv(file.path("csv files", "train.csv"))
train <- train[order(train$No), , drop = FALSE]
rownames(train) <- NULL
test <- read.csv(file.path("csv files", "test.csv"))
test <- test[order(test$No), , drop = FALSE]
rownames(test) <- NULL
sample_submission <- read.csv(file.path("csv files", "sample_submission.csv"))
sample_submission <- sample_submission[order(sample_submission$No), , drop = FALSE]
rownames(sample_submission) <- NULL
stopifnot(
  nrow(train) == 21565L, nrow(test) == 4997L,
  length(unique(train$Case)) == 1135L, length(unique(test$Case)) == 263L,
  length(intersect(unique(train$Case), unique(test$Case))) == 0L,
  identical(test$No, sample_submission$No)
)

mlogit_orig_test <- read.csv("submission_mlogit_m8trpg_only.csv")
mlogit_orig_test <- mlogit_orig_test[match(test$No, mlogit_orig_test$No), probability_columns]
ensemble_v11_test <- read.csv("submission_ensemble_v11_pricegap.csv")
ensemble_v11_test <- ensemble_v11_test[match(test$No, ensemble_v11_test$No), probability_columns]
v14_reference_test <- read.csv("submission_set_context_v14_candidate.csv")
v14_reference_test <- v14_reference_test[match(test$No, v14_reference_test$No), probability_columns]

mlp_cache <- readRDS(file.path("data_processed", "codex_behavioral_round", "mlp_full_test_candidate.rds"))
mlp_test <- validate_probability(mlp_cache$test_prediction)

sc_cache <- readRDS(file.path("data_processed", "codex_set_context_full_build", "full_build_latest.rds"))
set_context_test <- validate_probability(sc_cache$set_context_prediction)
stopifnot(identical(as.integer(sc_cache$test_no), as.integer(test$No)))

## back out the standalone xgboost test predictions algebraically from the
## two already-deployed blends (avoids refitting xgboost, whose internal
## RNG draws for subsample/colsample are not seeded in submit_ensemble_v11.R
## and so are not guaranteed bit-reproducible on a fresh fit)
xgb_test <- (as.matrix(ensemble_v11_test) - 0.80 * as.matrix(mlogit_orig_test)) / 0.20
xgb_test <- validate_probability(xgb_test)

## correctness check: reconstructing the ORIGINAL (unpruned) pipeline from
## these pieces must exactly reproduce the deployed v14 submission
reconstructed_original <- validate_probability(
  0.889 * (0.85 * (0.80 * as.matrix(mlogit_orig_test) + 0.20 * xgb_test) + 0.15 * mlp_test) +
    0.111 * set_context_test
)
original_reproduction_diff <- max(abs(reconstructed_original - as.matrix(v14_reference_test)))
cat(sprintf("Original-pipeline reconstruction check: max abs diff vs. deployed v14 = %.3e\n", original_reproduction_diff))
stopifnot(original_reproduction_diff < 1e-6)
cat("PASS: component pieces exactly reconstruct the deployed v14 submission.\n\n")

## ---- refit ONLY the pruned mlogit, on ALL 1135 training respondents ----
train_long <- to_long(train)
test_placeholder <- test
test_placeholder$Ch1 <- 1; test_placeholder$Ch2 <- 0
test_placeholder$Ch3 <- 0; test_placeholder$Ch4 <- 0
test_long <- to_long(test_placeholder)
ctr <- sapply(train_long[scaler_vars], mean, na.rm = TRUE)
scl <- sapply(train_long[scaler_vars], sd, na.rm = TRUE)
scl[scl == 0] <- 1
train_feat <- make_features(train_long, ctr, scl)
test_feat <- make_features(test_long, ctr, scl)
mdat_tr <- dfidx(train_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
mdat_te <- dfidx(test_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
cat("Fitting the pruned mlogit (In_seg3, In_seg5 dropped) on all 1,135 training respondents...\n")
pruned_model <- mlogit(fml_pruned, data = mdat_tr)
pruned_mlogit_test <- align_prediction(predict(pruned_model, newdata = mdat_te), test_long, test)
pruned_mlogit_test <- validate_probability(pruned_mlogit_test)

candidate_prediction <- validate_probability(
  0.889 * (0.85 * (0.80 * pruned_mlogit_test + 0.20 * xgb_test) + 0.15 * mlp_test) +
    0.111 * set_context_test
)

diagnostics <- data.frame(
  mean_absolute_probability_change = mean(abs(candidate_prediction - reconstructed_original)),
  max_probability_change = max(abs(candidate_prediction - reconstructed_original)),
  argmax_flip_rate = mean(max.col(candidate_prediction) != max.col(reconstructed_original)),
  flattened_probability_correlation = cor(as.vector(candidate_prediction), as.vector(reconstructed_original))
)
cat("\nDiagnostics (pruned candidate vs. reconstructed original v14):\n")
print(diagnostics, digits = 6)

## ---- reproducibility gate: this script must be run twice and agree ----
input_hashes <- c(
  train = unname(tools::md5sum(file.path("csv files", "train.csv"))),
  test = unname(tools::md5sum(file.path("csv files", "test.csv"))),
  mlogit_orig = unname(tools::md5sum("submission_mlogit_m8trpg_only.csv")),
  ensemble_v11 = unname(tools::md5sum("submission_ensemble_v11_pricegap.csv")),
  v14_reference = unname(tools::md5sum("submission_set_context_v14_candidate.csv")),
  mlp_cache = unname(tools::md5sum(file.path("data_processed", "codex_behavioral_round", "mlp_full_test_candidate.rds"))),
  set_context_cache = unname(tools::md5sum(file.path("data_processed", "codex_set_context_full_build", "full_build_latest.rds")))
)
build_result <- list(
  input_hashes = input_hashes, test_no = test$No,
  pruned_mlogit_test = pruned_mlogit_test, candidate_prediction = candidate_prediction,
  diagnostics = diagnostics
)

reference_path <- file.path(output_dir, "full_build_reference.rds")
if (file.exists(reference_path)) {
  reference <- readRDS(reference_path)
  stopifnot(
    identical(reference$input_hashes, build_result$input_hashes),
    identical(as.integer(reference$test_no), as.integer(build_result$test_no))
  )
  candidate_difference <- max(abs(reference$candidate_prediction - build_result$candidate_prediction))
  reproduction_pass <- candidate_difference <= reproduction_tolerance
  reproduction_status <- if (reproduction_pass) "REPRODUCED_PASS" else "REPRODUCED_FAIL"
  cat(sprintf("\nTwo-run reproducibility check: max abs diff = %.3e, status = %s\n", candidate_difference, reproduction_status))
  if (!reproduction_pass) {
    stop("Full-data reproduction check failed. No candidate CSV was written.")
  }

  submission <- data.frame(
    No = test$No,
    Ch1 = candidate_prediction[, 1L], Ch2 = candidate_prediction[, 2L],
    Ch3 = candidate_prediction[, 3L], Ch4 = candidate_prediction[, 4L]
  )
  stopifnot(
    identical(names(submission), names(sample_submission)),
    identical(submission$No, sample_submission$No),
    !anyNA(submission),
    max(abs(rowSums(submission[, probability_columns]) - 1)) < 1e-10
  )
  candidate_path <- "submission_segment_shift_v15_candidate.csv"
  write.csv(submission, candidate_path, row.names = FALSE)
  candidate_md5 <- unname(tools::md5sum(candidate_path))
  write.csv(
    cbind(data.frame(status = reproduction_status, candidate_path = candidate_path, candidate_md5 = candidate_md5), diagnostics),
    file.path(output_dir, "full_build_summary.csv"),
    row.names = FALSE
  )
  cat("\nCandidate written:", candidate_path, "\n")
  cat("Candidate MD5:", candidate_md5, "\n")
} else {
  saveRDS(build_result, reference_path)
  cat("\nReference artifact created. Run this script again to perform the reproducibility check and write the candidate CSV.\n")
}
saveRDS(build_result, file.path(output_dir, "full_build_latest.rds"))
