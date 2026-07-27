# Generate (but do not submit) the CV-confirmed MLP blend candidate.
#
# The current public-scored ensemble_v11 CSV is used as the exact baseline,
# avoiding any accidental change to its mlogit/xgboost random state. The MLP
# is fit on all training respondents with the frozen CV specification and five
# seeds, then receives the CV-selected 15% blend weight.

old_stage <- Sys.getenv("CODEX_STAGE", unset = NA_character_)
Sys.setenv(CODEX_STAGE = "define")
source("R/codex_mlp_ensemble.R")
if (is.na(old_stage)) {
  Sys.unsetenv("CODEX_STAGE")
} else {
  Sys.setenv(CODEX_STAGE = old_stage)
}

output_dir <- "data_processed/codex_behavioral_round"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

train <- read.csv("csv files/train.csv")
test <- read.csv("csv files/test.csv")
truth <- as.matrix(train[, paste0("Ch", 1:4)])
reference <- feature_reference(train)
scaler <- continuous_scaler(train)

for (index in seq_along(respondent_factor_vars)) {
  variable <- respondent_factor_vars[[index]]
  stopifnot(all(
    test[[variable]] %in% reference$factor_levels[[index]]
  ))
}

train_matrix <- build_mlp_matrix(train, reference, scaler)
test_matrix <- build_mlp_matrix(
  test, reference, scaler,
  keep_columns = train_matrix$keep_columns
)
stopifnot(
  ncol(train_matrix$x) >= 300L,
  identical(colnames(train_matrix$x), colnames(test_matrix$x)),
  all(is.finite(train_matrix$x)),
  all(is.finite(test_matrix$x))
)

fitted <- fit_mlp_average(
  train_matrix$x,
  truth,
  test_matrix$x,
  size = 8L,
  decay = 0.1,
  seeds = 4821L + 0:4,
  max_iterations = 200L
)
pred_mlp <- fitted$pred
stopifnot(
  !anyNA(pred_mlp),
  max(abs(rowSums(pred_mlp) - 1)) < 1e-10,
  all(pred_mlp > 0)
)

v11 <- read.csv("submission_ensemble_v11_pricegap.csv")
sample_submission <- read.csv("csv files/sample_submission.csv")
probability_columns <- paste0("Ch", 1:4)
stopifnot(
  identical(names(v11), names(sample_submission)),
  identical(v11$No, test$No),
  identical(v11$No, sample_submission$No),
  !anyNA(v11),
  max(abs(rowSums(v11[, probability_columns]) - 1)) < 1e-8
)

mlp_weight <- 0.15
candidate_probability <-
  (1 - mlp_weight) * as.matrix(v11[, probability_columns]) +
  mlp_weight * pred_mlp
candidate_probability <-
  candidate_probability / rowSums(candidate_probability)

submission <- data.frame(
  No = test$No,
  Ch1 = candidate_probability[, 1],
  Ch2 = candidate_probability[, 2],
  Ch3 = candidate_probability[, 3],
  Ch4 = candidate_probability[, 4]
)
stopifnot(
  identical(names(submission), names(sample_submission)),
  identical(submission$No, sample_submission$No),
  !anyNA(submission),
  all(as.matrix(submission[, probability_columns]) > 0),
  max(abs(rowSums(submission[, probability_columns]) - 1)) < 1e-10
)

write.csv(
  submission,
  "submission_codex_mlp_v12_candidate.csv",
  row.names = FALSE
)
saveRDS(
  list(
    specification = list(
      size = 8L,
      decay = 0.1,
      max_iterations = 200L,
      seeds = 4821L + 0:4,
      mlp_weight = mlp_weight
    ),
    fits = fitted$fits,
    models = fitted$models,
    test_prediction = pred_mlp,
    candidate_prediction = candidate_probability,
    feature_columns = train_matrix$keep_columns
  ),
  file.path(output_dir, "mlp_full_test_candidate.rds")
)

cat("Wrote submission_codex_mlp_v12_candidate.csv; not submitted.\n")
print(fitted$fits, digits = 7)
