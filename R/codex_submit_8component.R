# Build, but do not submit, the frozen canonical eight-component candidate.
# Weights were fixed before this run in codex_repeated_cv_preregister.md.

source("R/codex_repeat_cv_common.R")

output_dir <- "data_processed/codex_repeat_cv"
deep_checkpoint_dir <- file.path(
  output_dir, "submission_deep_checkpoints"
)
dir.create(
  deep_checkpoint_dir, recursive = TRUE, showWarnings = FALSE
)

train <- read.csv("csv files/train.csv")
test <- read.csv("csv files/test.csv")
sample_submission <- read.csv(
  "csv files/sample_submission.csv"
)
probability_columns <- paste0("Ch", 1:4)
stopifnot(
  identical(test$No, sample_submission$No),
  identical(test$No, sort(test$No)),
  nrow(test) == 4997L
)

load_submission_probability <- function(path) {
  value <- read.csv(path)
  stopifnot(
    identical(value$No, test$No),
    identical(names(value), names(sample_submission))
  )
  repeat_validate_prediction(
    as.matrix(
      value[, probability_columns, drop = FALSE]
    ),
    nrow(test)
  )
}

component_path <- file.path(
  output_dir, "submission_8component_components.rds"
)
components <- if (file.exists(component_path)) {
  readRDS(component_path)
} else {
  list()
}

if (is.null(components$mlogit)) {
  components$mlogit <- load_submission_probability(
    "submission_mlogit_m8trpg_only.csv"
  )
  saveRDS(components, component_path)
  cat("Loaded exact full-data m8trpg component.\n")
}

if (is.null(components$original_xgb)) {
  v11 <- load_submission_probability(
    "submission_ensemble_v11_pricegap.csv"
  )
  components$original_xgb <-
    (v11 - 0.8 * components$mlogit) / 0.2
  components$original_xgb <-
    repeat_validate_prediction(
      components$original_xgb, nrow(test)
    )
  stopifnot(max(abs(
    0.8 * components$mlogit +
      0.2 * components$original_xgb -
      v11
  )) < 1e-12)
  saveRDS(components, component_path)
  cat("Recovered and verified exact v11 xgboost component.\n")
}

if (is.null(components$shallow_mlp)) {
  shallow <- readRDS(
    "data_processed/codex_behavioral_round/mlp_full_test_candidate.rds"
  )
  components$shallow_mlp <- repeat_validate_prediction(
    shallow$test_prediction, nrow(test)
  )
  saveRDS(components, component_path)
  cat("Loaded verified full-data shallow MLP component.\n")
}

test_for_long <- test
test_for_long$Ch1 <- 1L
test_for_long$Ch2 <- 0L
test_for_long$Ch3 <- 0L
test_for_long$Ch4 <- 0L
train_long <- reshape_choice_long(train)
test_long <- reshape_choice_long(test_for_long)

if (is.null(components$triple_mlogit)) {
  fitted <- fit_predict_candidate(
    train_long, test_long,
    "triple_price_income_miles"
  )
  components$triple_mlogit <-
    repeat_validate_prediction(
      fitted$pred[match(test$No, fitted$no), , drop = FALSE],
      nrow(test)
    )
  saveRDS(components, component_path)
  cat(sprintf(
    "Fitted full-data triple mlogit; coefficient %.9f.\n",
    fitted$extra_coef[["P_income_miles"]]
  ))
}

if (is.null(components$rank_ndcg)) {
  model <- repeat_fit_rank(train_long)
  margin <- repeat_rank_margin_matrix(
    model, test_long
  )
  rank_scale <- 1.125
  components$rank_ndcg <- repeat_validate_prediction(
    softmax_margins(
      as.vector(t(margin)), rank_scale
    ),
    nrow(test)
  )
  saveRDS(components, component_path)
  cat("Fitted full-data rank:ndcg component at CV scale 1.125.\n")
}

if (is.null(components$retuned_xgb)) {
  model <- repeat_fit_retuned_xgb(train)
  components$retuned_xgb <-
    repeat_validate_prediction(
      predict_multiclass(model, test), nrow(test)
    )
  saveRDS(components, component_path)
  cat("Fitted full-data retuned xgboost component.\n")
}

if (is.null(components$cox)) {
  attr_max <- as.list(vapply(
    rbind(
      train_long[, attrs, drop = FALSE],
      test_long[, attrs, drop = FALSE]
    ),
    max,
    numeric(1)
  ))
  fitted <- fit_cox_cv(
    train_long, attr_max, seed = 4821L
  )
  components$cox <- repeat_validate_prediction(
    predict_cox_choice(
      fitted, test_long, fitted$cvfit$lambda.min
    ),
    nrow(test)
  )
  saveRDS(components, component_path)
  cat(sprintf(
    "Fitted full-data Cox component; lambda %.9g, %d interactions.\n",
    fitted$cvfit$lambda.min,
    selected_interactions(
      fitted, fitted$cvfit$lambda.min
    )
  ))
}

if (is.null(components$deep_mlp)) {
  reference <- feature_reference(train)
  scaler <- continuous_scaler(train)
  train_matrix <- build_mlp_matrix(
    train, reference, scaler
  )
  test_matrix <- build_mlp_matrix(
    test, reference, scaler,
    keep_columns = train_matrix$keep_columns
  )
  fitted <- fit_torch_average(
    train_matrix$x,
    as.matrix(
      train[, probability_columns, drop = FALSE]
    ),
    test_matrix$x,
    config = repeat_deep_spec,
    seeds = c(9201L, 9202L, 9203L),
    checkpoint_prefix = file.path(
      deep_checkpoint_dir, "full"
    )
  )
  components$deep_mlp <- repeat_validate_prediction(
    fitted$prediction, nrow(test)
  )
  saveRDS(components, component_path)
  cat("Fitted full-data deep MLP component.\n")
}

stopifnot(
  setequal(names(components), repeat_component_names)
)
components <- components[repeat_component_names]
for (name in names(components)) {
  components[[name]] <- repeat_validate_prediction(
    components[[name]], nrow(test)
  )
}

weights <- c(
  mlogit = 0.06246872652458,
  original_xgb = 0.00474541986529,
  rank_ndcg = 0.09330490494295,
  retuned_xgb = 0.01555896281673,
  cox = 0.13111978085328,
  shallow_mlp = 0.08008528400844,
  triple_mlogit = 0.47925772846576,
  deep_mlp = 0.13345919252297
)
stopifnot(
  identical(names(weights), names(components)),
  abs(sum(weights) - 1) < 1e-12
)
candidate <- Reduce(
  `+`,
  Map(
    function(prediction, weight) prediction * weight,
    components, weights
  )
)
candidate <- repeat_validate_prediction(
  candidate, nrow(test)
)
submission <- data.frame(
  No = test$No,
  Ch1 = candidate[, 1],
  Ch2 = candidate[, 2],
  Ch3 = candidate[, 3],
  Ch4 = candidate[, 4]
)
stopifnot(
  identical(names(submission), names(sample_submission)),
  identical(submission$No, sample_submission$No),
  !anyNA(submission),
  all(is.finite(as.matrix(
    submission[, probability_columns]
  ))),
  all(as.matrix(
    submission[, probability_columns]
  ) > 0),
  max(abs(rowSums(
    submission[, probability_columns]
  ) - 1)) < 1e-10
)
write.csv(
  submission,
  "submission_codex_8component_candidate.csv",
  row.names = FALSE
)

current <- load_submission_probability(
  "submission_codex_mlp_v12_candidate.csv"
)
absolute_change <- abs(candidate - current)
row_max_change <- apply(absolute_change, 1, max)
max_row <- which.max(row_max_change)
max_alt <- which.max(absolute_change[max_row, ])
outlier_case <- test$Case[test$No == 22637]
stopifnot(length(outlier_case) == 1L)
outlier_rows <- test$Case == outlier_case

audit_overall <- data.frame(
  comparison = c("all_test", "outlier_case"),
  rows = c(nrow(test), sum(outlier_rows)),
  max_abs_change = c(
    max(row_max_change),
    max(row_max_change[outlier_rows])
  ),
  mean_abs_change = c(
    mean(absolute_change),
    mean(absolute_change[outlier_rows, ])
  ),
  rows_over_005 = c(
    sum(row_max_change > 0.05),
    sum(row_max_change[outlier_rows] > 0.05)
  ),
  rows_over_010 = c(
    sum(row_max_change > 0.10),
    sum(row_max_change[outlier_rows] > 0.10)
  ),
  rows_over_015 = c(
    sum(row_max_change > 0.15),
    sum(row_max_change[outlier_rows] > 0.15)
  ),
  maximum_no = c(
    test$No[[max_row]],
    test$No[outlier_rows][[
      which.max(row_max_change[outlier_rows])
    ]]
  )
)

component_audit <- do.call(rbind, lapply(
  names(components),
  function(name) {
    component_change <- components[[name]] - current
    data.frame(
      component = name,
      weight = weights[[name]],
      component_max_abs_change =
        max(abs(component_change)),
      weighted_max_abs_change =
        max(abs(weights[[name]] * component_change)),
      selected_row_probability =
        components[[name]][max_row, max_alt],
      selected_row_weighted_contribution =
        weights[[name]] * component_change[max_row, max_alt]
    )
  }
))
stopifnot(abs(
  sum(component_audit$selected_row_weighted_contribution) -
    (candidate[max_row, max_alt] - current[max_row, max_alt])
) < 1e-12)

outlier_detail <- data.frame(
  No = test$No[outlier_rows],
  Case = test$Case[outlier_rows],
  Task = test$Task[outlier_rows],
  incomea = test$incomea[outlier_rows],
  max_abs_change = row_max_change[outlier_rows],
  current_max_probability =
    apply(current[outlier_rows, , drop = FALSE], 1, max),
  candidate_max_probability =
    apply(candidate[outlier_rows, , drop = FALSE], 1, max)
)

write.csv(
  audit_overall,
  file.path(output_dir, "submission_8component_outlier_audit.csv"),
  row.names = FALSE
)
write.csv(
  component_audit,
  file.path(output_dir, "submission_8component_component_audit.csv"),
  row.names = FALSE
)
write.csv(
  outlier_detail,
  file.path(output_dir, "submission_8component_outlier_detail.csv"),
  row.names = FALSE
)
saveRDS(
  list(
    weights = weights,
    components = components,
    candidate = candidate,
    current = current,
    audit_overall = audit_overall,
    component_audit = component_audit,
    outlier_detail = outlier_detail,
    maximum_change = list(
      row = max_row,
      No = test$No[[max_row]],
      Case = test$Case[[max_row]],
      Task = test$Task[[max_row]],
      alternative = max_alt,
      current = current[max_row, max_alt],
      candidate = candidate[max_row, max_alt]
    )
  ),
  file.path(output_dir, "submission_8component_audit.rds")
)

cat("Wrote submission_codex_8component_candidate.csv; not submitted.\n")
print(audit_overall, digits = 12, row.names = FALSE)
print(component_audit, digits = 12, row.names = FALSE)
