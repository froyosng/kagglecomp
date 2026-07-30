# Build and audit the promoted set-context full-data candidate.
#
# This script never submits to Kaggle. It reuses the exact model definitions
# from the smoke-tested/CV-tested v2 runner, trains three full-data networks,
# and blends them into the exact public-scored shallow-MLP submission.
#
# Smoke test:
#   Sys.setenv(SET_CONTEXT_BUILD_SMOKE = "1")
#   source("R/codex_set_context_candidate_submission.R")
#   Sys.unsetenv("SET_CONTEXT_BUILD_SMOKE")
#
# Full build (run twice; the second run performs the reproduction check):
#   Sys.unsetenv("SET_CONTEXT_BUILD_SMOKE")
#   source("R/codex_set_context_candidate_submission.R")

options(stringsAsFactors = FALSE)

build_smoke <- identical(
  Sys.getenv("SET_CONTEXT_BUILD_SMOKE", "0"),
  "1"
)
default_runner_path <- if (
  file.exists("R/codex_set_context_network_v2.R")
) {
  "R/codex_set_context_network_v2.R"
} else {
  "R/codex_set_context_network.R"
}
runner_path <- Sys.getenv(
  "SET_CONTEXT_RUNNER",
  default_runner_path
)
baseline_path <- Sys.getenv(
  "SET_CONTEXT_BASELINE_SUBMISSION",
  "submission_codex_mlp_v12_candidate.csv"
)
output_dir <- file.path(
  "data_processed",
  "codex_set_context_full_build"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

expected_runner_md5 <- "05acc3a35472689b65c185ba869954d3"
expected_baseline_md5 <- "569e02efb106890fb111463f7c7b7da6"
expected_experiment <- "set_context_utility_network_v1"
full_network_seeds <- 12501:12503
expected_full_weight <- 0.111
reproduction_tolerance <- 1e-6
probability_columns <- paste0("Ch", 1:4)

required_files <- c(
  runner_path,
  file.path("csv files", "train.csv"),
  file.path("csv files", "test.csv"),
  file.path("csv files", "sample_submission.csv"),
  file.path("data_processed", "oof_ensemble_v10.rds"),
  file.path(
    "data_processed",
    "codex_set_context_network",
    "repeated_cv_summary.csv"
  ),
  file.path(
    "data_processed",
    "codex_set_context_network",
    "repeated_cv_fold_selection.csv"
  ),
  file.path(
    "data_processed",
    "codex_set_context_network",
    "verdict.txt"
  ),
  baseline_path
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing required project file(s):\n  ",
    paste(missing_files, collapse = "\n  "),
    "\nRun this script from the kagglecomp repository root."
  )
}

runner_md5 <- unname(tools::md5sum(runner_path))
if (!identical(runner_md5, expected_runner_md5)) {
  stop(
    "The set-context v2 runner does not match the validated copy.\n",
    "Expected MD5: ", expected_runner_md5, "\n",
    "Observed MD5: ", runner_md5
  )
}
baseline_md5 <- unname(tools::md5sum(baseline_path))
if (!identical(baseline_md5, expected_baseline_md5)) {
  stop(
    "The baseline submission does not match the public-scored copy.\n",
    "Expected MD5: ", expected_baseline_md5, "\n",
    "Observed MD5: ", baseline_md5
  )
}

repeated_summary <- read.csv(file.path(
  "data_processed",
  "codex_set_context_network",
  "repeated_cv_summary.csv"
))
fold_selection <- read.csv(file.path(
  "data_processed",
  "codex_set_context_network",
  "repeated_cv_fold_selection.csv"
))
saved_verdict <- readLines(
  file.path(
    "data_processed",
    "codex_set_context_network",
    "verdict.txt"
  ),
  warn = FALSE
)
stopifnot(
  nrow(repeated_summary) == 1L,
  identical(
    repeated_summary$experiment[[1L]],
    expected_experiment
  ),
  identical(repeated_summary$stage[[1L]], "repeated_cv"),
  isTRUE(repeated_summary$promote[[1L]]),
  repeated_summary$point_gain[[1L]] > 0,
  repeated_summary$lower_95[[1L]] > 0,
  repeated_summary$positive_repeats[[1L]] == 6L,
  repeated_summary$n_repeats[[1L]] == 6L,
  length(saved_verdict) == 1L,
  startsWith(
    saved_verdict,
    "PROMOTE set-context utility network"
  ),
  nrow(fold_selection) == 30L,
  identical(
    sort(unique(as.integer(fold_selection$seed))),
    sort(c(4821L, 1907L, 2719L, 6151L, 8293L, 104729L))
  ),
  all(table(fold_selection$seed) == 5L),
  all(fold_selection$fold %in% 1:5),
  all(is.finite(fold_selection$set_context_weight))
)
full_weight <- mean(fold_selection$set_context_weight)
if (
  abs(full_weight - expected_full_weight) >
    .Machine$double.eps^0.5
) {
  stop(
    "Saved fold selections do not reproduce the frozen weight.\n",
    "Expected: ", expected_full_weight, "\n",
    "Observed: ", format(full_weight, digits = 16)
  )
}

# Load every definition from the validated runner into a private environment
# without executing its terminal run_experiment() call.
runner_lines <- readLines(runner_path, warn = FALSE)
nonblank <- which(nzchar(trimws(runner_lines)))
terminal_line <- tail(nonblank, 1L)
if (
  trimws(runner_lines[[terminal_line]]) !=
    "run_experiment()"
) {
  stop(
    "Validated runner no longer ends in the expected ",
    "run_experiment() call."
  )
}
runner_definition <- runner_lines[-terminal_line]
runner_environment <- new.env(parent = globalenv())

old_smoke <- Sys.getenv(
  "SET_CONTEXT_SMOKE",
  unset = NA_character_
)
old_repeated <- Sys.getenv(
  "SET_CONTEXT_REPEATED",
  unset = NA_character_
)
Sys.unsetenv("SET_CONTEXT_SMOKE")
Sys.setenv(SET_CONTEXT_REPEATED = "never")
eval(
  parse(text = runner_definition, keep.source = TRUE),
  envir = runner_environment
)
if (is.na(old_smoke)) {
  Sys.unsetenv("SET_CONTEXT_SMOKE")
} else {
  Sys.setenv(SET_CONTEXT_SMOKE = old_smoke)
}
if (is.na(old_repeated)) {
  Sys.unsetenv("SET_CONTEXT_REPEATED")
} else {
  Sys.setenv(SET_CONTEXT_REPEATED = old_repeated)
}

stopifnot(
  identical(
    runner_environment$experiment_id,
    expected_experiment
  ),
  identical(
    as.integer(runner_environment$canonical_network_seeds),
    as.integer(full_network_seeds)
  )
)

train_path <- file.path("csv files", "train.csv")
test_path <- file.path("csv files", "test.csv")
sample_path <- file.path(
  "csv files",
  "sample_submission.csv"
)
train <- read.csv(train_path)
test <- read.csv(test_path)
sample_submission <- read.csv(sample_path)
train <- train[order(train$No), , drop = FALSE]
test <- test[order(test$No), , drop = FALSE]
sample_submission <- sample_submission[
  order(sample_submission$No), , drop = FALSE
]
rownames(train) <- NULL
rownames(test) <- NULL
rownames(sample_submission) <- NULL

train_truth <- as.matrix(
  train[, probability_columns, drop = FALSE]
)
stopifnot(
  nrow(train) == 21565L,
  nrow(test) == 4997L,
  length(unique(train$Case)) == 1135L,
  length(unique(test$Case)) == 263L,
  length(intersect(
    unique(train$Case),
    unique(test$Case)
  )) == 0L,
  all(table(train$Case) == 19L),
  all(table(test$Case) == 19L),
  all(rowSums(train_truth) == 1L),
  identical(test$No, sample_submission$No),
  identical(names(sample_submission), c("No", probability_columns))
)

# Test outcomes are unknown. A valid one-hot placeholder is supplied only
# because the shared feature builder checks the shape of its truth matrix;
# these columns never enter the network inputs or prediction calculation.
test_placeholder <- test
for (column in probability_columns) {
  test_placeholder[[column]] <- 0L
}
test_placeholder$Ch1 <- 1L

training_long <- runner_environment$add_choice_context(
  runner_environment$build_full_long(train)
)
test_long <- runner_environment$add_choice_context(
  runner_environment$build_full_long(test_placeholder)
)
levels_by_attribute <-
  runner_environment$attribute_levels(training_long)
profile_scaler <- runner_environment$fit_scaler(
  training_long,
  runner_environment$profile_continuous
)
common_scaler <- runner_environment$fit_scaler(
  training_long,
  runner_environment$common_continuous
)
training_features <-
  runner_environment$build_set_context_features(
    training_long,
    levels_by_attribute,
    profile_scaler,
    common_scaler
  )
test_features <-
  runner_environment$build_set_context_features(
    test_long,
    levels_by_attribute,
    profile_scaler,
    common_scaler
  )
stopifnot(
  identical(
    training_features$profile_columns,
    test_features$profile_columns
  ),
  identical(
    training_features$common_columns,
    test_features$common_columns
  ),
  identical(
    as.integer(test_features$no),
    as.integer(test$No)
  ),
  ncol(training_features$profile) == 99L,
  ncol(training_features$common) == 24L
)

build_config <- runner_environment$network_config
build_seeds <- full_network_seeds
if (build_smoke) {
  build_config$epochs <- 2L
  build_seeds <- full_network_seeds[[1L]]
  cat(
    "FULL-BUILD SMOKE MODE: all training rows, ",
    "one seed, two epochs; no candidate CSV.\n",
    sep = ""
  )
} else {
  cat(
    "FULL BUILD: all training respondents, three seeds, ",
    "50 epochs; candidate is not submitted.\n",
    sep = ""
  )
}
cat(
  "Frozen set-context weight:",
  format(full_weight, digits = 6),
  "\n"
)

prediction_array <- array(
  NA_real_,
  dim = c(nrow(test), 4L, length(build_seeds))
)
fit_rows <- vector("list", length(build_seeds))
trace_rows <- vector("list", length(build_seeds))
for (seed_index in seq_along(build_seeds)) {
  seed <- as.integer(build_seeds[[seed_index]])
  cat(sprintf(
    "\n=== full-data network seed %d (%d/%d) ===\n",
    seed,
    seed_index,
    length(build_seeds)
  ))
  fitted <- runner_environment$fit_set_context_once(
    training_features,
    test_features,
    build_config,
    seed,
    checkpoint = NULL
  )
  prediction_array[, , seed_index] <-
    fitted$prediction
  fit_rows[[seed_index]] <- data.frame(
    seed = seed,
    epochs = build_config$epochs,
    profile_features =
      ncol(training_features$profile),
    common_features =
      ncol(training_features$common),
    final_training_logloss =
      fitted$final_training_logloss,
    elapsed_seconds = fitted$elapsed_seconds
  )
  trace <- fitted$trace
  trace$seed <- seed
  trace_rows[[seed_index]] <- trace
  cat(sprintf(
    "seed %d complete: train %.6f, %.1fs\n",
    seed,
    fitted$final_training_logloss,
    fitted$elapsed_seconds
  ))
  rm(fitted)
  invisible(gc())
}

set_context_prediction <- apply(
  prediction_array,
  c(1L, 2L),
  mean
)
set_context_prediction <-
  runner_environment$validate_probability(
    set_context_prediction,
    nrow(test)
  )

baseline_submission <- read.csv(baseline_path)
baseline_submission <- baseline_submission[
  order(baseline_submission$No), , drop = FALSE
]
rownames(baseline_submission) <- NULL
stopifnot(
  identical(
    names(baseline_submission),
    names(sample_submission)
  ),
  identical(
    baseline_submission$No,
    sample_submission$No
  ),
  !anyNA(baseline_submission)
)
baseline_prediction <- as.matrix(
  baseline_submission[
    , probability_columns, drop = FALSE
  ]
)
baseline_prediction <-
  runner_environment$validate_probability(
    baseline_prediction,
    nrow(test)
  )

candidate_prediction <-
  (1 - full_weight) * baseline_prediction +
  full_weight * set_context_prediction
candidate_prediction <-
  candidate_prediction /
    rowSums(candidate_prediction)
candidate_prediction <-
  runner_environment$validate_probability(
    candidate_prediction,
    nrow(test)
  )

absolute_change <- abs(
  candidate_prediction - baseline_prediction
)
row_max_change <- apply(absolute_change, 1L, max)
stopifnot(
  max(absolute_change) <= full_weight + 1e-10
)
diagnostics <- data.frame(
  set_context_weight = full_weight,
  set_context_min_probability =
    min(set_context_prediction),
  set_context_max_probability =
    max(set_context_prediction),
  set_context_row_sum_error =
    max(abs(rowSums(set_context_prediction) - 1)),
  candidate_min_probability =
    min(candidate_prediction),
  candidate_max_probability =
    max(candidate_prediction),
  candidate_row_sum_error =
    max(abs(rowSums(candidate_prediction) - 1)),
  mean_absolute_probability_change =
    mean(absolute_change),
  median_row_max_change =
    unname(quantile(row_max_change, 0.50)),
  p95_row_max_change =
    unname(quantile(row_max_change, 0.95)),
  p99_row_max_change =
    unname(quantile(row_max_change, 0.99)),
  max_probability_change =
    max(absolute_change),
  argmax_flip_rate =
    mean(
      max.col(baseline_prediction) !=
        max.col(candidate_prediction)
    ),
  flattened_probability_correlation =
    cor(
      as.vector(baseline_prediction),
      as.vector(candidate_prediction)
    )
)
fits <- do.call(rbind, fit_rows)
traces <- do.call(rbind, trace_rows)

input_hashes <- c(
  runner = runner_md5,
  train = unname(tools::md5sum(train_path)),
  test = unname(tools::md5sum(test_path)),
  sample_submission =
    unname(tools::md5sum(sample_path)),
  baseline = unname(tools::md5sum(baseline_path)),
  repeated_summary = unname(tools::md5sum(file.path(
    "data_processed",
    "codex_set_context_network",
    "repeated_cv_summary.csv"
  ))),
  fold_selection = unname(tools::md5sum(file.path(
    "data_processed",
    "codex_set_context_network",
    "repeated_cv_fold_selection.csv"
  ))),
  verdict = unname(tools::md5sum(file.path(
    "data_processed",
    "codex_set_context_network",
    "verdict.txt"
  )))
)
specification <- list(
  experiment = expected_experiment,
  config = build_config,
  seeds = as.integer(build_seeds),
  set_context_weight = full_weight,
  baseline_path = baseline_path,
  runner_path = runner_path,
  runner_md5 = runner_md5
)
build_result <- list(
  specification = specification,
  input_hashes = input_hashes,
  test_no = test$No,
  set_context_prediction = set_context_prediction,
  baseline_prediction = baseline_prediction,
  candidate_prediction = candidate_prediction,
  diagnostics = diagnostics,
  fits = fits,
  traces = traces,
  profile_columns =
    training_features$profile_columns,
  common_columns =
    training_features$common_columns,
  levels_by_attribute = levels_by_attribute,
  profile_scaler = profile_scaler,
  common_scaler = common_scaler
)

if (build_smoke) {
  smoke_path <- file.path(
    output_dir,
    "full_build_smoke_result.rds"
  )
  saveRDS(build_result, smoke_path)
  write.csv(
    diagnostics,
    file.path(output_dir, "full_build_smoke_summary.csv"),
    row.names = FALSE
  )
  cat(
    "\nFull-build smoke test completed successfully: ",
    smoke_path,
    "\n",
    sep = ""
  )
  print(fits, digits = 7)
  print(diagnostics, digits = 7)
} else {
  reference_path <- file.path(
    output_dir,
    "full_build_reference.rds"
  )
  reproduction_status <- "REFERENCE_CREATED_RERUN_REQUIRED"
  component_difference <- NA_real_
  candidate_difference <- NA_real_
  reproduction_pass <- FALSE

  if (file.exists(reference_path)) {
    reference <- readRDS(reference_path)
    stopifnot(
      identical(
        reference$specification,
        build_result$specification
      ),
      identical(
        reference$input_hashes,
        build_result$input_hashes
      ),
      identical(
        as.integer(reference$test_no),
        as.integer(build_result$test_no)
      )
    )
    component_difference <- max(abs(
      reference$set_context_prediction -
        build_result$set_context_prediction
    ))
    candidate_difference <- max(abs(
      reference$candidate_prediction -
        build_result$candidate_prediction
    ))
    reproduction_pass <-
      component_difference <= reproduction_tolerance &&
      candidate_difference <= reproduction_tolerance
    reproduction_status <- if (reproduction_pass) {
      "REPRODUCED_PASS"
    } else {
      "REPRODUCED_FAIL"
    }
    if (!reproduction_pass) {
      write.csv(
        data.frame(
          status = reproduction_status,
          tolerance = reproduction_tolerance,
          component_max_abs_difference =
            component_difference,
          candidate_max_abs_difference =
            candidate_difference
        ),
        file.path(
          output_dir,
          "full_build_reproduction_check.csv"
        ),
        row.names = FALSE
      )
      stop(
        "Full-data reproduction check failed. ",
        "No candidate CSV was overwritten."
      )
    }
  } else {
    saveRDS(build_result, reference_path)
  }

  submission <- data.frame(
    No = test$No,
    Ch1 = candidate_prediction[, 1L],
    Ch2 = candidate_prediction[, 2L],
    Ch3 = candidate_prediction[, 3L],
    Ch4 = candidate_prediction[, 4L]
  )
  stopifnot(
    identical(names(submission), names(sample_submission)),
    identical(submission$No, sample_submission$No),
    !anyNA(submission),
    all(
      as.matrix(
        submission[, probability_columns, drop = FALSE]
      ) > 0
    ),
    max(abs(rowSums(
      submission[, probability_columns, drop = FALSE]
    ) - 1)) < 1e-10
  )
  candidate_path <-
    "submission_set_context_v14_candidate.csv"
  write.csv(
    submission,
    candidate_path,
    row.names = FALSE
  )
  candidate_md5 <- unname(
    tools::md5sum(candidate_path)
  )
  saveRDS(
    build_result,
    file.path(output_dir, "full_build_latest.rds")
  )
  write.csv(
    cbind(
      data.frame(
        status = reproduction_status,
        candidate_path = candidate_path,
        candidate_md5 = candidate_md5
      ),
      diagnostics
    ),
    file.path(output_dir, "full_build_summary.csv"),
    row.names = FALSE
  )
  write.csv(
    data.frame(
      status = reproduction_status,
      tolerance = reproduction_tolerance,
      component_max_abs_difference =
        component_difference,
      candidate_max_abs_difference =
        candidate_difference,
      pass = reproduction_pass
    ),
    file.path(
      output_dir,
      "full_build_reproduction_check.csv"
    ),
    row.names = FALSE
  )

  cat("\nFull-data candidate build completed.\n")
  cat("Status:", reproduction_status, "\n")
  cat("Candidate:", candidate_path, "\n")
  cat("Candidate MD5:", candidate_md5, "\n")
  if (!reproduction_pass) {
    cat(
      "Rerun the same full-build command once more ",
      "before considering submission.\n",
      sep = ""
    )
  }
  print(fits, digits = 7)
  print(diagnostics, digits = 7)
}
