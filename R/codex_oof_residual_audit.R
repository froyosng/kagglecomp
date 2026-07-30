# Repeated-OOF residual audit for the set-context v14 and rejected two-head
# ensemble. Statistical scope and decision thresholds are frozen in:
#   codex_oof_residual_audit_preregister.md
#
# Smoke:
#   Sys.setenv(OOF_RESIDUAL_SMOKE = "1")
#   source("R/codex_oof_residual_audit.R")
#   Sys.unsetenv("OOF_RESIDUAL_SMOKE")
#
# Full audit:
#   Sys.unsetenv("OOF_RESIDUAL_SMOKE")
#   source("R/codex_oof_residual_audit.R")

experiment_id <- "repeated_oof_head_residual_audit_v1"
output_dir <- file.path(
  "data_processed", "codex_oof_residual_audit"
)
v14_dir <- file.path(
  "data_processed", "codex_set_context_network"
)
two_head_dir <- file.path(
  "data_processed", "codex_two_head_ensemble"
)
all_seeds <- c(4821L, 1907L, 2719L, 6151L, 8293L, 104729L)
canonical_seed <- 4821L
matrix_tolerance <- 1e-10
metric_tolerance <- 1e-9
minimum_respondents <- 50L
minimum_absolute_gap <- 0.01
minimum_head_gain <- 0.00015
smoke_mode <- identical(
  Sys.getenv("OOF_RESIDUAL_SMOKE", unset = "0"),
  "1"
)

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

stop_with_context <- function(message) {
  stop(
    paste0("[", experiment_id, "] ", message),
    call. = FALSE
  )
}

validate_probability <- function(prediction, n_rows) {
  prediction <- as.matrix(prediction)
  if (
    nrow(prediction) != n_rows ||
    ncol(prediction) != 4L
  ) {
    stop_with_context("Probability matrix has the wrong dimensions.")
  }
  storage.mode(prediction) <- "double"
  if (
    any(!is.finite(prediction)) ||
    any(prediction <= 0) ||
    any(prediction >= 1) ||
    max(abs(rowSums(prediction) - 1)) > matrix_tolerance
  ) {
    stop_with_context(
      "Probability matrix is non-finite, non-positive, or not normalized."
    )
  }
  prediction
}

validate_fold_map <- function(train, fold_map) {
  if (
    is.null(names(fold_map)) ||
    anyDuplicated(names(fold_map))
  ) {
    stop_with_context("Fold map must be a uniquely named vector.")
  }
  case_levels <- sort(unique(train$Case))
  aligned <- unname(
    fold_map[as.character(case_levels)]
  )
  if (
    length(aligned) != length(case_levels) ||
    anyNA(aligned) ||
    !all(aligned %in% 1:5)
  ) {
    stop_with_context(
      "Fold map does not cover every training respondent exactly once."
    )
  }
  fold_map
}

artifact_path <- function(directory, seed) {
  if (seed == canonical_seed) {
    file.path(directory, "canonical_result.rds")
  } else {
    file.path(
      directory,
      sprintf("repeat_result_%d.rds", seed)
    )
  }
}

load_seed_artifacts <- function(train, seed) {
  v14_path <- artifact_path(v14_dir, seed)
  two_head_path <- artifact_path(two_head_dir, seed)
  if (!file.exists(v14_path)) {
    stop_with_context(
      paste("Missing v14 artifact:", v14_path)
    )
  }
  if (!file.exists(two_head_path)) {
    stop_with_context(
      paste("Missing two-head artifact:", two_head_path)
    )
  }

  v14 <- readRDS(v14_path)
  two_head <- readRDS(two_head_path)
  v14_required <- c(
    "experiment_id", "fold_map", "candidate_prediction"
  )
  two_head_required <- c(
    "experiment_id", "fold_map",
    "baseline_prediction", "candidate_prediction"
  )
  if (!all(v14_required %in% names(v14))) {
    stop_with_context(
      paste("Incomplete v14 artifact:", v14_path)
    )
  }
  if (!all(two_head_required %in% names(two_head))) {
    stop_with_context(
      paste("Incomplete two-head artifact:", two_head_path)
    )
  }
  if (!identical(
    v14$experiment_id,
    "set_context_utility_network_v1"
  )) {
    stop_with_context("Unexpected v14 experiment id.")
  }
  if (!identical(
    two_head$experiment_id,
    "two_head_optout_bundle_ensemble_v1"
  )) {
    stop_with_context("Unexpected two-head experiment id.")
  }

  v14_fold_map <- validate_fold_map(train, v14$fold_map)
  two_head_fold_map <- validate_fold_map(
    train, two_head$fold_map
  )
  case_levels <- sort(unique(train$Case))
  v14_fold <- unname(
    v14_fold_map[as.character(case_levels)]
  )
  two_head_fold <- unname(
    two_head_fold_map[as.character(case_levels)]
  )
  if (!identical(
    as.integer(v14_fold),
    as.integer(two_head_fold)
  )) {
    stop_with_context(
      paste("Fold maps differ for seed", seed)
    )
  }

  v14_prediction <- validate_probability(
    v14$candidate_prediction, nrow(train)
  )
  two_head_baseline <- validate_probability(
    two_head$baseline_prediction, nrow(train)
  )
  two_head_prediction <- validate_probability(
    two_head$candidate_prediction, nrow(train)
  )
  if (
    max(abs(v14_prediction - two_head_baseline)) >
      matrix_tolerance
  ) {
    stop_with_context(
      paste(
        "Two-head baseline does not reproduce v14 for seed",
        seed
      )
    )
  }

  list(
    seed = seed,
    v14_path = v14_path,
    two_head_path = two_head_path,
    fold_map = v14_fold_map,
    row_fold = unname(
      v14_fold_map[as.character(train$Case)]
    ),
    v14_prediction = v14_prediction,
    two_head_prediction = two_head_prediction
  )
}

loss_parts <- function(truth, prediction) {
  truth <- as.matrix(truth)
  prediction <- validate_probability(
    prediction, nrow(truth)
  )
  q <- prediction[, 4L]
  inside <- 1 - truth[, 4L]
  conditional <- prediction[, 1:3, drop = FALSE] /
    (1 - q)
  if (
    any(!is.finite(conditional)) ||
    any(conditional <= 0) ||
    max(abs(rowSums(conditional) - 1)) >
      matrix_tolerance
  ) {
    stop_with_context(
      "Conditional bundle probabilities are invalid."
    )
  }

  binary_loss <- -(
    truth[, 4L] * log(q) +
      inside * log(1 - q)
  )
  bundle_contribution <- -inside * rowSums(
    truth[, 1:3, drop = FALSE] *
      log(conditional)
  )
  total_loss <- -rowSums(
    truth * log(prediction)
  )
  if (
    max(
      abs(
        total_loss -
          binary_loss -
          bundle_contribution
      )
    ) > matrix_tolerance
  ) {
    stop_with_context(
      "Four-class loss does not equal the two-head decomposition."
    )
  }

  list(
    q = q,
    conditional = conditional,
    total_loss = total_loss,
    binary_loss = binary_loss,
    bundle_contribution = bundle_contribution
  )
}

read_reported_seed_gains <- function() {
  path <- file.path(
    two_head_dir, "repeated_cv_by_seed.csv"
  )
  if (!file.exists(path)) {
    stop_with_context(
      paste("Missing two-head seed summary:", path)
    )
  }
  reported <- read.csv(path)
  required <- c(
    "seed", "baseline_logloss",
    "candidate_logloss", "gain"
  )
  if (!all(required %in% names(reported))) {
    stop_with_context(
      "Two-head repeated_cv_by_seed.csv has the wrong schema."
    )
  }
  if (!setequal(
    as.integer(reported$seed),
    all_seeds
  )) {
    stop_with_context(
      "Two-head seed summary does not contain the frozen six seeds."
    )
  }
  reported
}

cluster_mean_stats <- function(residual, case) {
  keep <- is.finite(residual) & !is.na(case)
  residual <- residual[keep]
  case <- case[keep]
  n_rows <- length(residual)
  n_cases <- length(unique(case))
  if (n_rows == 0L || n_cases < 2L) {
    return(list(
      mean = if (n_rows == 0L) NA_real_ else mean(residual),
      se = NA_real_,
      lower_95 = NA_real_,
      upper_95 = NA_real_,
      p_value = NA_real_
    ))
  }
  estimate <- mean(residual)
  centered_cluster_sum <- tapply(
    residual - estimate,
    case,
    sum
  )
  se <- sqrt(
    n_cases / (n_cases - 1) *
      sum(centered_cluster_sum^2)
  ) / n_rows
  if (!is.finite(se) || se <= 0) {
    p_value <- if (
      isTRUE(all.equal(estimate, 0))
    ) {
      1
    } else {
      0
    }
  } else {
    p_value <- 2 * stats::pnorm(
      -abs(estimate / se)
    )
  }
  list(
    mean = estimate,
    se = se,
    lower_95 = estimate - 1.96 * se,
    upper_95 = estimate + 1.96 * se,
    p_value = p_value
  )
}

respondent_quintile <- function(train, variable) {
  respondent_rows <- !duplicated(train$Case)
  respondent_case <- train$Case[respondent_rows]
  respondent_value <- variable[respondent_rows]
  if (anyNA(respondent_value)) {
    stop_with_context(
      "Income quintile input contains missing values."
    )
  }
  ranked <- rank(
    respondent_value,
    ties.method = "average"
  )
  quintile <- pmin(
    5L,
    pmax(
      1L,
      ceiling(
        5 * ranked / length(ranked)
      )
    )
  )
  quintile_map <- setNames(
    paste0("Q", quintile),
    as.character(respondent_case)
  )
  unname(
    quintile_map[as.character(train$Case)]
  )
}

task_bin <- function(task) {
  cut(
    task,
    breaks = c(0, 5, 10, 15, 19),
    labels = c("1-5", "6-10", "11-15", "16-19"),
    include.lowest = TRUE,
    right = TRUE
  )
}

seed_sign_count <- function(
    seed_residuals,
    rows,
    reference_gap
) {
  if (!is.finite(reference_gap) || reference_gap == 0) {
    return(0L)
  }
  seed_gaps <- vapply(
    seed_residuals,
    function(residual) {
      mean(residual[rows])
    },
    numeric(1L)
  )
  as.integer(
    sum(
      sign(seed_gaps) == sign(reference_gap)
    )
  )
}

build_optout_slices <- function(
    train,
    truth,
    model_name,
    mean_prediction,
    seed_predictions
) {
  mean_parts <- loss_parts(truth, mean_prediction)
  mean_residual <- truth[, 4L] - mean_parts$q
  seed_residuals <- lapply(
    seed_predictions,
    function(prediction) {
      truth[, 4L] -
        loss_parts(truth, prediction)$q
    }
  )
  slices <- list(
    overall = rep("all", nrow(train)),
    task = as.character(train$Task),
    task_bin = as.character(task_bin(train$Task)),
    segmentind = as.character(train$segmentind),
    income_quintile = respondent_quintile(
      train, train$incomea
    ),
    ageind = as.character(train$ageind),
    regionind = as.character(train$regionind),
    pparkind = as.character(train$pparkind)
  )

  rows_out <- list()
  cursor <- 1L
  for (dimension in names(slices)) {
    group <- slices[[dimension]]
    for (level in sort(unique(group))) {
      rows <- which(group == level)
      stats <- cluster_mean_stats(
        mean_residual[rows],
        train$Case[rows]
      )
      rows_out[[cursor]] <- data.frame(
        model = model_name,
        head = "optout",
        dimension = dimension,
        level = as.character(level),
        n_tasks = length(rows),
        n_respondents = length(
          unique(train$Case[rows])
        ),
        observed_rate = mean(truth[rows, 4L]),
        predicted_rate = mean(mean_parts$q[rows]),
        calibration_gap = stats$mean,
        cluster_se = stats$se,
        lower_95 = stats$lower_95,
        upper_95 = stats$upper_95,
        p_value = stats$p_value,
        same_sign_seeds = seed_sign_count(
          seed_residuals, rows, stats$mean
        ),
        stringsAsFactors = FALSE
      )
      cursor <- cursor + 1L
    }
  }
  result <- do.call(rbind, rows_out)
  result$holm_p <- stats::p.adjust(
    result$p_value,
    method = "holm"
  )
  result$actionable <- (
    model_name == "v14" &
      result$n_respondents >= minimum_respondents &
      abs(result$calibration_gap) >=
        minimum_absolute_gap &
      result$holm_p < 0.05 &
      result$same_sign_seeds ==
        length(all_seeds)
  )
  result
}

gap_bucket <- function(gap) {
  ifelse(
    gap == 0,
    "0",
    ifelse(
      gap <= 2,
      "1-2",
      ifelse(gap <= 5, "3-5", "6+")
    )
  )
}

bundle_long_data <- function(
    train,
    truth,
    mean_prediction,
    seed_predictions
) {
  inside_rows <- which(truth[, 4L] == 0)
  if (length(inside_rows) == 0L) {
    stop_with_context(
      "No observed inside-bundle choices were found."
    )
  }
  task_row <- rep(inside_rows, each = 3L)
  alternative <- rep(1:3, times = length(inside_rows))
  observed <- as.vector(
    t(truth[inside_rows, 1:3, drop = FALSE])
  )
  mean_conditional <- loss_parts(
    truth, mean_prediction
  )$conditional
  predicted <- as.vector(
    t(mean_conditional[inside_rows, , drop = FALSE])
  )
  seed_predicted <- lapply(
    seed_predictions,
    function(prediction) {
      conditional <- loss_parts(
        truth, prediction
      )$conditional
      as.vector(
        t(conditional[
          inside_rows, , drop = FALSE
        ])
      )
    }
  )

  price_matrix <- as.matrix(
    train[
      ,
      paste0("Price", 1:3),
      drop = FALSE
    ]
  )
  inside_price <- price_matrix[
    inside_rows, , drop = FALSE
  ]
  price <- as.vector(t(inside_price))
  minimum_price <- apply(inside_price, 1L, min)
  maximum_price <- apply(inside_price, 1L, max)
  repeated_minimum <- rep(minimum_price, each = 3L)
  repeated_maximum <- rep(maximum_price, each = 3L)
  all_tied <- repeated_minimum == repeated_maximum
  price_role <- ifelse(
    all_tied,
    "all_tied",
    ifelse(
      price == repeated_minimum,
      "cheapest",
      ifelse(
        price == repeated_maximum,
        "dearest",
        "middle"
      )
    )
  )

  list(
    task_row = task_row,
    case = train$Case[task_row],
    observed = observed,
    predicted = predicted,
    residual = observed - predicted,
    seed_residuals = lapply(
      seed_predicted,
      function(value) observed - value
    ),
    slices = list(
      bundle_position = as.character(alternative),
      price_level = as.character(price),
      price_role = price_role,
      price_gap_min = gap_bucket(
        price - repeated_minimum
      ),
      price_gap_max = gap_bucket(
        repeated_maximum - price
      ),
      task_bin = as.character(
        task_bin(train$Task[task_row])
      )
    )
  )
}

build_bundle_slices <- function(
    train,
    truth,
    model_name,
    mean_prediction,
    seed_predictions
) {
  long <- bundle_long_data(
    train,
    truth,
    mean_prediction,
    seed_predictions
  )
  rows_out <- list()
  cursor <- 1L
  for (dimension in names(long$slices)) {
    group <- long$slices[[dimension]]
    for (level in sort(unique(group))) {
      rows <- which(group == level)
      stats <- cluster_mean_stats(
        long$residual[rows],
        long$case[rows]
      )
      rows_out[[cursor]] <- data.frame(
        model = model_name,
        head = "bundle",
        dimension = dimension,
        level = as.character(level),
        n_alternative_rows = length(rows),
        n_tasks = length(unique(long$task_row[rows])),
        n_respondents = length(
          unique(long$case[rows])
        ),
        observed_rate = mean(long$observed[rows]),
        predicted_rate = mean(long$predicted[rows]),
        calibration_gap = stats$mean,
        cluster_se = stats$se,
        lower_95 = stats$lower_95,
        upper_95 = stats$upper_95,
        p_value = stats$p_value,
        same_sign_seeds = seed_sign_count(
          long$seed_residuals,
          rows,
          stats$mean
        ),
        stringsAsFactors = FALSE
      )
      cursor <- cursor + 1L
    }
  }
  result <- do.call(rbind, rows_out)
  result$holm_p <- stats::p.adjust(
    result$p_value,
    method = "holm"
  )
  result$actionable <- (
    model_name == "v14" &
      result$n_respondents >= minimum_respondents &
      abs(result$calibration_gap) >=
        minimum_absolute_gap &
      result$holm_p < 0.05 &
      result$same_sign_seeds ==
        length(all_seeds)
  )
  result
}

mean_prediction <- function(registry, field) {
  total <- matrix(
    0,
    nrow = nrow(registry[[1L]][[field]]),
    ncol = 4L
  )
  for (artifact in registry) {
    total <- total + artifact[[field]]
  }
  validate_probability(
    total / length(registry),
    nrow(total)
  )
}

decompose_seed <- function(
    truth,
    artifact
) {
  v14 <- loss_parts(
    truth, artifact$v14_prediction
  )
  two_head <- loss_parts(
    truth, artifact$two_head_prediction
  )
  inside <- truth[, 4L] == 0
  data.frame(
    seed = artifact$seed,
    v14_total_logloss = mean(v14$total_loss),
    two_head_total_logloss = mean(
      two_head$total_loss
    ),
    total_gain = mean(
      v14$total_loss - two_head$total_loss
    ),
    v14_optout_loss = mean(v14$binary_loss),
    two_head_optout_loss = mean(
      two_head$binary_loss
    ),
    optout_gain = mean(
      v14$binary_loss - two_head$binary_loss
    ),
    v14_bundle_contribution = mean(
      v14$bundle_contribution
    ),
    two_head_bundle_contribution = mean(
      two_head$bundle_contribution
    ),
    bundle_gain = mean(
      v14$bundle_contribution -
        two_head$bundle_contribution
    ),
    v14_bundle_conditional_loss = mean(
      v14$bundle_contribution[inside]
    ),
    two_head_bundle_conditional_loss = mean(
      two_head$bundle_contribution[inside]
    ),
    stringsAsFactors = FALSE
  )
}

summarize_decomposition <- function(by_seed) {
  data.frame(
    head = c("optout", "bundle"),
    v14_head_loss = c(
      mean(by_seed$v14_optout_loss),
      mean(by_seed$v14_bundle_contribution)
    ),
    two_head_head_loss = c(
      mean(by_seed$two_head_optout_loss),
      mean(by_seed$two_head_bundle_contribution)
    ),
    mean_gain = c(
      mean(by_seed$optout_gain),
      mean(by_seed$bundle_gain)
    ),
    minimum_seed_gain = c(
      min(by_seed$optout_gain),
      min(by_seed$bundle_gain)
    ),
    maximum_seed_gain = c(
      max(by_seed$optout_gain),
      max(by_seed$bundle_gain)
    ),
    positive_seeds = c(
      sum(by_seed$optout_gain > 0),
      sum(by_seed$bundle_gain > 0)
    ),
    n_seeds = nrow(by_seed),
    stringsAsFactors = FALSE
  )
}

run_smoke <- function(train, truth) {
  cat("\nRunning residual-audit smoke test.\n")
  artifact <- load_seed_artifacts(
    train, canonical_seed
  )
  fold_rows <- which(artifact$row_fold == 1L)
  if (length(fold_rows) == 0L) {
    stop_with_context(
      "Canonical outer fold 1 has no rows."
    )
  }
  v14 <- loss_parts(
    truth[fold_rows, , drop = FALSE],
    artifact$v14_prediction[
      fold_rows, , drop = FALSE
    ]
  )
  two_head <- loss_parts(
    truth[fold_rows, , drop = FALSE],
    artifact$two_head_prediction[
      fold_rows, , drop = FALSE
    ]
  )
  checkpoint <- list(
    experiment_id = experiment_id,
    seed = canonical_seed,
    fold = 1L,
    n_rows = length(fold_rows),
    v14_loss = mean(v14$total_loss),
    two_head_loss = mean(two_head$total_loss)
  )
  checkpoint_path <- file.path(
    output_dir, "smoke_checkpoint.rds"
  )
  saveRDS(checkpoint, checkpoint_path)
  reloaded <- readRDS(checkpoint_path)
  if (!identical(checkpoint, reloaded)) {
    stop_with_context(
      "Smoke checkpoint failed exact save/reload."
    )
  }
  cat(
    sprintf(
      "Canonical fold 1 rows: %d\n",
      length(fold_rows)
    )
  )
  cat(
    sprintf(
      "Canonical fold 1 v14 loss: %.9f\n",
      mean(v14$total_loss)
    )
  )
  cat(
    sprintf(
      "Canonical fold 1 two-head loss: %.9f\n",
      mean(two_head$total_loss)
    )
  )
  cat("Smoke test completed successfully\n")
  invisible(checkpoint)
}

run_audit <- function() {
  train_path <- file.path("csv files", "train.csv")
  test_path <- file.path("csv files", "test.csv")
  if (!file.exists(train_path) || !file.exists(test_path)) {
    stop_with_context(
      "Expected csv files/train.csv and csv files/test.csv."
    )
  }
  train <- read.csv(train_path)
  test <- read.csv(test_path)
  train <- train[
    order(train$No), , drop = FALSE
  ]
  test <- test[
    order(test$No), , drop = FALSE
  ]
  rownames(train) <- NULL
  rownames(test) <- NULL
  truth <- as.matrix(
    train[
      ,
      paste0("Ch", 1:4),
      drop = FALSE
    ]
  )
  if (
    nrow(train) != 21565L ||
    nrow(test) != 4997L ||
    length(unique(train$Case)) != 1135L ||
    length(unique(test$Case)) != 263L ||
    length(intersect(
      unique(train$Case),
      unique(test$Case)
    )) != 0L ||
    !all(table(train$Case) == 19L) ||
    !all(rowSums(truth) == 1L)
  ) {
    stop_with_context(
      "Raw data do not match the frozen competition schema."
    )
  }

  if (smoke_mode) {
    return(run_smoke(train, truth))
  }

  cat("\nRunning", experiment_id, "\n")
  registry <- lapply(
    all_seeds,
    function(seed) {
      cat(sprintf(
        "Loading and validating seed %d\n",
        seed
      ))
      load_seed_artifacts(train, seed)
    }
  )
  names(registry) <- as.character(all_seeds)

  by_seed <- do.call(
    rbind,
    lapply(
      registry,
      function(artifact) {
        decompose_seed(truth, artifact)
      }
    )
  )
  rownames(by_seed) <- NULL
  if (
    max(
      abs(
        by_seed$total_gain -
          by_seed$optout_gain -
          by_seed$bundle_gain
      )
    ) > matrix_tolerance
  ) {
    stop_with_context(
      "Head gains do not sum to total gain."
    )
  }

  reported <- read_reported_seed_gains()
  reported <- reported[
    match(by_seed$seed, reported$seed),
    ,
    drop = FALSE
  ]
  if (
    max(
      abs(
        by_seed$v14_total_logloss -
          reported$baseline_logloss
      )
    ) > metric_tolerance ||
    max(
      abs(
        by_seed$two_head_total_logloss -
          reported$candidate_logloss
      )
    ) > metric_tolerance ||
    max(
      abs(
        by_seed$total_gain -
          reported$gain
      )
    ) > metric_tolerance
  ) {
    stop_with_context(
      "Recomputed seed losses do not reproduce the reported two-head result."
    )
  }

  decomposition_summary <- summarize_decomposition(
    by_seed
  )
  v14_seed_predictions <- lapply(
    registry,
    function(value) value$v14_prediction
  )
  two_head_seed_predictions <- lapply(
    registry,
    function(value) value$two_head_prediction
  )
  v14_mean <- mean_prediction(
    registry, "v14_prediction"
  )
  two_head_mean <- mean_prediction(
    registry, "two_head_prediction"
  )

  cat("Computing opt-out slice diagnostics.\n")
  optout_slices <- rbind(
    build_optout_slices(
      train,
      truth,
      "v14",
      v14_mean,
      v14_seed_predictions
    ),
    build_optout_slices(
      train,
      truth,
      "two_head",
      two_head_mean,
      two_head_seed_predictions
    )
  )
  rownames(optout_slices) <- NULL

  cat("Computing conditional-bundle slice diagnostics.\n")
  bundle_slices <- rbind(
    build_bundle_slices(
      train,
      truth,
      "v14",
      v14_mean,
      v14_seed_predictions
    ),
    build_bundle_slices(
      train,
      truth,
      "two_head",
      two_head_mean,
      two_head_seed_predictions
    )
  )
  rownames(bundle_slices) <- NULL

  actionable_columns <- c(
    "model", "head", "dimension", "level",
    "n_tasks", "n_respondents",
    "observed_rate", "predicted_rate",
    "calibration_gap", "cluster_se",
    "lower_95", "upper_95", "p_value",
    "same_sign_seeds", "holm_p", "actionable"
  )
  actionable <- rbind(
    optout_slices[
      optout_slices$actionable,
      actionable_columns,
      drop = FALSE
    ],
    bundle_slices[
      bundle_slices$actionable,
      actionable_columns,
      drop = FALSE
    ]
  )
  rownames(actionable) <- NULL
  actionable_count <- c(
    optout = sum(
      optout_slices$model == "v14" &
        optout_slices$actionable
    ),
    bundle = sum(
      bundle_slices$model == "v14" &
        bundle_slices$actionable
    )
  )
  audit_summary <- decomposition_summary
  audit_summary$actionable_slice_count <-
    unname(
      actionable_count[audit_summary$head]
    )
  audit_summary$gain_gate <- (
    audit_summary$positive_seeds >= 5L &
      audit_summary$mean_gain >= minimum_head_gain
  )
  audit_summary$slice_gate <-
    audit_summary$actionable_slice_count > 0L
  audit_summary$eligible <-
    audit_summary$gain_gate &
    audit_summary$slice_gate

  eligible_heads <- audit_summary$head[
    audit_summary$eligible
  ]
  if (length(eligible_heads) == 0L) {
    selected_head <- NA_character_
    verdict <- paste(
      "STOP MODEL SEARCH: neither head satisfies the",
      "pre-registered gain-and-residual rule; retain exact v14",
      "and redirect effort to the final report."
    )
  } else {
    eligible_rows <- audit_summary[
      audit_summary$eligible, , drop = FALSE
    ]
    selected_head <- eligible_rows$head[
      which.max(eligible_rows$mean_gain)
    ]
    model_label <- if (selected_head == "optout") {
      "a separate opt-out classifier"
    } else {
      "a separate conditional bundle-choice model"
    }
    verdict <- paste(
      "PROCEED TO A SEPARATE PRE-REGISTRATION:",
      model_label,
      "is supported by the frozen audit;",
      "do not fit or submit it without a new protocol."
    )
  }
  audit_summary$selected_head <-
    !is.na(selected_head) &
    audit_summary$head == selected_head

  write.csv(
    by_seed,
    file.path(
      output_dir,
      "head_decomposition_by_seed.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    decomposition_summary,
    file.path(
      output_dir,
      "head_decomposition_summary.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    optout_slices,
    file.path(
      output_dir,
      "optout_slice_diagnostics.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    bundle_slices,
    file.path(
      output_dir,
      "bundle_slice_diagnostics.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    actionable,
    file.path(
      output_dir,
      "actionable_slices.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    audit_summary,
    file.path(
      output_dir,
      "audit_summary.csv"
    ),
    row.names = FALSE
  )
  writeLines(
    verdict,
    file.path(output_dir, "verdict.txt")
  )
  saveRDS(
    list(
      experiment_id = experiment_id,
      seeds = all_seeds,
      head_decomposition_by_seed = by_seed,
      head_decomposition_summary =
        decomposition_summary,
      v14_repeated_oof_mean = v14_mean,
      two_head_repeated_oof_mean = two_head_mean,
      optout_slice_diagnostics = optout_slices,
      bundle_slice_diagnostics = bundle_slices,
      actionable_slices = actionable,
      audit_summary = audit_summary,
      verdict = verdict
    ),
    file.path(output_dir, "audit_result.rds")
  )

  cat("\nHead decomposition:\n")
  print(decomposition_summary, digits = 9)
  cat("\nAudit decision:\n")
  print(audit_summary, digits = 9)
  cat("\n", verdict, "\n", sep = "")
  invisible(list(
    decomposition = decomposition_summary,
    audit = audit_summary,
    actionable = actionable,
    verdict = verdict
  ))
}

run_audit()
