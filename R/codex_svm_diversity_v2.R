# Calibrated RBF-SVM as a diversity component for the frozen current best.
#
# Pre-registration:
#   codex_svm_diversity_preregister.md
#
# Smoke test:
#   Sys.setenv(SVM_DIVERSITY_SMOKE = "1")
#   source("R/codex_svm_diversity.R")
#   Sys.unsetenv("SVM_DIVERSITY_SMOKE")
#
# Full run:
#   Sys.unsetenv("SVM_DIVERSITY_SMOKE")
#   Sys.setenv(SVM_DIVERSITY_REPEATED = "auto")
#   source("R/codex_svm_diversity.R")

options(stringsAsFactors = FALSE)

required_packages <- c(
  "e1071", "nnet", "mlogit", "dfidx", "xgboost"
)
missing_packages <- required_packages[
  !vapply(
    required_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]
if (length(missing_packages) > 0L) {
  stop(
    "Missing R package(s): ",
    paste(missing_packages, collapse = ", "),
    "\nInstall them once with:\n  install.packages(c(",
    paste(sprintf('"%s"', missing_packages), collapse = ", "),
    "))"
  )
}

experiment_id <- "calibrated_rbf_svm_diversity_v1"
output_dir <- file.path(
  "data_processed", "codex_svm_diversity"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

smoke_mode <- identical(
  Sys.getenv("SVM_DIVERSITY_SMOKE", "0"), "1"
)
repeated_mode <- tolower(
  Sys.getenv("SVM_DIVERSITY_REPEATED", "auto")
)
stopifnot(repeated_mode %in% c("auto", "never", "always"))

canonical_baseline_target <- 1.143686618134879
submitted_crossfit_reference <- 1.14378944178118
near_miss_lower_limit <- -0.00025
additional_seeds <- c(
  1907L, 2719L, 6151L, 8293L, 104729L
)
cost_grid <- c(0.5, 2, 8)
temperature_grid <- c(0.75, 1, 1.25, 1.5)
blend_weight_grid <- seq(0, 0.20, by = 0.01)
bootstrap_replicates <- if (smoke_mode) {
  200L
} else {
  as.integer(Sys.getenv("SVM_DIVERSITY_N_BOOT", "100000"))
}

input_files <- c(
  file.path("csv files", "train.csv"),
  file.path("csv files", "test.csv"),
  file.path("data_processed", "oof_ensemble_v10.rds"),
  file.path("R", "codex_mlp_ensemble.R")
)
missing_files <- input_files[!file.exists(input_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing required project file(s):\n  ",
    paste(missing_files, collapse = "\n  "),
    "\nRun this script from the kagglecomp repository root."
  )
}

log_loss_matrix_local <- function(truth, prediction) {
  prediction <- validate_probability(prediction, nrow(truth))
  -mean(rowSums(
    as.matrix(truth) * log(pmax(prediction, 1e-15))
  ))
}

row_log_loss_local <- function(truth, prediction) {
  prediction <- validate_probability(prediction, nrow(truth))
  -rowSums(
    as.matrix(truth) * log(pmax(prediction, 1e-15))
  )
}

validate_probability <- function(prediction, n_rows = NULL) {
  prediction <- as.matrix(prediction)
  if (!is.null(n_rows)) {
    stopifnot(
      identical(
        dim(prediction),
        c(as.integer(n_rows), 4L)
      )
    )
  } else {
    stopifnot(ncol(prediction) == 4L)
  }
  stopifnot(
    !anyNA(prediction),
    all(is.finite(prediction)),
    all(prediction >= 0)
  )
  prediction <- pmax(prediction, 1e-15)
  prediction <- prediction / rowSums(prediction)
  stopifnot(
    max(abs(rowSums(prediction) - 1)) < 1e-10
  )
  prediction
}

apply_temperature <- function(prediction, temperature) {
  stopifnot(
    length(temperature) == 1L,
    is.finite(temperature),
    temperature > 0
  )
  prediction <- validate_probability(
    prediction, nrow(prediction)
  )
  adjusted <- prediction^(1 / temperature)
  validate_probability(adjusted, nrow(prediction))
}

balanced_fold_map <- function(cases, folds, seed) {
  cases <- sort(unique(as.integer(cases)))
  set.seed(as.integer(seed))
  assignment <- sample(
    rep(seq_len(folds), length.out = length(cases))
  )
  names(assignment) <- as.character(cases)
  stopifnot(
    length(assignment) == length(cases),
    all(table(assignment) %in%
      c(floor(length(cases) / folds),
        ceiling(length(cases) / folds)))
  )
  assignment
}

canonical_fold_map <- function(train, saved) {
  fold_map <- saved$fold_of_case
  stopifnot(
    length(fold_map) == length(unique(train$Case)),
    !anyNA(
      fold_map[as.character(unique(train$Case))]
    ),
    identical(
      sort(unique(as.integer(fold_map))),
      1:5
    )
  )
  fold_map
}

repeated_fold_map <- function(cases, seed) {
  balanced_fold_map(cases, 5L, seed)
}

mlp_helper_environment <- function() {
  helper <- new.env(parent = globalenv())
  fold_map_function <- canonical_fold_map
  helper$source <- function(file, local = TRUE, ...) {
    if (identical(local, FALSE)) {
      stop(
        "A nested MLP helper tried to source into the global ",
        "environment."
      )
    }
    base::source(file, local = helper, ...)
  }
  old_stage <- Sys.getenv(
    "CODEX_STAGE", unset = NA_character_
  )
  Sys.setenv(CODEX_STAGE = "define")
  on.exit({
    if (is.na(old_stage)) {
      Sys.unsetenv("CODEX_STAGE")
    } else {
      Sys.setenv(CODEX_STAGE = old_stage)
    }
  })
  sys.source(
    "R/codex_mlp_ensemble.R",
    envir = helper
  )
  rm("source", envir = helper)
  stopifnot(
    identical(canonical_fold_map, fold_map_function)
  )
  helper
}

extract_shallow_mlp <- function(n_rows) {
  candidate_paths <- c(
    file.path(
      "data_processed", "codex_behavioral_round",
      "mlp_oof.rds"
    ),
    file.path(
      "data_processed", "codex_componentwise_boost",
      "shallow_mlp_oof_rebuilt.rds"
    ),
    file.path(
      "data_processed", "codex_choice_set_geometry",
      "shallow_mlp_oof_rebuilt.rds"
    ),
    file.path(
      "data_processed", "codex_price_curve_shrinkage",
      "shallow_mlp_oof_rebuilt.rds"
    )
  )
  for (path in candidate_paths[file.exists(candidate_paths)]) {
    object <- readRDS(path)
    prediction <- NULL
    if (!is.null(object$oof)) {
      if (!is.null(object$oof[["h08_d0.100"]])) {
        prediction <- object$oof[["h08_d0.100"]]
      } else if (length(object$oof) == 1L) {
        prediction <- object$oof[[1L]]
      }
    } else if (!is.null(object$prediction)) {
      prediction <- object$prediction
    } else if (!is.null(object$mlp_oof)) {
      prediction <- object$mlp_oof
    }
    if (
      !is.null(prediction) &&
      identical(
        dim(as.matrix(prediction)),
        c(as.integer(n_rows), 4L)
      )
    ) {
      cat("Using frozen shallow-MLP OOF cache:", path, "\n")
      return(validate_probability(prediction, n_rows))
    }
  }
  stop(
    paste0(
      "The frozen shallow-MLP OOF cache was not found. ",
      "Run the already-verified componentwise or geometry runner ",
      "first; this SVM script will not silently rebuild a different ",
      "baseline."
    )
  )
}

canonical_baseline <- function(train, truth, saved) {
  base_mlogit <- validate_probability(
    saved$oof_mlogit, nrow(train)
  )
  original_xgb <- validate_probability(
    saved$oof_xgb, nrow(train)
  )
  shallow_mlp <- extract_shallow_mlp(nrow(train))
  baseline <- validate_probability(
    0.85 * (
      0.80 * base_mlogit +
        0.20 * original_xgb
    ) + 0.15 * shallow_mlp,
    nrow(train)
  )
  loss <- log_loss_matrix_local(truth, baseline)
  if (abs(loss - canonical_baseline_target) > 1e-6) {
    stop(sprintf(
      paste0(
        "Frozen flat baseline mismatch: got %.12f, ",
        "expected %.12f."
      ),
      loss,
      canonical_baseline_target
    ))
  }
  list(
    prediction = baseline,
    logloss = loss,
    mlogit = base_mlogit,
    xgboost = original_xgb,
    shallow_mlp = shallow_mlp
  )
}

svm_probability_prediction <- function(
    x_train,
    y_train,
    x_target,
    cost,
    seed) {
  stopifnot(
    nrow(x_train) == length(y_train),
    ncol(x_train) == ncol(x_target),
    all(is.finite(x_train)),
    all(is.finite(x_target)),
    identical(sort(unique(as.integer(y_train))), 1:4)
  )
  class_label <- factor(
    as.integer(y_train),
    levels = 1:4
  )
  gamma <- 1 / ncol(x_train)
  set.seed(as.integer(seed))
  started <- proc.time()[["elapsed"]]
  fit <- e1071::svm(
    x = x_train,
    y = class_label,
    type = "C-classification",
    kernel = "radial",
    cost = cost,
    gamma = gamma,
    scale = FALSE,
    probability = TRUE,
    shrinking = TRUE,
    tolerance = 0.001,
    cachesize = 1024
  )
  raw_class <- predict(
    fit,
    x_target,
    probability = TRUE,
    decision.values = FALSE
  )
  probability <- attr(raw_class, "probabilities")
  if (is.null(probability)) {
    stop("LIBSVM did not return probability estimates.")
  }
  probability <- as.matrix(probability)
  required_columns <- as.character(1:4)
  stopifnot(all(required_columns %in% colnames(probability)))
  probability <- probability[
    , required_columns, drop = FALSE
  ]
  elapsed <- proc.time()[["elapsed"]] - started
  list(
    prediction = validate_probability(
      probability, nrow(x_target)
    ),
    elapsed_seconds = elapsed,
    support_vectors = fit$tot.nSV,
    gamma = gamma
  )
}

select_inner_specification <- function(
    fitting,
    helper,
    reference,
    outer_seed,
    outer_fold,
    selected_cost_grid = cost_grid,
    selected_temperature_grid = temperature_grid,
    inner_folds = 3L) {
  inner_map <- balanced_fold_map(
    fitting$Case,
    inner_folds,
    as.integer(outer_seed + 30000L + outer_fold)
  )
  row_inner <- unname(
    inner_map[as.character(fitting$Case)]
  )
  stopifnot(!anyNA(row_inner))
  truth <- as.matrix(
    fitting[, paste0("Ch", 1:4), drop = FALSE]
  )
  base_predictions <- lapply(
    selected_cost_grid,
    function(x) matrix(
      NA_real_, nrow(fitting), 4L
    )
  )
  names(base_predictions) <- format(
    selected_cost_grid, scientific = FALSE
  )
  fit_rows <- list()

  for (inner_fold in seq_len(inner_folds)) {
    training_rows <- row_inner != inner_fold
    validation_rows <- row_inner == inner_fold
    inner_train <- fitting[
      training_rows, , drop = FALSE
    ]
    inner_valid <- fitting[
      validation_rows, , drop = FALSE
    ]
    scaler <- helper$continuous_scaler(inner_train)
    train_matrix <- helper$build_mlp_matrix(
      inner_train, reference, scaler
    )
    valid_matrix <- helper$build_mlp_matrix(
      inner_valid,
      reference,
      scaler,
      keep_columns = train_matrix$keep_columns
    )
    y_train <- max.col(
      as.matrix(
        inner_train[
          , paste0("Ch", 1:4), drop = FALSE
        ]
      )
    )

    for (cost_index in seq_along(selected_cost_grid)) {
      cost <- selected_cost_grid[[cost_index]]
      fitted <- svm_probability_prediction(
        train_matrix$x,
        y_train,
        valid_matrix$x,
        cost,
        seed = outer_seed +
          outer_fold * 1000L +
          inner_fold * 100L +
          cost_index
      )
      base_predictions[[cost_index]][
        validation_rows, 
      ] <- fitted$prediction
      fit_rows[[length(fit_rows) + 1L]] <- data.frame(
        outer_seed = outer_seed,
        outer_fold = outer_fold,
        inner_fold = inner_fold,
        cost = cost,
        gamma = fitted$gamma,
        n_features = ncol(train_matrix$x),
        support_vectors = fitted$support_vectors,
        elapsed_seconds = fitted$elapsed_seconds
      )
      cat(sprintf(
        paste0(
          "    inner %d/%d, C=%g: ",
          "%d SVs, %.1fs\n"
        ),
        inner_fold,
        inner_folds,
        cost,
        fitted$support_vectors,
        fitted$elapsed_seconds
      ))
      flush.console()
    }
  }

  curve_rows <- list()
  for (cost_index in seq_along(selected_cost_grid)) {
    prediction <- base_predictions[[cost_index]]
    stopifnot(!anyNA(prediction))
    for (
      temperature_index in
      seq_along(selected_temperature_grid)
    ) {
      temperature <-
        selected_temperature_grid[[temperature_index]]
      calibrated <- apply_temperature(
        prediction, temperature
      )
      curve_rows[[length(curve_rows) + 1L]] <- data.frame(
        cost = selected_cost_grid[[cost_index]],
        temperature = temperature,
        inner_logloss = log_loss_matrix_local(
          truth, calibrated
        )
      )
    }
  }
  curve <- do.call(rbind, curve_rows)
  curve$temperature_distance <- abs(
    curve$temperature - 1
  )
  curve <- curve[
    order(
      curve$inner_logloss,
      curve$temperature_distance,
      curve$cost
    ),
    ,
    drop = FALSE
  ]
  selected <- curve[1L, , drop = FALSE]
  list(
    cost = selected$cost[[1L]],
    temperature = selected$temperature[[1L]],
    inner_logloss = selected$inner_logloss[[1L]],
    curve = curve[
      , c("cost", "temperature", "inner_logloss"),
      drop = FALSE
    ],
    fits = do.call(rbind, fit_rows),
    inner_fold_map = inner_map
  )
}

fit_outer_svm <- function(
    fitting,
    validation,
    helper,
    reference,
    outer_seed,
    outer_fold,
    selected_cost_grid = cost_grid,
    selected_temperature_grid = temperature_grid,
    inner_folds = 3L) {
  selection <- select_inner_specification(
    fitting,
    helper,
    reference,
    outer_seed,
    outer_fold,
    selected_cost_grid,
    selected_temperature_grid,
    inner_folds
  )
  cat(sprintf(
    paste0(
      "  selected C=%g, temperature=%.2f, ",
      "inner loss %.9f\n"
    ),
    selection$cost,
    selection$temperature,
    selection$inner_logloss
  ))

  scaler <- helper$continuous_scaler(fitting)
  train_matrix <- helper$build_mlp_matrix(
    fitting, reference, scaler
  )
  validation_matrix <- helper$build_mlp_matrix(
    validation,
    reference,
    scaler,
    keep_columns = train_matrix$keep_columns
  )
  y_train <- max.col(as.matrix(
    fitting[, paste0("Ch", 1:4), drop = FALSE]
  ))
  refit <- svm_probability_prediction(
    train_matrix$x,
    y_train,
    validation_matrix$x,
    selection$cost,
    seed = outer_seed + outer_fold * 10000L + 999L
  )
  prediction <- apply_temperature(
    refit$prediction,
    selection$temperature
  )
  list(
    validation_no = validation$No,
    prediction = prediction,
    selected_cost = selection$cost,
    selected_temperature = selection$temperature,
    inner_logloss = selection$inner_logloss,
    inner_curve = selection$curve,
    inner_fits = selection$fits,
    outer_gamma = refit$gamma,
    outer_support_vectors = refit$support_vectors,
    outer_elapsed_seconds = refit$elapsed_seconds,
    feature_columns = train_matrix$keep_columns
  )
}

run_svm_oof <- function(
    train,
    fold_map,
    helper,
    reference,
    checkpoint_tag,
    outer_seed) {
  row_fold <- unname(
    fold_map[as.character(train$Case)]
  )
  stopifnot(!anyNA(row_fold))
  prediction <- matrix(
    NA_real_, nrow(train), 4L
  )
  fold_results <- vector("list", 5L)

  for (outer_fold in 1:5) {
    cat(sprintf(
      "\n=== %s: SVM outer fold %d/5 ===\n",
      checkpoint_tag, outer_fold
    ))
    validation_rows <- row_fold == outer_fold
    validation_no <- train$No[validation_rows]
    checkpoint <- file.path(
      output_dir,
      sprintf(
        "svm_%s_fold_%d.rds",
        checkpoint_tag,
        outer_fold
      )
    )
    result <- NULL
    if (file.exists(checkpoint)) {
      saved <- readRDS(checkpoint)
      if (
        identical(saved$experiment_id, experiment_id) &&
        identical(
          as.integer(saved$result$validation_no),
          as.integer(validation_no)
        ) &&
        identical(saved$cost_grid, cost_grid) &&
        identical(
          saved$temperature_grid,
          temperature_grid
        )
      ) {
        result <- saved$result
        cat("  loaded from checkpoint\n")
      }
    }
    if (is.null(result)) {
      result <- fit_outer_svm(
        train[!validation_rows, , drop = FALSE],
        train[validation_rows, , drop = FALSE],
        helper,
        reference,
        outer_seed,
        outer_fold
      )
      saveRDS(
        list(
          experiment_id = experiment_id,
          cost_grid = cost_grid,
          temperature_grid = temperature_grid,
          result = result
        ),
        checkpoint
      )
    }
    rows <- match(result$validation_no, train$No)
    stopifnot(
      !anyNA(rows),
      all(row_fold[rows] == outer_fold)
    )
    prediction[rows, ] <- result$prediction
    fold_results[[outer_fold]] <- result
    cat(sprintf(
      paste0(
        "  fold %d complete: C=%g, T=%.2f, ",
        "%d SVs\n"
      ),
      outer_fold,
      result$selected_cost,
      result$selected_temperature,
      result$outer_support_vectors
    ))
    flush.console()
  }
  list(
    prediction = validate_probability(
      prediction, nrow(train)
    ),
    fold_results = fold_results,
    fold_map = fold_map,
    row_fold = row_fold
  )
}

crossfit_blend <- function(
    truth,
    baseline,
    component,
    row_fold) {
  baseline <- validate_probability(
    baseline, nrow(truth)
  )
  component <- validate_probability(
    component, nrow(truth)
  )
  candidate <- matrix(
    NA_real_, nrow(truth), 4L
  )
  weight_rows <- list()
  for (fold in 1:5) {
    tuning_rows <- row_fold != fold
    validation_rows <- row_fold == fold
    losses <- vapply(
      blend_weight_grid,
      function(weight) {
        log_loss_matrix_local(
          truth[tuning_rows, , drop = FALSE],
          (1 - weight) *
            baseline[tuning_rows, , drop = FALSE] +
            weight *
            component[tuning_rows, , drop = FALSE]
        )
      },
      numeric(1)
    )
    best_index <- which.min(losses)
    best_weight <- blend_weight_grid[[best_index]]
    candidate[validation_rows, ] <-
      (1 - best_weight) *
        baseline[validation_rows, , drop = FALSE] +
        best_weight *
        component[validation_rows, , drop = FALSE]
    weight_rows[[fold]] <- data.frame(
      fold = fold,
      svm_weight = best_weight,
      tuning_logloss = losses[[best_index]]
    )
  }
  list(
    prediction = validate_probability(
      candidate, nrow(truth)
    ),
    weights = do.call(rbind, weight_rows)
  )
}

bootstrap_case_means <- function(
    respondent_gain,
    replicates,
    seed = 4821L,
    chunk_size = 1000L) {
  respondent_gain <- as.numeric(respondent_gain)
  stopifnot(
    length(respondent_gain) > 1L,
    all(is.finite(respondent_gain)),
    replicates > 0L
  )
  set.seed(as.integer(seed))
  output <- numeric(replicates)
  start <- 1L
  while (start <= replicates) {
    count <- min(
      chunk_size,
      replicates - start + 1L
    )
    index <- matrix(
      sample.int(
        length(respondent_gain),
        length(respondent_gain) * count,
        replace = TRUE
      ),
      nrow = length(respondent_gain),
      ncol = count
    )
    output[start:(start + count - 1L)] <-
      colMeans(matrix(
        respondent_gain[index],
        nrow = length(respondent_gain),
        ncol = count
      ))
    start <- start + count
  }
  output
}

respondent_gain <- function(
    truth,
    baseline,
    candidate,
    case) {
  row_gain <-
    row_log_loss_local(truth, baseline) -
    row_log_loss_local(truth, candidate)
  output <- tapply(row_gain, case, mean)
  output[order(as.integer(names(output)))]
}

bootstrap_summary <- function(
    truth,
    baseline,
    candidate,
    case,
    replicates,
    seed = 4821L) {
  case_gain <- respondent_gain(
    truth, baseline, candidate, case
  )
  bootstrap <- bootstrap_case_means(
    case_gain, replicates, seed
  )
  list(
    summary = data.frame(
      point_gain = mean(case_gain),
      bootstrap_mean = mean(bootstrap),
      bootstrap_sd = sd(bootstrap),
      lower_95 = unname(
        quantile(bootstrap, 0.025)
      ),
      upper_95 = unname(
        quantile(bootstrap, 0.975)
      ),
      lower_99 = unname(
        quantile(bootstrap, 0.005)
      ),
      upper_99 = unname(
        quantile(bootstrap, 0.995)
      ),
      win_rate = mean(bootstrap > 0),
      n_boot = length(bootstrap)
    ),
    respondent_gain = case_gain,
    bootstrap = bootstrap
  )
}

fold_selection_table <- function(
    fold_results,
    weights,
    seed) {
  do.call(
    rbind,
    lapply(seq_along(fold_results), function(fold) {
      result <- fold_results[[fold]]
      data.frame(
        seed = seed,
        fold = fold,
        selected_cost = result$selected_cost,
        selected_temperature =
          result$selected_temperature,
        inner_logloss = result$inner_logloss,
        outer_gamma = result$outer_gamma,
        outer_support_vectors =
          result$outer_support_vectors,
        outer_elapsed_seconds =
          result$outer_elapsed_seconds,
        svm_weight = weights$svm_weight[
          weights$fold == fold
        ]
      )
    })
  )
}

load_repeated_baseline <- function(
    train,
    truth,
    seed) {
  expected_map <- repeated_fold_map(
    unique(train$Case), seed
  )
  flat_candidate_paths <- c(
    file.path(
      "data_processed", "codex_componentwise_boost",
      sprintf("repeat_seed_%d.rds", seed)
    ),
    file.path(
      "data_processed", "codex_choice_set_geometry",
      sprintf("repeat_seed_%d.rds", seed)
    )
  )
  for (
    path in flat_candidate_paths[
      file.exists(flat_candidate_paths)
    ]
  ) {
    object <- readRDS(path)
    if (
      !is.null(object$fold_map) &&
      !is.null(object$baseline_prediction)
    ) {
      object_map <- object$fold_map
      map_match <- identical(
        as.integer(
          object_map[
            names(expected_map)
          ]
        ),
        as.integer(expected_map)
      )
      if (map_match) {
        baseline <- validate_probability(
          object$baseline_prediction,
          nrow(train)
        )
        cat(
          "Using verified repeated baseline cache:",
          path, "\n"
        )
        return(list(
          fold_map = object_map,
          prediction = baseline,
          logloss = log_loss_matrix_local(
            truth, baseline
          ),
          source = path
        ))
      }
    }
  }

  repeat_path <- file.path(
    "data_processed", "codex_repeat_cv",
    sprintf("repeat_seed_%d.rds", seed)
  )
  if (file.exists(repeat_path)) {
    object <- readRDS(repeat_path)
    if (
      !is.null(object$fold_map) &&
      !is.null(object$components)
    ) {
      object_map <- object$fold_map
      map_match <- identical(
        as.integer(
          object_map[
            names(expected_map)
          ]
        ),
        as.integer(expected_map)
      )
      required <- c(
        "mlogit", "original_xgb", "shallow_mlp"
      )
      if (
        map_match &&
        all(required %in% names(object$components))
      ) {
        baseline <- validate_probability(
          0.85 * (
            0.80 * object$components$mlogit +
              0.20 *
              object$components$original_xgb
          ) + 0.15 *
            object$components$shallow_mlp,
          nrow(train)
        )
        cat(
          "Using verified repeated component cache:",
          repeat_path, "\n"
        )
        return(list(
          fold_map = object_map,
          prediction = baseline,
          logloss = log_loss_matrix_local(
            truth, baseline
          ),
          source = repeat_path
        ))
      }
    }
  }

  stop(
    paste0(
      "No verified baseline cache exists for repeated seed ",
      seed,
      ". The earlier componentwise repeated-CV run should ",
      "have created it. Stopping instead of refitting a ",
      "potentially mismatched baseline."
    )
  )
}

run_smoke_test <- function(train, helper, reference) {
  saved <- readRDS(file.path(
    "data_processed", "oof_ensemble_v10.rds"
  ))
  fold_map <- canonical_fold_map(train, saved)
  outer_fold <- 1L
  fitting_cases <- as.integer(
    names(fold_map)[fold_map != outer_fold]
  )
  validation_cases <- as.integer(
    names(fold_map)[fold_map == outer_fold]
  )
  fitting_cases <- fitting_cases[
    seq_len(min(240L, length(fitting_cases)))
  ]
  validation_cases <- validation_cases[
    seq_len(min(60L, length(validation_cases)))
  ]
  fitting <- train[
    train$Case %in% fitting_cases, , drop = FALSE
  ]
  validation <- train[
    train$Case %in% validation_cases, , drop = FALSE
  ]
  cat(
    "SMOKE MODE: 240 fitting respondents, ",
    "60 validation respondents, two inner folds.\n",
    sep = ""
  )
  result <- fit_outer_svm(
    fitting,
    validation,
    helper,
    reference,
    outer_seed = 4821L,
    outer_fold = 1L,
    selected_cost_grid = c(0.5, 2),
    selected_temperature_grid = c(1, 1.25),
    inner_folds = 2L
  )
  truth <- as.matrix(
    validation[, paste0("Ch", 1:4), drop = FALSE]
  )
  result$smoke_logloss <- log_loss_matrix_local(
    truth, result$prediction
  )
  path <- file.path(output_dir, "smoke_result.rds")
  saveRDS(
    list(
      experiment_id = experiment_id,
      result = result
    ),
    path
  )
  cat(sprintf(
    paste0(
      "\nSmoke test completed successfully: %s\n",
      "Smoke SVM log loss: %.6f\n"
    ),
    path,
    result$smoke_logloss
  ))
  invisible(result)
}

run_experiment <- function() {
  train <- read.csv(
    file.path("csv files", "train.csv")
  )
  train <- train[order(train$No), , drop = FALSE]
  rownames(train) <- NULL
  test <- read.csv(
    file.path("csv files", "test.csv")
  )
  truth <- as.matrix(
    train[, paste0("Ch", 1:4), drop = FALSE]
  )
  stopifnot(
    nrow(train) == 21565L,
    nrow(test) == 4997L,
    length(unique(train$Case)) == 1135L,
    all(table(train$Case) == 19L),
    all(rowSums(truth) == 1L)
  )

  helper <- mlp_helper_environment()
  reference <- helper$feature_reference(train)
  cat("\nRunning", experiment_id, "\n")
  cat(
    "Frozen baseline target:",
    sprintf("%.12f", canonical_baseline_target),
    "\n"
  )
  cat(
    "Secondary submitted-CV reference:",
    sprintf("%.12f", submitted_crossfit_reference),
    "\n"
  )

  if (smoke_mode) {
    return(run_smoke_test(
      train, helper, reference
    ))
  }

  saved <- readRDS(file.path(
    "data_processed", "oof_ensemble_v10.rds"
  ))
  fold_map <- canonical_fold_map(train, saved)
  baseline <- canonical_baseline(
    train, truth, saved
  )
  svm_oof <- run_svm_oof(
    train,
    fold_map,
    helper,
    reference,
    checkpoint_tag = "seed_4821",
    outer_seed = 4821L
  )
  blended <- crossfit_blend(
    truth,
    baseline$prediction,
    svm_oof$prediction,
    svm_oof$row_fold
  )
  candidate_loss <- log_loss_matrix_local(
    truth, blended$prediction
  )
  canonical_bootstrap <- bootstrap_summary(
    truth,
    baseline$prediction,
    blended$prediction,
    train$Case,
    bootstrap_replicates,
    seed = 4821L
  )
  canonical_summary <- cbind(
    data.frame(
      experiment = experiment_id,
      stage = "canonical",
      baseline_logloss = baseline$logloss,
      svm_component_logloss =
        log_loss_matrix_local(
          truth, svm_oof$prediction
        ),
      candidate_logloss = candidate_loss
    ),
    canonical_bootstrap$summary
  )
  canonical_summary$canonical_pass <-
    canonical_summary$point_gain > 0 &
    canonical_summary$lower_95 > 0
  canonical_summary$near_miss <-
    canonical_summary$point_gain > 0 &
    canonical_summary$lower_95 <= 0 &
    canonical_summary$lower_95 >=
      near_miss_lower_limit
  canonical_selection <- fold_selection_table(
    svm_oof$fold_results,
    blended$weights,
    seed = 4821L
  )

  write.csv(
    canonical_summary,
    file.path(output_dir, "canonical_summary.csv"),
    row.names = FALSE
  )
  write.csv(
    canonical_selection,
    file.path(
      output_dir, "canonical_fold_selection.csv"
    ),
    row.names = FALSE
  )
  saveRDS(
    list(
      experiment_id = experiment_id,
      summary = canonical_summary,
      baseline_prediction = baseline$prediction,
      svm_prediction = svm_oof$prediction,
      candidate_prediction = blended$prediction,
      fold_map = fold_map,
      fold_results = svm_oof$fold_results,
      weights = blended$weights,
      respondent_gain =
        canonical_bootstrap$respondent_gain,
      bootstrap = canonical_bootstrap$bootstrap
    ),
    file.path(output_dir, "canonical_result.rds")
  )
  cat("\nCanonical result:\n")
  print(canonical_summary, digits = 9)
  cat("\nFold selections:\n")
  print(canonical_selection, digits = 7)

  run_repeated <- repeated_mode == "always" ||
    (
      repeated_mode == "auto" &&
      (
        isTRUE(canonical_summary$canonical_pass) ||
        isTRUE(canonical_summary$near_miss)
      )
    )

  if (!run_repeated) {
    reason <- if (repeated_mode == "never") {
      "Repeated CV disabled by SVM_DIVERSITY_REPEATED=never."
    } else {
      paste0(
        "Repeated CV not triggered: canonical result ",
        "was neither a pass nor the pre-registered near miss."
      )
    }
    verdict <- paste(
      "REJECT calibrated RBF-SVM diversity component.",
      reason,
      sprintf(
        "Canonical gain %.9f; 95%% CI [%.9f, %.9f].",
        canonical_summary$point_gain,
        canonical_summary$lower_95,
        canonical_summary$upper_95
      )
    )
    writeLines(
      verdict,
      file.path(output_dir, "verdict.txt")
    )
    cat("\n", verdict, "\n", sep = "")
    return(invisible(list(
      canonical = canonical_summary,
      repeated = NULL,
      verdict = verdict
    )))
  }

  cat(
    "\nRepeated-CV escalation activated (",
    repeated_mode,
    ").\n",
    sep = ""
  )
  repeat_names <- c(
    "4821", as.character(additional_seeds)
  )
  case_levels <- sort(unique(train$Case))
  case_gain_matrix <- matrix(
    NA_real_,
    nrow = length(case_levels),
    ncol = length(repeat_names),
    dimnames = list(
      as.character(case_levels), repeat_names
    )
  )
  case_gain_matrix[, "4821"] <-
    canonical_bootstrap$respondent_gain
  repeat_rows <- list(
    `4821` = data.frame(
      seed = 4821L,
      baseline_logloss = baseline$logloss,
      svm_component_logloss =
        log_loss_matrix_local(
          truth, svm_oof$prediction
        ),
      candidate_logloss = candidate_loss,
      gain = baseline$logloss - candidate_loss
    )
  )
  selection_rows <- list(
    `4821` = canonical_selection
  )

  for (seed_index in seq_along(additional_seeds)) {
    seed <- additional_seeds[[seed_index]]
    seed_name <- as.character(seed)
    cat(sprintf(
      "\n######## repeated CV seed %d (%d/5) ########\n",
      seed,
      seed_index
    ))
    repeat_baseline <- load_repeated_baseline(
      train, truth, seed
    )
    repeat_svm <- run_svm_oof(
      train,
      repeat_baseline$fold_map,
      helper,
      reference,
      checkpoint_tag = paste0("seed_", seed),
      outer_seed = seed
    )
    repeat_blend <- crossfit_blend(
      truth,
      repeat_baseline$prediction,
      repeat_svm$prediction,
      repeat_svm$row_fold
    )
    repeat_candidate_loss <- log_loss_matrix_local(
      truth, repeat_blend$prediction
    )
    repeat_case_gain <- respondent_gain(
      truth,
      repeat_baseline$prediction,
      repeat_blend$prediction,
      train$Case
    )
    case_gain_matrix[, seed_name] <-
      repeat_case_gain
    repeat_rows[[seed_name]] <- data.frame(
      seed = seed,
      baseline_logloss = repeat_baseline$logloss,
      svm_component_logloss =
        log_loss_matrix_local(
          truth, repeat_svm$prediction
        ),
      candidate_logloss = repeat_candidate_loss,
      gain = repeat_baseline$logloss -
        repeat_candidate_loss
    )
    selection_rows[[seed_name]] <-
      fold_selection_table(
        repeat_svm$fold_results,
        repeat_blend$weights,
        seed
      )
    saveRDS(
      list(
        experiment_id = experiment_id,
        seed = seed,
        baseline_source = repeat_baseline$source,
        fold_map = repeat_baseline$fold_map,
        baseline_prediction =
          repeat_baseline$prediction,
        svm_prediction = repeat_svm$prediction,
        candidate_prediction =
          repeat_blend$prediction,
        weights = repeat_blend$weights,
        fold_results = repeat_svm$fold_results,
        respondent_gain = repeat_case_gain
      ),
      file.path(
        output_dir,
        sprintf("repeat_result_%d.rds", seed)
      )
    )
  }

  repeated_by_seed <- do.call(rbind, repeat_rows)
  average_case_gain <- rowMeans(case_gain_matrix)
  pooled_bootstrap <- bootstrap_case_means(
    average_case_gain,
    bootstrap_replicates,
    seed = 4821L
  )
  repeated_summary <- data.frame(
    experiment = experiment_id,
    stage = "repeated_cv",
    point_gain = mean(average_case_gain),
    bootstrap_mean = mean(pooled_bootstrap),
    bootstrap_sd = sd(pooled_bootstrap),
    lower_95 = unname(
      quantile(pooled_bootstrap, 0.025)
    ),
    upper_95 = unname(
      quantile(pooled_bootstrap, 0.975)
    ),
    lower_99 = unname(
      quantile(pooled_bootstrap, 0.005)
    ),
    upper_99 = unname(
      quantile(pooled_bootstrap, 0.995)
    ),
    win_rate = mean(pooled_bootstrap > 0),
    positive_repeats =
      sum(repeated_by_seed$gain > 0),
    n_repeats = nrow(repeated_by_seed),
    n_boot = length(pooled_bootstrap)
  )
  repeated_summary$promote <-
    repeated_summary$point_gain > 0 &
    repeated_summary$lower_95 > 0 &
    repeated_summary$positive_repeats >= 5L

  write.csv(
    repeated_by_seed,
    file.path(output_dir, "repeated_cv_by_seed.csv"),
    row.names = FALSE
  )
  write.csv(
    repeated_summary,
    file.path(output_dir, "repeated_cv_summary.csv"),
    row.names = FALSE
  )
  write.csv(
    do.call(rbind, selection_rows),
    file.path(
      output_dir, "repeated_cv_fold_selection.csv"
    ),
    row.names = FALSE
  )
  saveRDS(
    list(
      experiment_id = experiment_id,
      repeated_by_seed = repeated_by_seed,
      repeated_summary = repeated_summary,
      case_gain_matrix = case_gain_matrix,
      pooled_bootstrap = pooled_bootstrap
    ),
    file.path(output_dir, "repeated_cv_result.rds")
  )

  if (isTRUE(repeated_summary$promote)) {
    verdict <- paste0(
      "PROMOTE calibrated RBF-SVM diversity component for ",
      "a separate full-data build audit. Do not submit until ",
      "that artifact is reproduced and checked."
    )
  } else {
    verdict <- paste0(
      "REJECT calibrated RBF-SVM diversity component; retain ",
      "the existing ensemble_v11 + shallow MLP submission."
    )
  }
  writeLines(
    verdict,
    file.path(output_dir, "verdict.txt")
  )
  cat("\nRepeated-CV result:\n")
  print(repeated_summary, digits = 9)
  cat("\n", verdict, "\n", sep = "")
  invisible(list(
    canonical = canonical_summary,
    repeated = repeated_summary,
    verdict = verdict
  ))
}

run_experiment()
