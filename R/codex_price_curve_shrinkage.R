# Curvature shrinkage of m8trpg's saturated main-price curve.
#
# Pre-registration:
#   codex_price_curve_shrinkage_preregister.md
#
# Run from the repository root:
#   source("R/codex_price_curve_shrinkage.R")
#
# Smoke test:
#   Sys.setenv(PRICE_CURVE_SMOKE = "1")
#   source("R/codex_price_curve_shrinkage.R")
#   Sys.unsetenv("PRICE_CURVE_SMOKE")
#
# The complete run performs canonical respondent-grouped five-fold CV. It
# automatically escalates to five additional fold seeds when the canonical
# result passes or is a pre-registered near miss.

options(stringsAsFactors = FALSE)

required_packages <- c("mlogit", "dfidx", "xgboost", "nnet")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
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

suppressPackageStartupMessages({
  library(mlogit)
  library(dfidx)
  library(xgboost)
  library(nnet)
})

source("R/codex_shared_utility_common.R")

experiment_id <- "price_curve_curvature_shrink_v1"
output_dir <- file.path(
  "data_processed", "codex_price_curve_shrinkage"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

smoke_mode <- identical(Sys.getenv("PRICE_CURVE_SMOKE", "0"), "1")
repeated_mode <- tolower(
  Sys.getenv("PRICE_CURVE_REPEATED", "auto")
)
stopifnot(repeated_mode %in% c("auto", "never", "always"))

bootstrap_replicates <- if (smoke_mode) {
  200L
} else {
  as.integer(Sys.getenv("PRICE_CURVE_N_BOOT", "100000"))
}
near_miss_lower_limit <- -0.00025
additional_seeds <- c(
  1907L, 2719L, 6151L, 8293L, 104729L
)
canonical_baseline_target <- 1.143686618134879

input_files <- c(
  file.path("csv files", "train.csv"),
  file.path("csv files", "test.csv"),
  file.path("data_processed", "oof_ensemble_v10.rds")
)
missing_files <- input_files[!file.exists(input_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing required project file(s):\n  ",
    paste(missing_files, collapse = "\n  "),
    "\nRun this script from the kagglecomp repository root."
  )
}

price_lambda_grid <- c(
  0, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, Inf
)

log_loss_matrix_local <- function(truth, prediction) {
  prediction <- as.matrix(prediction)
  prediction <- prediction / rowSums(prediction)
  prediction <- pmin(
    pmax(prediction, 1e-15), 1 - 1e-15
  )
  -mean(rowSums(as.matrix(truth) * log(prediction)))
}

row_log_loss_local <- function(truth, prediction) {
  prediction <- as.matrix(prediction)
  prediction <- prediction / rowSums(prediction)
  prediction <- pmin(
    pmax(prediction, 1e-15), 1 - 1e-15
  )
  -rowSums(as.matrix(truth) * log(prediction))
}

validate_probability <- function(prediction, n_rows = NULL) {
  prediction <- as.matrix(prediction)
  if (!is.null(n_rows)) {
    stopifnot(
      identical(dim(prediction), c(as.integer(n_rows), 4L))
    )
  } else {
    stopifnot(ncol(prediction) == 4L)
  }
  stopifnot(
    !anyNA(prediction),
    all(is.finite(prediction)),
    all(prediction > 0)
  )
  prediction <- prediction / rowSums(prediction)
  stopifnot(
    max(abs(rowSums(prediction) - 1)) < 1e-10
  )
  prediction
}

sort_choice_long_local <- function(data) {
  data <- as.data.frame(data)
  data <- data[
    order(data$No, as.integer(data$alt)), ,
    drop = FALSE
  ]
  rownames(data) <- NULL
  stopifnot(
    nrow(data) %% 4L == 0L,
    identical(
      as.integer(data$alt),
      rep(1:4, times = nrow(data) / 4L)
    ),
    all(table(data$No) == 4L)
  )
  data
}

price_term_names <- paste0("Pr_lvl", 2:12)

extract_price_curve <- function(coefficient) {
  stopifnot(all(price_term_names %in% names(coefficient)))
  curve <- c(0, as.numeric(coefficient[price_term_names]))
  names(curve) <- as.character(1:12)
  stopifnot(length(curve) == 12L, all(is.finite(curve)))
  curve
}

is_shape_valid <- function(curve, tolerance = 1e-8) {
  curve <- as.numeric(curve)
  length(curve) == 12L &&
    abs(curve[[1L]]) <= tolerance &&
    all(diff(curve) <= tolerance) &&
    all(diff(curve, differences = 2L) >= -tolerance)
}

smooth_price_curve <- function(raw_curve, lambda) {
  raw_curve <- as.numeric(raw_curve)
  stopifnot(
    length(raw_curve) == 12L,
    abs(raw_curve[[1L]]) < 1e-10,
    all(is.finite(raw_curve)),
    length(lambda) == 1L,
    lambda >= 0
  )
  if (is.infinite(lambda)) {
    level_offset <- 0:11
    slope <- sum(
      level_offset * raw_curve
    ) / sum(level_offset^2)
    result <- slope * level_offset
  } else if (lambda == 0) {
    result <- raw_curve
  } else {
    second_difference <- diff(diag(12L), differences = 2L)
    free_operator <- second_difference[, 2:12, drop = FALSE]
    system <- diag(11L) +
      lambda * crossprod(free_operator)
    result <- c(
      0,
      as.numeric(solve(system, raw_curve[2:12]))
    )
  }
  names(result) <- as.character(1:12)
  result
}

lambda_label <- function(lambda) {
  if (is.infinite(lambda)) "Inf" else format(lambda, trim = TRUE)
}

fit_choice_model <- function(train_long) {
  train_long <- sort_choice_long_local(train_long)
  scaler <- choice_scaler(train_long)
  features <- make_m8trpg_features(
    train_long, scaler$ctr, scaler$scl, "none"
  )
  formula <- m8trpg_formula("none")
  environment(formula) <- environment()
  model <- mlogit(
    formula,
    data = features,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen"
  )
  coefficient <- coef(model)
  stopifnot(
    all(is.finite(coefficient)),
    all(price_term_names %in% names(coefficient))
  )
  list(
    model = model,
    scaler = scaler,
    formula = formula,
    coefficient = coefficient,
    price_curve = extract_price_curve(coefficient)
  )
}

predict_choice_model <- function(
    fitted,
    long_df,
    replacement_curve = NULL) {
  long_df <- sort_choice_long_local(long_df)
  features <- make_m8trpg_features(
    long_df,
    fitted$scaler$ctr,
    fitted$scaler$scl,
    "none"
  )
  indexed <- dfidx(
    features,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen"
  )
  class(indexed) <- c("dfidx_mlogit", class(indexed))
  model_frame <- model.frame(
    indexed, fitted$formula, balanced = TRUE
  )
  design <- model.matrix(model_frame, rhs = 1:3)
  coefficient <- fitted$coefficient
  stopifnot(all(names(coefficient) %in% colnames(design)))
  raw_margin <- as.numeric(
    design[, names(coefficient), drop = FALSE] %*%
      coefficient
  )

  task_map <- unique(features[, c("chid", "No")])
  task_map <- task_map[order(task_map$No), , drop = FALSE]
  margin_chid <- as.character(dfidx::idx(model_frame, 1))
  margin_alt <- as.integer(
    as.character(dfidx::idx(model_frame, 2))
  )
  margin_order <- order(
    match(margin_chid, task_map$chid),
    margin_alt
  )
  stopifnot(
    identical(
      margin_chid[margin_order],
      rep(task_map$chid, each = 4L)
    ),
    identical(
      margin_alt[margin_order],
      rep(1:4, times = nrow(task_map))
    )
  )
  margin <- raw_margin[margin_order]
  if (!is.null(replacement_curve)) {
    replacement_curve <- as.numeric(replacement_curve)
    stopifnot(
      length(replacement_curve) == 12L,
      is_shape_valid(replacement_curve)
    )
    price <- as.integer(features$Price_num)
    stopifnot(
      length(price) == length(margin),
      all(price >= 0L),
      all(price <= 12L)
    )
    original_lookup <- c(0, fitted$price_curve)
    replacement_lookup <- c(0, replacement_curve)
    margin <- margin -
      original_lookup[price + 1L] +
      replacement_lookup[price + 1L]
  }
  list(
    pred = validate_probability(
      softmax_margins(margin), nrow(task_map)
    ),
    no = task_map$No,
    coefficient = coefficient,
    price_curve = if (is.null(replacement_curve)) {
      fitted$price_curve
    } else {
      replacement_curve
    }
  )
}

inner_fold_map <- function(cases, outer_seed, outer_fold) {
  cases <- sort(unique(as.integer(cases)))
  set.seed(as.integer(outer_seed + 10000L + outer_fold))
  fold <- sample(rep(1:4, length.out = length(cases)))
  names(fold) <- as.character(cases)
  stopifnot(
    length(fold) == length(cases),
    all(table(fold) == length(cases) / 4L)
  )
  fold
}

select_price_lambda <- function(
    fitting_long,
    outer_seed,
    outer_fold) {
  fitting_long <- sort_choice_long_local(fitting_long)
  cases <- sort(unique(fitting_long$Case))
  fold_map <- inner_fold_map(cases, outer_seed, outer_fold)
  row_fold <- unname(
    fold_map[as.character(fitting_long$Case)]
  )
  stopifnot(!anyNA(row_fold))

  truth <- matrix(
    as.integer(fitting_long$chosen),
    ncol = 4L,
    byrow = TRUE
  )
  truth_no <- unique(fitting_long$No)
  stopifnot(
    nrow(truth) == length(truth_no),
    all(rowSums(truth) == 1L)
  )
  lambda_names <- vapply(
    price_lambda_grid, lambda_label, character(1)
  )
  prediction <- lapply(
    price_lambda_grid,
    function(unused) {
      matrix(
        NA_real_,
        nrow = nrow(truth),
        ncol = 4L
      )
    }
  )
  names(prediction) <- lambda_names
  eligibility <- matrix(
    FALSE,
    nrow = 4L,
    ncol = length(price_lambda_grid),
    dimnames = list(
      paste0("inner_", 1:4), lambda_names
    )
  )

  for (inner_fold in 1:4) {
    inner_train <- fitting_long[
      row_fold != inner_fold, , drop = FALSE
    ]
    inner_validation <- fitting_long[
      row_fold == inner_fold, , drop = FALSE
    ]
    model <- fit_choice_model(inner_train)
    validation_no <- unique(inner_validation$No)
    validation_rows <- match(validation_no, truth_no)
    stopifnot(
      !anyNA(validation_rows),
      length(intersect(
        unique(inner_train$Case),
        unique(inner_validation$Case)
      )) == 0L
    )
    for (lambda_index in seq_along(price_lambda_grid)) {
      curve <- smooth_price_curve(
        model$price_curve,
        price_lambda_grid[[lambda_index]]
      )
      eligibility[inner_fold, lambda_index] <-
        is_shape_valid(curve)
      if (!eligibility[inner_fold, lambda_index]) next
      predicted <- predict_choice_model(
        model,
        inner_validation,
        replacement_curve = curve
      )
      stopifnot(identical(
        as.integer(predicted$no),
        as.integer(validation_no)
      ))
      prediction[[lambda_index]][validation_rows, ] <-
        predicted$pred
    }
  }

  loss <- rep(Inf, length(price_lambda_grid))
  for (lambda_index in seq_along(price_lambda_grid)) {
    candidate <- prediction[[lambda_index]]
    if (
      all(eligibility[, lambda_index]) &&
      !anyNA(candidate)
    ) {
      loss[[lambda_index]] <- log_loss_matrix_local(
        truth, candidate
      )
    }
  }
  stopifnot(any(is.finite(loss)))
  best_loss <- min(loss)
  tied <- which(
    is.finite(loss) &
      loss <= best_loss + 1e-12
  )
  selected_index <- tied[
    which.max(price_lambda_grid[tied])
  ]
  list(
    lambda = price_lambda_grid[[selected_index]],
    loss = data.frame(
      lambda = lambda_names,
      inner_logloss = loss,
      eligible_all_folds =
        colSums(eligibility) == 4L
    ),
    eligibility = eligibility
  )
}

canonical_fold_map <- function(train, saved) {
  fold_map <- saved$fold_of_case
  stopifnot(
    length(fold_map) == length(unique(train$Case)),
    !anyNA(
      fold_map[as.character(unique(train$Case))]
    ),
    identical(
      sort(unique(as.integer(fold_map))), 1:5
    )
  )
  fold_map
}

repeated_fold_map <- function(cases, seed) {
  set.seed(seed)
  fold <- sample(rep(1:5, length.out = length(cases)))
  names(fold) <- as.character(cases)
  stopifnot(all(table(fold) == 227L))
  fold
}

run_price_curve_fold <- function(
    train_long,
    train_wide,
    fold_map,
    outer_fold,
    outer_seed) {
  row_fold <- unname(
    fold_map[as.character(train_wide$Case)]
  )
  long_fold <- unname(
    fold_map[as.character(train_long$Case)]
  )
  stopifnot(!anyNA(row_fold), !anyNA(long_fold))

  fitting_long <- train_long[
    long_fold != outer_fold, , drop = FALSE
  ]
  validation_long <- train_long[
    long_fold == outer_fold, , drop = FALSE
  ]
  validation_wide <- train_wide[
    row_fold == outer_fold, , drop = FALSE
  ]
  stopifnot(
    length(intersect(
      unique(fitting_long$Case),
      unique(validation_long$Case)
    )) == 0L
  )

  cat("  nested respondent-grouped price-penalty selection\n")
  selection <- select_price_lambda(
    fitting_long,
    outer_seed = outer_seed,
    outer_fold = outer_fold
  )
  cat(
    "  selected lambda:",
    lambda_label(selection$lambda), "\n"
  )

  cat("  fitting outer plain m8trpg\n")
  base_model <- fit_choice_model(fitting_long)
  smoothed_curve <- smooth_price_curve(
    base_model$price_curve,
    selection$lambda
  )
  if (!is_shape_valid(smoothed_curve)) {
    stop(
      "The inner-selected lambda did not produce an eligible ",
      "monotone-convex curve in the outer fit."
    )
  }
  base_prediction <- predict_choice_model(
    base_model, validation_long
  )
  candidate_prediction <- predict_choice_model(
    base_model,
    validation_long,
    replacement_curve = smoothed_curve
  )
  stopifnot(
    identical(
      as.integer(base_prediction$no),
      as.integer(candidate_prediction$no)
    ),
    identical(
      as.integer(base_prediction$no),
      as.integer(validation_wide$No)
    )
  )

  list(
    outer_fold = outer_fold,
    outer_seed = outer_seed,
    validation_no = base_prediction$no,
    baseline_prediction = base_prediction$pred,
    candidate_prediction = candidate_prediction$pred,
    selected_lambda = selection$lambda,
    inner_loss = selection$loss,
    inner_eligibility = selection$eligibility,
    original_price_curve = base_model$price_curve,
    smoothed_price_curve = smoothed_curve,
    coefficient_count = length(base_model$coefficient)
  )
}

run_price_curve_oof <- function(
    train,
    train_long,
    fold_map,
    checkpoint_tag,
    outer_seed) {
  row_fold <- unname(
    fold_map[as.character(train$Case)]
  )
  baseline <- matrix(
    NA_real_, nrow(train), 4L
  )
  candidate <- baseline
  fold_results <- vector("list", 5L)
  folds_to_run <- if (smoke_mode) 1L else 1:5

  for (fold in folds_to_run) {
    cat(sprintf(
      "\n=== %s: price-curve outer fold %d/5 ===\n",
      checkpoint_tag, fold
    ))
    path <- file.path(
      output_dir,
      sprintf(
        "price_curve_%s_fold_%d.rds",
        checkpoint_tag, fold
      )
    )
    validation_rows <- row_fold == fold
    result <- NULL
    if (file.exists(path)) {
      saved <- readRDS(path)
      if (
        identical(saved$experiment_id, experiment_id) &&
        identical(
          as.integer(saved$result$outer_seed),
          as.integer(outer_seed)
        ) &&
        identical(
          as.integer(saved$result$validation_no),
          as.integer(train$No[validation_rows])
        )
      ) {
        result <- saved$result
        cat("  loaded from checkpoint\n")
      }
    }
    if (is.null(result)) {
      result <- run_price_curve_fold(
        train_long,
        train,
        fold_map,
        fold,
        outer_seed
      )
      saveRDS(
        list(
          experiment_id = experiment_id,
          result = result
        ),
        path
      )
    }
    rows <- match(result$validation_no, train$No)
    stopifnot(
      !anyNA(rows),
      all(row_fold[rows] == fold)
    )
    baseline[rows, ] <- result$baseline_prediction
    candidate[rows, ] <- result$candidate_prediction
    fold_results[[fold]] <- result
    cat("  fold", fold, "complete\n")
    flush.console()
    invisible(gc())
  }

  if (smoke_mode) {
    return(list(
      baseline = baseline,
      candidate = candidate,
      fold_results = fold_results
    ))
  }
  stopifnot(!anyNA(baseline), !anyNA(candidate))
  list(
    baseline = validate_probability(
      baseline, nrow(train)
    ),
    candidate = validate_probability(
      candidate, nrow(train)
    ),
    fold_results = fold_results
  )
}

load_mlp_from_cache <- function(n_rows) {
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
      output_dir, "shallow_mlp_oof_rebuilt.rds"
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
    }
    if (
      !is.null(prediction) &&
      identical(
        dim(as.matrix(prediction)),
        c(n_rows, 4L)
      )
    ) {
      cat("Using frozen shallow-MLP OOF cache:", path, "\n")
      return(validate_probability(prediction, n_rows))
    }
  }
  NULL
}

mlp_helper_environment <- function() {
  helper <- new.env(parent = globalenv())
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
    "R/codex_mlp_ensemble.R", envir = helper
  )
  helper
}

fit_shallow_mlp_oof <- function(
    train,
    fold_map,
    repeat_index,
    checkpoint_tag) {
  helper <- mlp_helper_environment()
  reference <- helper$feature_reference(train)
  row_fold <- unname(
    fold_map[as.character(train$Case)]
  )
  prediction <- matrix(
    NA_real_, nrow(train), 4L
  )

  for (fold in 1:5) {
    path <- file.path(
      output_dir,
      sprintf(
        "mlp_%s_fold_%d.rds",
        checkpoint_tag, fold
      )
    )
    validation_rows <- row_fold == fold
    if (file.exists(path)) {
      saved <- readRDS(path)
      if (identical(
        as.integer(saved$validation_no),
        as.integer(train$No[validation_rows])
      )) {
        prediction[validation_rows, ] <-
          saved$prediction
        cat("  shallow MLP fold", fold,
            "loaded from checkpoint\n")
        next
      }
    }

    fitting <- train[
      !validation_rows, , drop = FALSE
    ]
    validation <- train[
      validation_rows, , drop = FALSE
    ]
    scaler <- helper$continuous_scaler(fitting)
    fitting_matrix <- helper$build_mlp_matrix(
      fitting, reference, scaler
    )
    validation_matrix <- helper$build_mlp_matrix(
      validation,
      reference,
      scaler,
      keep_columns = fitting_matrix$keep_columns
    )
    fitted <- helper$fit_mlp_average(
      fitting_matrix$x,
      as.matrix(
        fitting[, paste0("Ch", 1:4), drop = FALSE]
      ),
      validation_matrix$x,
      size = 8L,
      decay = 0.1,
      seeds = 4821L + 0:4 +
        repeat_index * 10000L + fold * 100L,
      max_iterations = 200L
    )
    prediction[validation_rows, ] <- fitted$pred
    saveRDS(
      list(
        experiment_id = experiment_id,
        validation_no = validation$No,
        prediction = fitted$pred,
        fits = fitted$fits
      ),
      path
    )
    cat("  shallow MLP fold", fold, "complete\n")
    flush.console()
  }
  validate_probability(prediction, nrow(train))
}

fit_original_xgb_oof <- function(
    train,
    fold_map,
    checkpoint_tag) {
  row_fold <- unname(
    fold_map[as.character(train$Case)]
  )
  prediction <- matrix(
    NA_real_, nrow(train), 4L
  )
  for (fold in 1:5) {
    path <- file.path(
      output_dir,
      sprintf(
        "xgb_%s_fold_%d.rds",
        checkpoint_tag, fold
      )
    )
    validation_rows <- row_fold == fold
    if (file.exists(path)) {
      saved <- readRDS(path)
      if (identical(
        as.integer(saved$validation_no),
        as.integer(train$No[validation_rows])
      )) {
        prediction[validation_rows, ] <-
          saved$prediction
        cat("  original xgboost fold", fold,
            "loaded from checkpoint\n")
        next
      }
    }
    fitting <- train[
      !validation_rows, , drop = FALSE
    ]
    validation <- train[
      validation_rows, , drop = FALSE
    ]
    label <- max.col(
      fitting[
        , paste0("Ch", 1:4), drop = FALSE
      ]
    ) - 1L
    model <- xgb.train(
      params = list(
        objective = "multi:softprob",
        num_class = 4L,
        eval_metric = "mlogloss",
        eta = 0.1,
        max_depth = 4L,
        subsample = 0.8,
        colsample_bytree = 0.8,
        tree_method = "hist",
        seed = 4821L,
        nthread = 1L
      ),
      data = xgb.DMatrix(
        wide_feature_matrix(fitting),
        label = label
      ),
      nrounds = 73L,
      verbose = 0
    )
    fold_prediction <- predict(
      model,
      xgb.DMatrix(
        wide_feature_matrix(validation)
      )
    )
    if (is.null(dim(fold_prediction))) {
      fold_prediction <- matrix(
        fold_prediction,
        ncol = 4L,
        byrow = TRUE
      )
    }
    fold_prediction <- validate_probability(
      fold_prediction, nrow(validation)
    )
    prediction[validation_rows, ] <- fold_prediction
    saveRDS(
      list(
        experiment_id = experiment_id,
        validation_no = validation$No,
        prediction = fold_prediction
      ),
      path
    )
    cat("  original xgboost fold", fold, "complete\n")
    flush.console()
  }
  validate_probability(prediction, nrow(train))
}

load_repeat_components <- function(
    train,
    fold_map,
    seed,
    seed_index) {
  source_path <- file.path(
    "data_processed", "codex_componentwise_boost",
    sprintf("repeat_seed_%d.rds", seed)
  )
  if (file.exists(source_path)) {
    saved <- readRDS(source_path)
    fold_match <- identical(
      as.integer(
        saved$fold_map[
          as.character(sort(unique(train$Case)))
        ]
      ),
      as.integer(
        fold_map[
          as.character(sort(unique(train$Case)))
        ]
      )
    )
    if (
      identical(as.integer(saved$seed), as.integer(seed)) &&
      fold_match &&
      identical(
        dim(as.matrix(saved$xgb_oof)),
        c(nrow(train), 4L)
      ) &&
      identical(
        dim(as.matrix(saved$mlp_oof)),
        c(nrow(train), 4L)
      )
    ) {
      cat(
        "Reusing verified exact-fold xgboost/MLP cache:",
        source_path, "\n"
      )
      return(list(
        xgb = validate_probability(
          saved$xgb_oof, nrow(train)
        ),
        mlp = validate_probability(
          saved$mlp_oof, nrow(train)
        ),
        source = source_path
      ))
    }
    cat(
      "Existing repeat cache failed fold/spec checks; ",
      "refitting components.\n",
      sep = ""
    )
  }
  tag <- paste0("seed_", seed)
  list(
    xgb = fit_original_xgb_oof(
      train, fold_map, tag
    ),
    mlp = fit_shallow_mlp_oof(
      train, fold_map, seed_index, tag
    ),
    source = "refitted"
  )
}

bootstrap_case_gain <- function(
    gain_by_case,
    replicates,
    seed = 4821L,
    chunk_size = 1000L) {
  set.seed(seed)
  n_case <- length(gain_by_case)
  result <- numeric(replicates)
  for (
    start in seq.int(
      1L, replicates, by = chunk_size
    )
  ) {
    stop_at <- min(
      replicates, start + chunk_size - 1L
    )
    n_this <- stop_at - start + 1L
    sampled <- matrix(
      sample.int(
        n_case,
        n_case * n_this,
        replace = TRUE
      ),
      nrow = n_case
    )
    result[start:stop_at] <- colMeans(matrix(
      gain_by_case[sampled],
      nrow = n_case
    ))
  }
  result
}

bootstrap_summary <- function(
    truth,
    baseline,
    candidate,
    case,
    replicates = 100000L,
    seed = 4821L) {
  respondent_gain <- unname(tapply(
    row_log_loss_local(truth, baseline) -
      row_log_loss_local(truth, candidate),
    case,
    mean
  ))
  bootstrap <- bootstrap_case_gain(
    respondent_gain, replicates, seed
  )
  list(
    summary = data.frame(
      point_gain = mean(respondent_gain),
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
    respondent_gain = respondent_gain,
    bootstrap = bootstrap
  )
}

fold_curve_table <- function(
    fold_results,
    repeat_id) {
  do.call(rbind, lapply(
    seq_along(fold_results),
    function(fold) {
      result <- fold_results[[fold]]
      if (is.null(result)) return(NULL)
      data.frame(
        repeat_id = repeat_id,
        fold = fold,
        selected_lambda =
          lambda_label(result$selected_lambda),
        price_level = 1:12,
        original_utility = as.numeric(
          result$original_price_curve
        ),
        smoothed_utility = as.numeric(
          result$smoothed_price_curve
        )
      )
    }
  ))
}

run_price_curve_experiment <- function() {
  train <- read.csv(
    file.path("csv files", "train.csv")
  )
  train <- train[order(train$No), , drop = FALSE]
  rownames(train) <- NULL
  test <- read.csv(
    file.path("csv files", "test.csv")
  )
  saved_base <- readRDS(file.path(
    "data_processed", "oof_ensemble_v10.rds"
  ))
  truth <- as.matrix(
    train[, paste0("Ch", 1:4), drop = FALSE]
  )
  train_long <- sort_choice_long_local(
    reshape_choice_long(train)
  )

  stopifnot(
    nrow(train) == 21565L,
    nrow(test) == 4997L,
    length(unique(train$Case)) == 1135L,
    all(table(train$Case) == 19L),
    all(rowSums(truth) == 1L)
  )

  canonical_map <- canonical_fold_map(
    train, saved_base
  )
  cat("\nRunning", experiment_id, "\n")
  cat(
    "Canonical baseline target:",
    sprintf("%.12f", canonical_baseline_target),
    "\n"
  )
  if (smoke_mode) {
    cat(
      "SMOKE MODE: one outer fold; no model verdict.\n"
    )
  }

  canonical_candidate <- run_price_curve_oof(
    train,
    train_long,
    canonical_map,
    "seed_4821",
    outer_seed = 4821L
  )

  if (smoke_mode) {
    smoke_path <- file.path(
      output_dir, "smoke_result.rds"
    )
    saveRDS(
      list(
        experiment_id = experiment_id,
        result = canonical_candidate
      ),
      smoke_path
    )
    cat(
      "\nSmoke test completed successfully:",
      smoke_path, "\n"
    )
    return(invisible(list(
      smoke = TRUE, path = smoke_path
    )))
  }

  base_mlogit <- validate_probability(
    saved_base$oof_mlogit, nrow(train)
  )
  original_xgb <- validate_probability(
    saved_base$oof_xgb, nrow(train)
  )
  maximum_baseline_difference <- max(abs(
    canonical_candidate$baseline - base_mlogit
  ))
  cat(sprintf(
    paste0(
      "Canonical m8trpg reconstruction max ",
      "abs difference: %.3g\n"
    ),
    maximum_baseline_difference
  ))
  if (maximum_baseline_difference > 1e-6) {
    stop(
      paste0(
        "Fold-refitted m8trpg does not reproduce the ",
        "canonical OOF cache. Stopping rather than ",
        "comparing mismatched baselines."
      )
    )
  }

  shallow_mlp <- load_mlp_from_cache(nrow(train))
  if (is.null(shallow_mlp)) {
    cat(
      paste0(
        "Frozen shallow-MLP OOF cache not found; ",
        "rebuilding the exact component.\n"
      )
    )
    shallow_mlp <- fit_shallow_mlp_oof(
      train,
      canonical_map,
      repeat_index = 0L,
      checkpoint_tag = "canonical"
    )
    saveRDS(
      list(
        experiment_id = experiment_id,
        prediction = shallow_mlp
      ),
      file.path(
        output_dir,
        "shallow_mlp_oof_rebuilt.rds"
      )
    )
  }

  flat_baseline <- validate_probability(
    0.85 * (
      0.80 * base_mlogit +
        0.20 * original_xgb
    ) + 0.15 * shallow_mlp,
    nrow(train)
  )
  flat_candidate <- validate_probability(
    0.85 * (
      0.80 * canonical_candidate$candidate +
        0.20 * original_xgb
    ) + 0.15 * shallow_mlp,
    nrow(train)
  )
  baseline_loss <- log_loss_matrix_local(
    truth, flat_baseline
  )
  candidate_loss <- log_loss_matrix_local(
    truth, flat_candidate
  )
  if (
    abs(
      baseline_loss - canonical_baseline_target
    ) > 1e-6
  ) {
    stop(sprintf(
      paste0(
        "Frozen flat baseline mismatch: got %.12f, ",
        "expected %.12f."
      ),
      baseline_loss,
      canonical_baseline_target
    ))
  }

  canonical_bootstrap <- bootstrap_summary(
    truth,
    flat_baseline,
    flat_candidate,
    train$Case,
    bootstrap_replicates
  )
  canonical_summary <- cbind(
    data.frame(
      experiment = experiment_id,
      stage = "canonical",
      baseline_logloss = baseline_loss,
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

  canonical_curves <- fold_curve_table(
    canonical_candidate$fold_results,
    "4821"
  )
  write.csv(
    canonical_summary,
    file.path(output_dir, "canonical_summary.csv"),
    row.names = FALSE
  )
  write.csv(
    canonical_curves,
    file.path(
      output_dir, "canonical_price_curves.csv"
    ),
    row.names = FALSE
  )
  saveRDS(
    list(
      experiment_id = experiment_id,
      summary = canonical_summary,
      bootstrap = canonical_bootstrap,
      baseline_prediction = flat_baseline,
      candidate_prediction = flat_candidate,
      candidate_mlogit =
        canonical_candidate$candidate,
      candidate_folds =
        canonical_candidate$fold_results
    ),
    file.path(output_dir, "canonical_result.rds")
  )

  cat("\nCanonical result:\n")
  print(canonical_summary, digits = 9)

  run_repeated <- repeated_mode == "always" ||
    (
      repeated_mode == "auto" &&
      (
        isTRUE(canonical_summary$canonical_pass) ||
        isTRUE(canonical_summary$near_miss)
      )
    )

  if (!run_repeated) {
    verdict <- if (
      canonical_summary$point_gain <= 0
    ) {
      "REJECT: candidate point gain is non-positive."
    } else if (
      canonical_summary$lower_95 <
        near_miss_lower_limit
    ) {
      paste0(
        "REJECT: canonical CI crosses zero by more ",
        "than the pre-registered near-miss allowance."
      )
    } else {
      paste0(
        "CANONICAL PASS/NEAR MISS, but repeated CV ",
        "was disabled with PRICE_CURVE_REPEATED=never."
      )
    }
    writeLines(
      verdict, file.path(output_dir, "verdict.txt")
    )
    cat("\n", verdict, "\n", sep = "")
    cat("Results:", normalizePath(output_dir), "\n")
    return(invisible(list(
      canonical = canonical_summary,
      verdict = verdict
    )))
  }

  cat(
    "\nRepeated-CV escalation activated (",
    repeated_mode, ").\n",
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
      baseline_logloss = baseline_loss,
      candidate_logloss = candidate_loss,
      gain = baseline_loss - candidate_loss,
      component_source = "canonical_cache"
    )
  )
  curve_tables <- list(
    `4821` = canonical_curves
  )

  for (
    seed_index in seq_along(additional_seeds)
  ) {
    seed <- additional_seeds[[seed_index]]
    seed_name <- as.character(seed)
    tag <- paste0("seed_", seed)
    repeat_path <- file.path(
      output_dir,
      sprintf("repeat_seed_%d.rds", seed)
    )
    repeat_result <- NULL
    if (file.exists(repeat_path)) {
      saved <- readRDS(repeat_path)
      if (identical(
        saved$experiment_id, experiment_id
      )) {
        repeat_result <- saved
        cat(
          "\nRepeat seed", seed,
          "loaded from checkpoint.\n"
        )
      }
    }

    if (is.null(repeat_result)) {
      cat(sprintf(
        paste0(
          "\n######## repeated CV seed %d ",
          "(%d/5) ########\n"
        ),
        seed, seed_index
      ))
      fold_map <- repeated_fold_map(
        case_levels, seed
      )
      price_curve_oof <- run_price_curve_oof(
        train,
        train_long,
        fold_map,
        tag,
        outer_seed = seed
      )
      components <- load_repeat_components(
        train,
        fold_map,
        seed,
        seed_index
      )
      repeat_baseline <- validate_probability(
        0.85 * (
          0.80 * price_curve_oof$baseline +
            0.20 * components$xgb
        ) + 0.15 * components$mlp,
        nrow(train)
      )
      repeat_candidate <- validate_probability(
        0.85 * (
          0.80 * price_curve_oof$candidate +
            0.20 * components$xgb
        ) + 0.15 * components$mlp,
        nrow(train)
      )
      respondent_gain <- unname(tapply(
        row_log_loss_local(
          truth, repeat_baseline
        ) -
          row_log_loss_local(
            truth, repeat_candidate
          ),
        train$Case,
        mean
      ))
      repeat_result <- list(
        experiment_id = experiment_id,
        seed = seed,
        fold_map = fold_map,
        baseline_prediction = repeat_baseline,
        candidate_prediction = repeat_candidate,
        respondent_gain = respondent_gain,
        component_source = components$source,
        fold_results = price_curve_oof$fold_results
      )
      saveRDS(repeat_result, repeat_path)
    }

    case_gain_matrix[, seed_name] <-
      repeat_result$respondent_gain
    repeat_baseline_loss <- log_loss_matrix_local(
      truth, repeat_result$baseline_prediction
    )
    repeat_candidate_loss <- log_loss_matrix_local(
      truth, repeat_result$candidate_prediction
    )
    repeat_rows[[seed_name]] <- data.frame(
      seed = seed,
      baseline_logloss = repeat_baseline_loss,
      candidate_logloss = repeat_candidate_loss,
      gain = repeat_baseline_loss -
        repeat_candidate_loss,
      component_source =
        repeat_result$component_source
    )
    curve_tables[[seed_name]] <-
      fold_curve_table(
        repeat_result$fold_results, seed_name
      )
  }

  repeat_summary <- do.call(rbind, repeat_rows)
  average_case_gain <- rowMeans(case_gain_matrix)
  pooled_bootstrap <- bootstrap_case_gain(
    average_case_gain,
    bootstrap_replicates,
    seed = 4821L
  )
  pooled_summary <- data.frame(
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
    positive_repeats = sum(repeat_summary$gain > 0),
    n_repeats = nrow(repeat_summary),
    n_boot = length(pooled_bootstrap)
  )
  pooled_summary$promote <-
    pooled_summary$point_gain > 0 &
    pooled_summary$lower_95 > 0 &
    pooled_summary$positive_repeats >= 5L

  write.csv(
    repeat_summary,
    file.path(
      output_dir, "repeated_cv_by_seed.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    pooled_summary,
    file.path(
      output_dir, "repeated_cv_summary.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    do.call(rbind, curve_tables),
    file.path(
      output_dir, "repeated_cv_price_curves.csv"
    ),
    row.names = FALSE
  )
  saveRDS(
    list(
      experiment_id = experiment_id,
      canonical = canonical_summary,
      by_seed = repeat_summary,
      pooled = pooled_summary,
      case_gain_matrix = case_gain_matrix,
      pooled_bootstrap = pooled_bootstrap
    ),
    file.path(
      output_dir, "repeated_cv_result.rds"
    )
  )

  verdict <- if (isTRUE(pooled_summary$promote)) {
    paste0(
      "PASS: repeated-CV pooled ordinary 95% lower ",
      "bound excludes zero and at least 5/6 repeats ",
      "improve. Eligible for a separate submission audit."
    )
  } else {
    paste0(
      "REJECT: repeated-CV promotion rule was not ",
      "fully satisfied; the standing model remains ",
      "unchanged."
    )
  }
  writeLines(
    verdict, file.path(output_dir, "verdict.txt")
  )

  cat("\nRepeated-CV results by seed:\n")
  print(repeat_summary, digits = 9)
  cat("\nPooled repeated-CV result:\n")
  print(pooled_summary, digits = 9)
  cat("\n", verdict, "\n", sep = "")
  cat("Results:", normalizePath(output_dir), "\n")

  invisible(list(
    canonical = canonical_summary,
    repeated = pooled_summary,
    verdict = verdict
  ))
}

price_curve_result <- run_price_curve_experiment()
