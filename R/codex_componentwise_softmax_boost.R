# Component-wise exact-softmax residual boosting on the m8trpg offset.
#
# Pre-registration:
#   codex_componentwise_boost_preregister.md
#
# Run from the repository root:
#   source("R/codex_componentwise_softmax_boost.R")
# or:
#   Rscript R/codex_componentwise_softmax_boost.R
#
# The default is the complete canonical five-fold experiment. If the frozen
# near-miss rule fires, the five additional repeated-CV seeds run
# automatically. Set COMPBOOST_REPEATED=never to stop after canonical CV, or
# COMPBOOST_REPEATED=always to force the repeated-CV stage.
#
# A fast plumbing-only check is available with COMPBOOST_SMOKE=1. Smoke-mode
# output is explicitly not a model verdict.

options(stringsAsFactors = FALSE)

required_packages <- c(
  "Matrix", "mlogit", "dfidx", "xgboost", "nnet"
)
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
  library(Matrix)
  library(mlogit)
  library(dfidx)
  library(xgboost)
  library(nnet)
})

source("R/codex_shared_utility_common.R")

experiment_id <- "componentwise_softmax_boost_v1"
output_dir <- "data_processed/codex_componentwise_boost"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

smoke_mode <- identical(Sys.getenv("COMPBOOST_SMOKE", "0"), "1")
repeated_mode <- tolower(Sys.getenv("COMPBOOST_REPEATED", "auto"))
stopifnot(repeated_mode %in% c("auto", "never", "always"))

learning_rate <- 0.05
maximum_iterations <- if (smoke_mode) 5L else 250L
bootstrap_replicates <- if (smoke_mode) {
  200L
} else {
  as.integer(Sys.getenv("COMPBOOST_N_BOOT", "100000"))
}
near_miss_lower_limit <- -0.00025
additional_seeds <- c(1907L, 2719L, 6151L, 8293L, 104729L)

input_files <- c(
  "csv files/train.csv",
  "csv files/test.csv",
  "data_processed/oof_ensemble_v10.rds"
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
  prediction <- as.matrix(prediction)
  prediction <- prediction / rowSums(prediction)
  prediction <- pmin(pmax(prediction, 1e-15), 1 - 1e-15)
  -mean(rowSums(as.matrix(truth) * log(prediction)))
}

row_log_loss_local <- function(truth, prediction) {
  prediction <- as.matrix(prediction)
  prediction <- prediction / rowSums(prediction)
  prediction <- pmin(pmax(prediction, 1e-15), 1 - 1e-15)
  -rowSums(as.matrix(truth) * log(prediction))
}

validate_probability <- function(prediction, n_rows = NULL) {
  prediction <- as.matrix(prediction)
  if (!is.null(n_rows)) {
    stopifnot(identical(dim(prediction), c(as.integer(n_rows), 4L)))
  } else {
    stopifnot(ncol(prediction) == 4L)
  }
  stopifnot(
    !anyNA(prediction),
    all(is.finite(prediction)),
    all(prediction > 0)
  )
  prediction <- prediction / rowSums(prediction)
  stopifnot(max(abs(rowSums(prediction) - 1)) < 1e-10)
  prediction
}

softmax_long <- function(utility) {
  stopifnot(length(utility) %% 4L == 0L)
  matrix_utility <- matrix(utility, ncol = 4L, byrow = TRUE)
  matrix_utility <- matrix_utility -
    apply(matrix_utility, 1L, max)
  exponential <- exp(matrix_utility)
  as.vector(t(exponential / rowSums(exponential)))
}

long_log_loss <- function(chosen, utility) {
  probability <- softmax_long(utility)
  -sum(chosen * log(pmax(probability, 1e-15))) /
    (length(chosen) / 4L)
}

sort_choice_long <- function(data) {
  data <- as.data.frame(data)
  data <- data[order(data$No, as.integer(data$alt)), , drop = FALSE]
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

make_sparse_indicator <- function(data, variables, level_list,
                                  prefix) {
  n <- nrow(data)
  column_names <- character()
  row_index <- list()
  column_index <- list()
  offset <- 0L
  for (variable in variables) {
    levels <- level_list[[variable]]
    if (length(levels) == 0L) next
    for (level in levels) {
      offset <- offset + 1L
      rows <- which(as.numeric(data[[variable]]) == level)
      row_index[[offset]] <- rows
      column_index[[offset]] <- rep.int(offset, length(rows))
      column_names[[offset]] <- paste0(
        prefix, variable, "_L", format(level, scientific = FALSE)
      )
    }
  }
  if (offset == 0L) {
    return(Matrix(0, nrow = n, ncol = 0L, sparse = TRUE))
  }
  rows <- unlist(row_index, use.names = FALSE)
  columns <- unlist(column_index, use.names = FALSE)
  out <- sparseMatrix(
    i = rows,
    j = columns,
    x = rep.int(1, length(rows)),
    dims = c(n, offset)
  )
  colnames(out) <- column_names
  out
}

make_dictionary_recipe <- function(fitting_long) {
  fitting_long <- sort_choice_long(fitting_long)
  scaler <- choice_scaler(fitting_long)

  attribute_levels <- setNames(
    lapply(attrs, function(attribute) {
      sort(unique(as.numeric(
        fitting_long[[attribute]][fitting_long[[attribute]] > 0]
      )))
    }),
    attrs
  )
  price_levels <- list(
    Price = 2:12
  )

  pair_specs <- list()
  pair_index <- 0L
  for (left_index in seq_len(length(attrs) - 1L)) {
    for (right_index in (left_index + 1L):length(attrs)) {
      left <- attrs[[left_index]]
      right <- attrs[[right_index]]
      left_levels <- attribute_levels[[left]]
      right_levels <- attribute_levels[[right]]
      left_code <- match(
        as.numeric(fitting_long[[left]]), left_levels
      )
      right_code <- match(
        as.numeric(fitting_long[[right]]), right_levels
      )
      usable <- !is.na(left_code) & !is.na(right_code)
      combination <- (
        left_code[usable] - 1L
      ) * length(right_levels) + right_code[usable]
      observed <- sort(unique(combination))
      if (length(observed) == 0L) next
      pair_index <- pair_index + 1L
      pair_specs[[pair_index]] <- list(
        left = left,
        right = right,
        left_levels = left_levels,
        right_levels = right_levels,
        combination = observed
      )
    }
  }

  list(
    scaler = scaler,
    attribute_levels = attribute_levels,
    price_levels = price_levels,
    pair_specs = pair_specs
  )
}

make_context_matrix <- function(data, recipe) {
  data <- sort_choice_long(data)
  features <- make_m8trpg_features(
    data,
    recipe$scaler$ctr,
    recipe$scaler$scl,
    "none"
  )
  standardized_names <- c(
    "incomea", "agea", "milesa", "nighta",
    "genderind", "Urbind", "educind"
  )
  context <- list()
  for (variable in standardized_names) {
    context[[paste0("z_", variable)]] <- (
      as.numeric(features[[variable]]) -
        recipe$scaler$ctr[[variable]]
    ) / recipe$scaler$scl[[variable]]
  }
  for (level in 2:6) {
    context[[paste0("segment_", level)]] <-
      as.numeric(features$segmentind == level)
  }
  for (level in 2:5) {
    context[[paste0("region_", level)]] <-
      as.numeric(features$regionind == level)
  }
  for (level in 2:5) {
    context[[paste0("ppark_", level)]] <-
      as.numeric(features$pparkind == level)
  }
  context$Task_c <- features$Task_c
  context$is_cheapest <- features$is_cheapest
  context$is_dearest <- features$is_dearest
  context$price_gap_min_scaled <- features$price_gap_min / 11
  context$price_gap_max_scaled <- features$price_gap_max / 11

  out <- do.call(cbind, context)
  storage.mode(out) <- "double"
  stopifnot(
    nrow(out) == nrow(data),
    all(is.finite(out))
  )
  out
}

make_pair_indicator <- function(data, pair_specs) {
  data <- sort_choice_long(data)
  n <- nrow(data)
  if (length(pair_specs) == 0L) {
    return(Matrix(0, nrow = n, ncol = 0L, sparse = TRUE))
  }

  row_parts <- list()
  column_parts <- list()
  column_names <- character()
  offset <- 0L

  for (spec in pair_specs) {
    left_code <- match(
      as.numeric(data[[spec$left]]), spec$left_levels
    )
    right_code <- match(
      as.numeric(data[[spec$right]]), spec$right_levels
    )
    usable <- !is.na(left_code) & !is.na(right_code)
    combination <- rep.int(NA_integer_, n)
    combination[usable] <- (
      left_code[usable] - 1L
    ) * length(spec$right_levels) + right_code[usable]
    local_column <- match(combination, spec$combination)
    rows <- which(!is.na(local_column))

    if (length(rows) > 0L) {
      row_parts[[length(row_parts) + 1L]] <- rows
      column_parts[[length(column_parts) + 1L]] <-
        offset + local_column[rows]
    }

    for (combination_id in spec$combination) {
      left_level_index <- (
        (combination_id - 1L) %/% length(spec$right_levels)
      ) + 1L
      right_level_index <- (
        (combination_id - 1L) %% length(spec$right_levels)
      ) + 1L
      column_names[[offset + 1L]] <- paste0(
        "pair__", spec$left, "_L",
        format(
          spec$left_levels[[left_level_index]],
          scientific = FALSE
        ),
        "__", spec$right, "_L",
        format(
          spec$right_levels[[right_level_index]],
          scientific = FALSE
        )
      )
      offset <- offset + 1L
    }
  }

  rows <- unlist(row_parts, use.names = FALSE)
  columns <- unlist(column_parts, use.names = FALSE)
  out <- sparseMatrix(
    i = rows,
    j = columns,
    x = rep.int(1, length(rows)),
    dims = c(n, offset)
  )
  colnames(out) <- column_names
  out
}

make_raw_dictionary <- function(data, recipe) {
  data <- sort_choice_long(data)
  attribute_atom <- make_sparse_indicator(
    data,
    attrs,
    recipe$attribute_levels,
    "attr__"
  )
  price_atom <- make_sparse_indicator(
    data,
    "Price",
    recipe$price_levels,
    "price__"
  )
  context <- make_context_matrix(data, recipe)

  attribute_context <- vector("list", ncol(context))
  price_context <- vector("list", ncol(context))
  for (context_index in seq_len(ncol(context))) {
    row_scaler <- Diagonal(x = context[, context_index])
    attribute_context[[context_index]] <-
      row_scaler %*% attribute_atom
    price_context[[context_index]] <-
      row_scaler %*% price_atom
    colnames(attribute_context[[context_index]]) <- paste0(
      colnames(attribute_atom),
      "__x__", colnames(context)[[context_index]]
    )
    colnames(price_context[[context_index]]) <- paste0(
      colnames(price_atom),
      "__x__", colnames(context)[[context_index]]
    )
  }
  attribute_context <- do.call(cbind, attribute_context)
  price_context <- do.call(cbind, price_context)
  pair_indicator <- make_pair_indicator(data, recipe$pair_specs)

  out <- cbind(
    attribute_context,
    price_context,
    pair_indicator
  )
  out <- as(out, "dgCMatrix")
  stopifnot(
    nrow(out) == nrow(data),
    !anyDuplicated(colnames(out)),
    all(is.finite(out@x))
  )
  out
}

make_dictionary_pair <- function(fitting_long, prediction_long) {
  recipe <- make_dictionary_recipe(fitting_long)
  fitting_matrix <- make_raw_dictionary(fitting_long, recipe)
  sum_of_squares <- Matrix::colSums(fitting_matrix ^ 2)
  keep <- is.finite(sum_of_squares) & sum_of_squares > 0
  fitting_matrix <- fitting_matrix[, keep, drop = FALSE]
  sum_of_squares <- sum_of_squares[keep]

  prediction_matrix <- make_raw_dictionary(
    prediction_long, recipe
  )
  prediction_column <- match(
    colnames(fitting_matrix),
    colnames(prediction_matrix)
  )
  stopifnot(!anyNA(prediction_column))
  prediction_matrix <- prediction_matrix[
    ,
    prediction_column,
    drop = FALSE
  ]
  stopifnot(
    identical(
      colnames(fitting_matrix),
      colnames(prediction_matrix)
    )
  )

  list(
    fitting = fitting_matrix,
    prediction = prediction_matrix,
    sum_of_squares = as.numeric(sum_of_squares),
    recipe = recipe
  )
}

componentwise_path <- function(
    fitting_matrix,
    chosen,
    fitting_offset,
    prediction_matrix = NULL,
    prediction_chosen = NULL,
    prediction_offset = NULL,
    iterations,
    learning_rate) {
  stopifnot(
    nrow(fitting_matrix) == length(chosen),
    length(fitting_offset) == length(chosen),
    nrow(fitting_matrix) %% 4L == 0L
  )
  denominator <- as.numeric(
    Matrix::colSums(fitting_matrix ^ 2)
  )
  stopifnot(
    all(is.finite(denominator)),
    all(denominator > 0)
  )

  fitting_utility <- as.numeric(fitting_offset)
  use_prediction <- !is.null(prediction_matrix)
  track_validation <- use_prediction &&
    !is.null(prediction_chosen)
  if (use_prediction) {
    stopifnot(
      ncol(prediction_matrix) == ncol(fitting_matrix),
      identical(
        colnames(prediction_matrix),
        colnames(fitting_matrix)
      ),
      length(prediction_offset) == nrow(prediction_matrix)
    )
    prediction_utility <- as.numeric(prediction_offset)
  } else {
    prediction_utility <- NULL
  }
  if (track_validation) {
    stopifnot(
      length(prediction_chosen) == nrow(prediction_matrix)
    )
    validation_loss <- numeric(iterations + 1L)
    validation_loss[[1L]] <- long_log_loss(
      prediction_chosen, prediction_utility
    )
  } else {
    validation_loss <- NULL
  }

  selected_index <- integer(iterations)
  selected_step <- numeric(iterations)
  selected_score <- numeric(iterations)
  completed <- 0L

  for (iteration in seq_len(iterations)) {
    negative_gradient <- chosen - softmax_long(fitting_utility)
    correlation <- as.numeric(
      Matrix::crossprod(fitting_matrix, negative_gradient)
    )
    score <- correlation ^ 2 / denominator
    score[!is.finite(score)] <- -Inf
    selected <- which.max(score)
    if (length(selected) == 0L || score[[selected]] <= 0) {
      break
    }
    coefficient <- correlation[[selected]] / denominator[[selected]]
    step <- learning_rate * coefficient
    fitting_utility <- fitting_utility +
      step * as.numeric(fitting_matrix[, selected])
    if (use_prediction) {
      prediction_utility <- prediction_utility +
        step * as.numeric(prediction_matrix[, selected])
    }
    if (track_validation) {
      validation_loss[[iteration + 1L]] <- long_log_loss(
        prediction_chosen, prediction_utility
      )
    }
    selected_index[[iteration]] <- selected
    selected_step[[iteration]] <- step
    selected_score[[iteration]] <- score[[selected]]
    completed <- iteration
  }

  if (completed < iterations && track_validation) {
    validation_loss[
      (completed + 2L):(iterations + 1L)
    ] <- validation_loss[[completed + 1L]]
  }

  path <- if (completed > 0L) {
    data.frame(
      iteration = seq_len(completed),
      feature = colnames(fitting_matrix)[
        selected_index[seq_len(completed)]
      ],
      step = selected_step[seq_len(completed)],
      score = selected_score[seq_len(completed)]
    )
  } else {
    data.frame(
      iteration = integer(),
      feature = character(),
      step = numeric(),
      score = numeric()
    )
  }

  list(
    fitting_utility = fitting_utility,
    prediction_utility = prediction_utility,
    validation_loss = validation_loss,
    path = path,
    completed = completed
  )
}

next_inner_fold <- function(outer_fold) {
  if (outer_fold == 5L) 1L else outer_fold + 1L
}

run_candidate_fold <- function(
    train_long,
    train_wide,
    fold_map,
    outer_fold,
    maximum_iterations,
    learning_rate) {
  row_fold <- unname(fold_map[as.character(train_wide$Case)])
  long_fold <- unname(fold_map[as.character(train_long$Case)])
  stopifnot(!anyNA(row_fold), !anyNA(long_fold))

  inner_validation_fold <- next_inner_fold(outer_fold)
  inner_fitting_long <- sort_choice_long(
    train_long[
      long_fold != outer_fold &
        long_fold != inner_validation_fold,
      ,
      drop = FALSE
    ]
  )
  inner_validation_long <- sort_choice_long(
    train_long[
      long_fold == inner_validation_fold,
      ,
      drop = FALSE
    ]
  )
  stopifnot(
    length(intersect(
      unique(inner_fitting_long$Case),
      unique(inner_validation_long$Case)
    )) == 0L
  )

  cat(sprintf(
    "  inner fit (outer %d, early-stop fold %d): m8trpg offset\n",
    outer_fold, inner_validation_fold
  ))
  inner_base <- fit_m8trpg_model(inner_fitting_long)
  inner_fitting_offset <- predict_m8trpg_margin(
    inner_base, inner_fitting_long
  )
  inner_validation_offset <- predict_m8trpg_margin(
    inner_base, inner_validation_long
  )
  inner_dictionary <- make_dictionary_pair(
    inner_fitting_long, inner_validation_long
  )
  cat(sprintf(
    "  inner dictionary: %d rows x %d components\n",
    nrow(inner_dictionary$fitting),
    ncol(inner_dictionary$fitting)
  ))
  inner_path <- componentwise_path(
    fitting_matrix = inner_dictionary$fitting,
    chosen = as.numeric(inner_fitting_long$chosen),
    fitting_offset = inner_fitting_offset$margin,
    prediction_matrix = inner_dictionary$prediction,
    prediction_chosen = as.numeric(inner_validation_long$chosen),
    prediction_offset = inner_validation_offset$margin,
    iterations = maximum_iterations,
    learning_rate = learning_rate
  )
  chosen_iterations <- which.min(inner_path$validation_loss) - 1L
  inner_curve <- data.frame(
    iteration = 0:maximum_iterations,
    logloss = inner_path$validation_loss
  )
  cat(sprintf(
    "  selected %d/%d iterations; inner loss %.9f -> %.9f\n",
    chosen_iterations,
    maximum_iterations,
    inner_curve$logloss[[1L]],
    inner_curve$logloss[[chosen_iterations + 1L]]
  ))

  rm(
    inner_base,
    inner_fitting_offset,
    inner_validation_offset,
    inner_dictionary
  )
  invisible(gc())

  outer_fitting_long <- sort_choice_long(
    train_long[long_fold != outer_fold, , drop = FALSE]
  )
  outer_validation_long <- sort_choice_long(
    train_long[long_fold == outer_fold, , drop = FALSE]
  )
  outer_validation_wide <- train_wide[
    row_fold == outer_fold, , drop = FALSE
  ]
  stopifnot(
    length(intersect(
      unique(outer_fitting_long$Case),
      unique(outer_validation_long$Case)
    )) == 0L
  )

  cat("  outer refit: m8trpg offset and frozen component count\n")
  outer_base <- fit_m8trpg_model(outer_fitting_long)
  outer_fitting_offset <- predict_m8trpg_margin(
    outer_base, outer_fitting_long
  )
  outer_validation_offset <- predict_m8trpg_margin(
    outer_base, outer_validation_long
  )
  outer_dictionary <- make_dictionary_pair(
    outer_fitting_long, outer_validation_long
  )
  cat(sprintf(
    "  outer dictionary: %d rows x %d components\n",
    nrow(outer_dictionary$fitting),
    ncol(outer_dictionary$fitting)
  ))

  outer_path <- componentwise_path(
    fitting_matrix = outer_dictionary$fitting,
    chosen = as.numeric(outer_fitting_long$chosen),
    fitting_offset = outer_fitting_offset$margin,
    prediction_matrix = outer_dictionary$prediction,
    prediction_offset = outer_validation_offset$margin,
    iterations = chosen_iterations,
    learning_rate = learning_rate
  )
  baseline_prediction <- validate_probability(
    outer_validation_offset$pred,
    nrow(outer_validation_wide)
  )
  candidate_prediction <- validate_probability(
    matrix(
      softmax_long(outer_path$prediction_utility),
      ncol = 4L,
      byrow = TRUE
    ),
    nrow(outer_validation_wide)
  )
  validation_no <- outer_validation_offset$no
  stopifnot(
    identical(
      as.integer(validation_no),
      as.integer(outer_validation_wide$No)
    )
  )

  list(
    outer_fold = outer_fold,
    inner_validation_fold = inner_validation_fold,
    selected_iterations = chosen_iterations,
    inner_curve = inner_curve,
    inner_path = inner_path$path,
    outer_path = outer_path$path,
    baseline_prediction = baseline_prediction,
    candidate_prediction = candidate_prediction,
    validation_no = validation_no,
    dictionary_size = ncol(outer_dictionary$fitting)
  )
}

canonical_fold_map <- function(train, saved) {
  fold_map <- saved$fold_of_case
  stopifnot(
    length(fold_map) == length(unique(train$Case)),
    !anyNA(fold_map[as.character(unique(train$Case))]),
    all(sort(unique(as.integer(fold_map))) == 1:5)
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

load_mlp_from_cache <- function(n_rows) {
  candidate_paths <- c(
    "data_processed/codex_behavioral_round/mlp_oof.rds",
    file.path(output_dir, "shallow_mlp_oof_rebuilt.rds")
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
    if (!is.null(prediction) &&
        identical(dim(as.matrix(prediction)), c(n_rows, 4L))) {
      cat("Using frozen shallow-MLP OOF cache:", path, "\n")
      return(validate_probability(prediction, n_rows))
    }
  }
  NULL
}

mlp_helper_environment <- function() {
  helper <- new.env(parent = globalenv())
  old_stage <- Sys.getenv("CODEX_STAGE", unset = NA_character_)
  Sys.setenv(CODEX_STAGE = "define")
  on.exit({
    if (is.na(old_stage)) {
      Sys.unsetenv("CODEX_STAGE")
    } else {
      Sys.setenv(CODEX_STAGE = old_stage)
    }
  })
  sys.source("R/codex_mlp_ensemble.R", envir = helper)
  helper
}

fit_shallow_mlp_oof <- function(
    train,
    fold_map,
    repeat_index = 0L,
    checkpoint_tag = "canonical") {
  helper <- mlp_helper_environment()
  reference <- helper$feature_reference(train)
  row_fold <- unname(fold_map[as.character(train$Case)])
  prediction <- matrix(NA_real_, nrow(train), 4L)

  for (fold in 1:5) {
    path <- file.path(
      output_dir,
      sprintf("mlp_%s_fold_%d.rds", checkpoint_tag, fold)
    )
    validation_rows <- row_fold == fold
    if (file.exists(path)) {
      saved <- readRDS(path)
      if (identical(
        as.integer(saved$validation_no),
        as.integer(train$No[validation_rows])
      )) {
        prediction[validation_rows, ] <- saved$prediction
        cat("  shallow MLP fold", fold, "loaded from checkpoint\n")
        next
      }
    }

    fitting <- train[!validation_rows, , drop = FALSE]
    validation <- train[validation_rows, , drop = FALSE]
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
      as.matrix(fitting[, paste0("Ch", 1:4), drop = FALSE]),
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
  row_fold <- unname(fold_map[as.character(train$Case)])
  prediction <- matrix(NA_real_, nrow(train), 4L)
  for (fold in 1:5) {
    path <- file.path(
      output_dir,
      sprintf("xgb_%s_fold_%d.rds", checkpoint_tag, fold)
    )
    validation_rows <- row_fold == fold
    if (file.exists(path)) {
      saved <- readRDS(path)
      if (identical(
        as.integer(saved$validation_no),
        as.integer(train$No[validation_rows])
      )) {
        prediction[validation_rows, ] <- saved$prediction
        cat("  original xgboost fold", fold,
            "loaded from checkpoint\n")
        next
      }
    }
    fitting <- train[!validation_rows, , drop = FALSE]
    validation <- train[validation_rows, , drop = FALSE]
    label <- max.col(
      fitting[, paste0("Ch", 1:4), drop = FALSE]
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
      xgb.DMatrix(wide_feature_matrix(validation))
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

run_candidate_oof <- function(
    train,
    train_long,
    fold_map,
    checkpoint_tag) {
  row_fold <- unname(fold_map[as.character(train$Case)])
  baseline <- matrix(NA_real_, nrow(train), 4L)
  candidate <- matrix(NA_real_, nrow(train), 4L)
  fold_results <- vector("list", 5L)

  folds_to_run <- if (smoke_mode) 1L else 1:5
  for (fold in folds_to_run) {
    cat(sprintf(
      "\n=== %s: candidate outer fold %d/5 ===\n",
      checkpoint_tag, fold
    ))
    path <- file.path(
      output_dir,
      sprintf("candidate_%s_fold_%d.rds", checkpoint_tag, fold)
    )
    validation_rows <- row_fold == fold
    result <- NULL
    if (file.exists(path)) {
      saved <- readRDS(path)
      if (
        identical(saved$experiment_id, experiment_id) &&
        identical(
          as.integer(saved$result$validation_no),
          as.integer(train$No[validation_rows])
        ) &&
        identical(
          as.integer(saved$maximum_iterations),
          as.integer(maximum_iterations)
        )
      ) {
        result <- saved$result
        cat("  loaded from checkpoint\n")
      }
    }
    if (is.null(result)) {
      result <- run_candidate_fold(
        train_long,
        train,
        fold_map,
        fold,
        maximum_iterations,
        learning_rate
      )
      saveRDS(
        list(
          experiment_id = experiment_id,
          maximum_iterations = maximum_iterations,
          learning_rate = learning_rate,
          result = result
        ),
        path
      )
    }
    rows <- match(result$validation_no, train$No)
    stopifnot(!anyNA(rows), all(row_fold[rows] == fold))
    baseline[rows, ] <- result$baseline_prediction
    candidate[rows, ] <- result$candidate_prediction
    fold_results[[fold]] <- result
    cat(sprintf(
      "  fold %d candidate complete: %d iterations\n",
      fold, result$selected_iterations
    ))
    flush.console()
    invisible(gc())
  }

  if (smoke_mode) {
    return(list(
      baseline = baseline,
      candidate = candidate,
      fold_results = fold_results,
      completed_folds = folds_to_run
    ))
  }
  stopifnot(!anyNA(baseline), !anyNA(candidate))
  list(
    baseline = validate_probability(baseline, nrow(train)),
    candidate = validate_probability(candidate, nrow(train)),
    fold_results = fold_results,
    completed_folds = folds_to_run
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
  for (start in seq.int(1L, replicates, by = chunk_size)) {
    stop_at <- min(replicates, start + chunk_size - 1L)
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
    respondent_gain,
    replicates,
    seed
  )
  list(
    summary = data.frame(
      point_gain = mean(respondent_gain),
      bootstrap_mean = mean(bootstrap),
      bootstrap_sd = sd(bootstrap),
      lower_95 = unname(quantile(bootstrap, 0.025)),
      upper_95 = unname(quantile(bootstrap, 0.975)),
      lower_99 = unname(quantile(bootstrap, 0.005)),
      upper_99 = unname(quantile(bootstrap, 0.995)),
      win_rate = mean(bootstrap > 0),
      n_boot = length(bootstrap)
    ),
    respondent_gain = respondent_gain,
    bootstrap = bootstrap
  )
}

selection_tables <- function(fold_results, tag) {
  fold_table <- do.call(rbind, lapply(seq_along(fold_results), function(fold) {
    result <- fold_results[[fold]]
    if (is.null(result)) return(NULL)
    data.frame(
      repeat = tag,
      fold = fold,
      inner_validation_fold = result$inner_validation_fold,
      selected_iterations = result$selected_iterations,
      dictionary_size = result$dictionary_size,
      inner_initial_loss = result$inner_curve$logloss[[1L]],
      inner_selected_loss = result$inner_curve$logloss[[
        result$selected_iterations + 1L
      ]]
    )
  }))

  path_table <- do.call(rbind, lapply(seq_along(fold_results), function(fold) {
    result <- fold_results[[fold]]
    if (is.null(result) || nrow(result$outer_path) == 0L) return(NULL)
    cbind(
      data.frame(repeat = tag, fold = fold),
      result$outer_path
    )
  }))
  if (is.null(path_table)) {
    path_table <- data.frame(
      repeat = character(),
      fold = integer(),
      iteration = integer(),
      feature = character(),
      step = numeric(),
      score = numeric()
    )
  }
  list(folds = fold_table, path = path_table)
}

run_componentwise_experiment <- function() {
train <- read.csv("csv files/train.csv")
train <- train[order(train$No), , drop = FALSE]
rownames(train) <- NULL
test <- read.csv("csv files/test.csv")
saved_base <- readRDS("data_processed/oof_ensemble_v10.rds")
truth <- as.matrix(
  train[, paste0("Ch", 1:4), drop = FALSE]
)
train_long <- sort_choice_long(reshape_choice_long(train))

stopifnot(
  nrow(train) == 21565L,
  nrow(test) == 4997L,
  length(unique(train$Case)) == 1135L,
  all(table(train$Case) == 19L),
  all(rowSums(truth) == 1L)
)

canonical_map <- canonical_fold_map(train, saved_base)

cat("\nRunning", experiment_id, "\n")
cat("Learning rate:", learning_rate,
    "| max iterations:", maximum_iterations, "\n")
cat("Canonical baseline target: 1.143686618\n")
if (smoke_mode) {
  cat("SMOKE MODE: one outer fold, five iterations; no verdict.\n")
}

canonical_candidate <- run_candidate_oof(
  train,
  train_long,
  canonical_map,
  "seed_4821"
)

if (smoke_mode) {
  smoke_path <- file.path(output_dir, "smoke_result.rds")
  saveRDS(
    list(
      experiment_id = experiment_id,
      result = canonical_candidate
    ),
    smoke_path
  )
  cat("\nSmoke test completed successfully:", smoke_path, "\n")
  return(invisible(list(smoke = TRUE, path = smoke_path)))
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
  "Canonical m8trpg reconstruction max abs difference: %.3g\n",
  maximum_baseline_difference
))
if (maximum_baseline_difference > 1e-6) {
  stop(
    "Fold-refitted m8trpg does not reproduce the canonical OOF cache. ",
    "Stopping rather than comparing mismatched baselines."
  )
}

shallow_mlp <- load_mlp_from_cache(nrow(train))
if (is.null(shallow_mlp)) {
  cat(
    "Frozen shallow-MLP OOF cache not found; rebuilding the exact ",
    "8-unit/decay-0.1/200-iteration/five-seed component.\n",
    sep = ""
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
    file.path(output_dir, "shallow_mlp_oof_rebuilt.rds")
  )
}

flat_baseline <- validate_probability(
  0.85 * (
    0.80 * base_mlogit + 0.20 * original_xgb
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
flat_baseline_loss <- log_loss_matrix_local(truth, flat_baseline)
flat_candidate_loss <- log_loss_matrix_local(truth, flat_candidate)
if (abs(flat_baseline_loss - 1.143686618134879) > 1e-6) {
  stop(
    sprintf(
      paste0(
        "Frozen flat baseline mismatch: got %.12f, expected ",
        "1.143686618135. Stopping rather than changing the baseline."
      ),
      flat_baseline_loss
    )
  )
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
    baseline_logloss = flat_baseline_loss,
    candidate_logloss = flat_candidate_loss
  ),
  canonical_bootstrap$summary
)
canonical_summary$canonical_pass <-
  canonical_summary$point_gain > 0 &
  canonical_summary$lower_95 > 0
canonical_summary$near_miss <-
  canonical_summary$point_gain > 0 &
  canonical_summary$lower_95 <= 0 &
  canonical_summary$lower_95 >= near_miss_lower_limit

canonical_selection <- selection_tables(
  canonical_candidate$fold_results,
  "4821"
)
write.csv(
  canonical_summary,
  file.path(output_dir, "canonical_summary.csv"),
  row.names = FALSE
)
write.csv(
  canonical_selection$folds,
  file.path(output_dir, "canonical_folds.csv"),
  row.names = FALSE
)
write.csv(
  canonical_selection$path,
  file.path(output_dir, "canonical_selected_path.csv"),
  row.names = FALSE
)
saveRDS(
  list(
    experiment_id = experiment_id,
    summary = canonical_summary,
    bootstrap = canonical_bootstrap,
    baseline_prediction = flat_baseline,
    candidate_prediction = flat_candidate,
    candidate_mlogit = canonical_candidate$candidate,
    candidate_folds = canonical_candidate$fold_results
  ),
  file.path(output_dir, "canonical_result.rds")
)

cat("\nCanonical result:\n")
print(canonical_summary, digits = 9)

run_repeated <- repeated_mode == "always" ||
  (
    repeated_mode == "auto" &&
      isTRUE(canonical_summary$near_miss)
  )

if (!run_repeated) {
  verdict <- if (isTRUE(canonical_summary$canonical_pass)) {
    "PASS: ordinary canonical 95% lower bound excludes zero."
  } else if (
    canonical_summary$point_gain <= 0
  ) {
    "REJECT: candidate point gain is non-positive."
  } else if (
    canonical_summary$lower_95 < near_miss_lower_limit
  ) {
    paste0(
      "REJECT: canonical CI crosses zero by more than the ",
      "pre-registered near-miss allowance."
    )
  } else {
    paste0(
      "CANONICAL NEAR MISS, but repeated CV was disabled with ",
      "COMPBOOST_REPEATED=never."
    )
  }
  writeLines(
    verdict,
    file.path(output_dir, "verdict.txt")
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
  repeated_mode,
  ").\n",
  sep = ""
)

repeat_names <- c("4821", as.character(additional_seeds))
case_levels <- sort(unique(train$Case))
case_gain_matrix <- matrix(
  NA_real_,
  nrow = length(case_levels),
  ncol = length(repeat_names),
  dimnames = list(as.character(case_levels), repeat_names)
)
case_gain_matrix[, "4821"] <-
  canonical_bootstrap$respondent_gain
repeat_rows <- list(
  `4821` = data.frame(
    seed = 4821L,
    baseline_logloss = flat_baseline_loss,
    candidate_logloss = flat_candidate_loss,
    gain = flat_baseline_loss - flat_candidate_loss
  )
)
all_fold_tables <- list(`4821` = canonical_selection$folds)
all_path_tables <- list(`4821` = canonical_selection$path)

for (seed_index in seq_along(additional_seeds)) {
  seed <- additional_seeds[[seed_index]]
  tag <- paste0("seed_", seed)
  repeat_result_path <- file.path(
    output_dir,
    sprintf("repeat_seed_%d.rds", seed)
  )
  repeat_result <- NULL
  if (file.exists(repeat_result_path)) {
    saved <- readRDS(repeat_result_path)
    if (identical(saved$experiment_id, experiment_id)) {
      repeat_result <- saved
      cat("\nRepeat seed", seed, "loaded from checkpoint.\n")
    }
  }

  if (is.null(repeat_result)) {
    cat(sprintf(
      "\n######## repeated CV seed %d (%d/5) ########\n",
      seed, seed_index
    ))
    fold_map <- repeated_fold_map(case_levels, seed)
    candidate_oof <- run_candidate_oof(
      train,
      train_long,
      fold_map,
      tag
    )
    xgb_oof <- fit_original_xgb_oof(
      train,
      fold_map,
      tag
    )
    mlp_oof <- fit_shallow_mlp_oof(
      train,
      fold_map,
      repeat_index = seed_index,
      checkpoint_tag = tag
    )
    baseline_prediction <- validate_probability(
      0.85 * (
        0.80 * candidate_oof$baseline +
          0.20 * xgb_oof
      ) + 0.15 * mlp_oof,
      nrow(train)
    )
    candidate_prediction <- validate_probability(
      0.85 * (
        0.80 * candidate_oof$candidate +
          0.20 * xgb_oof
      ) + 0.15 * mlp_oof,
      nrow(train)
    )
    respondent_gain <- unname(tapply(
      row_log_loss_local(truth, baseline_prediction) -
        row_log_loss_local(truth, candidate_prediction),
      train$Case,
      mean
    ))
    repeat_result <- list(
      experiment_id = experiment_id,
      seed = seed,
      fold_map = fold_map,
      baseline_prediction = baseline_prediction,
      candidate_prediction = candidate_prediction,
      candidate_mlogit = candidate_oof$candidate,
      baseline_mlogit = candidate_oof$baseline,
      xgb_oof = xgb_oof,
      mlp_oof = mlp_oof,
      respondent_gain = respondent_gain,
      fold_results = candidate_oof$fold_results
    )
    saveRDS(repeat_result, repeat_result_path)
  }

  seed_name <- as.character(seed)
  case_gain_matrix[, seed_name] <-
    repeat_result$respondent_gain
  baseline_loss <- log_loss_matrix_local(
    truth, repeat_result$baseline_prediction
  )
  candidate_loss <- log_loss_matrix_local(
    truth, repeat_result$candidate_prediction
  )
  repeat_rows[[seed_name]] <- data.frame(
    seed = seed,
    baseline_logloss = baseline_loss,
    candidate_logloss = candidate_loss,
    gain = baseline_loss - candidate_loss
  )
  selection <- selection_tables(
    repeat_result$fold_results,
    seed_name
  )
  all_fold_tables[[seed_name]] <- selection$folds
  all_path_tables[[seed_name]] <- selection$path
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
  lower_95 = unname(quantile(pooled_bootstrap, 0.025)),
  upper_95 = unname(quantile(pooled_bootstrap, 0.975)),
  lower_99 = unname(quantile(pooled_bootstrap, 0.005)),
  upper_99 = unname(quantile(pooled_bootstrap, 0.995)),
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
  file.path(output_dir, "repeated_cv_by_seed.csv"),
  row.names = FALSE
)
write.csv(
  pooled_summary,
  file.path(output_dir, "repeated_cv_summary.csv"),
  row.names = FALSE
)
write.csv(
  do.call(rbind, all_fold_tables),
  file.path(output_dir, "repeated_cv_folds.csv"),
  row.names = FALSE
)
write.csv(
  do.call(rbind, all_path_tables),
  file.path(output_dir, "repeated_cv_selected_path.csv"),
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
  file.path(output_dir, "repeated_cv_result.rds")
)

verdict <- if (isTRUE(pooled_summary$promote)) {
  paste0(
    "PASS: repeated-CV pooled ordinary 95% lower bound excludes ",
    "zero and at least 5/6 repeats improve."
  )
} else {
  paste0(
    "REJECT: repeated-CV promotion rule was not fully satisfied; ",
    "the standing model remains unchanged."
  )
}
writeLines(verdict, file.path(output_dir, "verdict.txt"))

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

componentwise_result <- run_componentwise_experiment()
