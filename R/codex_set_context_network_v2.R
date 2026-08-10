# Permutation-equivariant set-context utility network as a diversity component
# for the frozen current best.
#
# Pre-registration:
#   codex_set_context_preregister.md
#
# Smoke test:
#   Sys.setenv(SET_CONTEXT_SMOKE = "1")
#   source("R/codex_set_context_network.R")
#   Sys.unsetenv("SET_CONTEXT_SMOKE")
#
# Full run:
#   Sys.unsetenv("SET_CONTEXT_SMOKE")
#   Sys.setenv(SET_CONTEXT_REPEATED = "auto")
#   source("R/codex_set_context_network.R")

options(stringsAsFactors = FALSE)

required_packages <- "torch"
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
suppressPackageStartupMessages(library(torch))

experiment_id <- "set_context_utility_network_v1"
output_dir <- file.path(
  "data_processed", "codex_set_context_network"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

smoke_mode <- identical(
  Sys.getenv("SET_CONTEXT_SMOKE", "0"), "1"
)
repeated_mode <- tolower(
  Sys.getenv("SET_CONTEXT_REPEATED", "auto")
)
stopifnot(repeated_mode %in% c("auto", "never", "always"))

canonical_baseline_target <- 1.143686618134879
submitted_crossfit_reference <- 1.14378944178118
near_miss_lower_limit <- -0.00075
additional_seeds <- c(
  1907L, 2719L, 6151L, 8293L, 104729L
)
blend_weight_grid <- seq(0, 0.25, by = 0.01)
network_config <- list(
  encoder_hidden = 64L,
  embedding_dim = 32L,
  encoder_dropout = 0.10,
  head_hidden = c(64L, 32L),
  head_dropout = 0.15,
  learning_rate = 0.001,
  weight_decay = 0.0005,
  epochs = 50L,
  batch_tasks = 256L
)
canonical_network_seeds <- 12501:12503
bootstrap_replicates <- if (smoke_mode) {
  200L
} else {
  as.integer(Sys.getenv("SET_CONTEXT_N_BOOT", "100000"))
}

input_files <- c(
  file.path("csv files", "train.csv"),
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

attrs <- c(
  "CC", "GN", "NS", "BU", "FA", "LD", "BZ", "FC", "FP", "RP",
  "PP", "KA", "SC", "TS", "NV", "MA", "LB", "AF", "HU"
)

torch::torch_set_num_threads(4L)

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
      "first; this set-context script will not silently rebuild a different ",
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

build_full_long <- function(wide) {
  pattern <- paste0(
    "^(",
    paste(c(attrs, "Price", "Ch"), collapse = "|"),
    ")([1-4])$"
  )
  fixed_names <- names(wide)[
    !grepl(pattern, names(wide))
  ]
  parts <- lapply(1:4, function(alternative) {
    output <- wide[, fixed_names, drop = FALSE]
    for (variable in c(attrs, "Price", "Ch")) {
      output[[variable]] <-
        wide[[paste0(variable, alternative)]]
    }
    output$alt <- alternative
    output$chosen <- as.integer(output$Ch == 1)
    output
  })
  output <- do.call(rbind, parts)
  output <- output[
    order(output$No, output$alt),
    ,
    drop = FALSE
  ]
  rownames(output) <- NULL
  output
}

add_choice_context <- function(long) {
  long <- long[
    order(long$No, long$alt),
    ,
    drop = FALSE
  ]
  rownames(long) <- NULL
  stopifnot(
    nrow(long) %% 4L == 0L,
    identical(
      as.integer(long$alt),
      rep(1:4, nrow(long) %/% 4L)
    )
  )
  long$inside <- as.integer(long$alt != 4L)
  long$d2 <- as.integer(long$alt == 2L)
  long$d3 <- as.integer(long$alt == 3L)
  long$Task_c <- (as.numeric(long$Task) - 10) / 9
  price <- as.numeric(long$Price)
  price_matrix <- matrix(
    price, ncol = 4L, byrow = TRUE
  )
  inside_price <- price_matrix[, 1:3, drop = FALSE]
  minimum_price <- apply(inside_price, 1L, min)
  maximum_price <- apply(inside_price, 1L, max)
  long$price_min <- rep(minimum_price, each = 4L)
  long$price_max <- rep(maximum_price, each = 4L)
  long$is_cheapest <- as.integer(
    long$inside == 1L &
      price == long$price_min
  )
  long$is_dearest <- as.integer(
    long$inside == 1L &
      price == long$price_max
  )
  long$price_gap_min <- ifelse(
    long$inside == 1L,
    price - long$price_min,
    0
  )
  long$price_gap_max <- ifelse(
    long$inside == 1L,
    long$price_max - price,
    0
  )
  long
}

profile_continuous <- c(
  "price_gap_min", "price_gap_max"
)
common_continuous <- c(
  "incomea", "agea", "milesa", "nighta",
  "genderind", "Urbind", "educind"
)

fit_scaler <- function(long, variables) {
  centre <- vapply(
    long[, variables, drop = FALSE],
    mean,
    numeric(1)
  )
  scale <- vapply(
    long[, variables, drop = FALSE],
    sd,
    numeric(1)
  )
  scale[!is.finite(scale) | scale == 0] <- 1
  list(centre = centre, scale = scale)
}

standardize_matrix <- function(long, variables, scaler) {
  matrix <- as.matrix(
    long[, variables, drop = FALSE]
  )
  storage.mode(matrix) <- "double"
  matrix <- sweep(
    sweep(matrix, 2L, scaler$centre, "-"),
    2L,
    scaler$scale,
    "/"
  )
  colnames(matrix) <- variables
  matrix
}

one_hot_matrix <- function(values, levels, prefix) {
  values <- as.integer(values)
  level_index <- match(values, levels)
  stopifnot(!anyNA(level_index))
  output <- matrix(
    0,
    nrow = length(values),
    ncol = length(levels)
  )
  output[
    cbind(seq_along(values), level_index)
  ] <- 1
  colnames(output) <- paste0(prefix, levels)
  output
}

attribute_levels <- function(training_long) {
  output <- lapply(attrs, function(attribute) {
    observed <- sort(unique(
      as.integer(training_long[[attribute]])
    ))
    stopifnot(
      length(observed) > 1L,
      identical(observed, 0:max(observed))
    )
    observed
  })
  names(output) <- attrs
  output
}

build_set_context_features <- function(
    long,
    levels_by_attribute,
    profile_scaler,
    common_scaler) {
  long <- add_choice_context(long)
  profile_parts <- list()
  for (attribute in attrs) {
    profile_parts[[attribute]] <- one_hot_matrix(
      long[[attribute]],
      levels_by_attribute[[attribute]],
      paste0(attribute, "_")
    )
  }
  profile_parts$Price <- one_hot_matrix(
    long$Price, 0:12, "Price_"
  )
  profile_parts$continuous <- standardize_matrix(
    long, profile_continuous, profile_scaler
  )
  profile_parts$flags <- as.matrix(
    long[
      ,
      c(
        "inside", "d2", "d3",
        "is_cheapest", "is_dearest"
      ),
      drop = FALSE
    ]
  )
  profile <- do.call(cbind, profile_parts)
  storage.mode(profile) <- "double"

  common_long <- cbind(
    one_hot_matrix(
      long$segmentind, 1:6, "segment_"
    ),
    one_hot_matrix(
      long$regionind, 1:5, "region_"
    ),
    one_hot_matrix(
      long$pparkind, 1:5, "ppark_"
    ),
    standardize_matrix(
      long, common_continuous, common_scaler
    ),
    Task_c = as.numeric(long$Task_c)
  )
  storage.mode(common_long) <- "double"

  first_rows <- seq.int(
    1L, nrow(long), by = 4L
  )
  common <- common_long[
    first_rows, , drop = FALSE
  ]
  reconstructed_common <- common[
    rep(seq_len(nrow(common)), each = 4L),
    ,
    drop = FALSE
  ]
  stopifnot(
    identical(
      colnames(common_long),
      colnames(reconstructed_common)
    ),
    max(abs(
      common_long - reconstructed_common
    )) < 1e-12
  )

  truth <- matrix(
    as.integer(long$chosen),
    ncol = 4L,
    byrow = TRUE
  )
  task_no <- as.integer(long$No[first_rows])
  stopifnot(
    nrow(profile) == nrow(common) * 4L,
    all(rowSums(truth) == 1L),
    length(unique(task_no)) == length(task_no),
    all(is.finite(profile)),
    all(is.finite(common))
  )
  list(
    profile = profile,
    common = common,
    truth = truth,
    no = task_no,
    profile_columns = colnames(profile),
    common_columns = colnames(common)
  )
}

set_context_net <- nn_module(
  "set_context_net",
  initialize = function(
      profile_dim,
      common_dim,
      config) {
    self$profile_dim <- as.integer(profile_dim)
    self$common_dim <- as.integer(common_dim)
    self$embedding_dim <-
      as.integer(config$embedding_dim)
    self$encoder <- nn_sequential(
      nn_linear(
        self$profile_dim,
        as.integer(config$encoder_hidden)
      ),
      nn_relu(),
      nn_dropout(p = config$encoder_dropout),
      nn_linear(
        as.integer(config$encoder_hidden),
        self$embedding_dim
      ),
      nn_relu()
    )
    head_input_dim <-
      4L * self$embedding_dim +
      self$common_dim
    self$head_input_dim <-
      as.integer(head_input_dim)
    self$head <- nn_sequential(
      nn_linear(
        self$head_input_dim,
        as.integer(config$head_hidden[[1L]])
      ),
      nn_relu(),
      nn_dropout(p = config$head_dropout),
      nn_linear(
        as.integer(config$head_hidden[[1L]]),
        as.integer(config$head_hidden[[2L]])
      ),
      nn_relu(),
      nn_linear(
        as.integer(config$head_hidden[[2L]]),
        1L
      )
    )
  },
  forward = function(profile, common) {
    n_tasks <- profile$size(1L)
    embedding <- self$encoder(
      profile$reshape(c(
        -1L, self$profile_dim
      ))
    )$reshape(c(
      n_tasks, 4L, self$embedding_dim
    ))
    inside_embedding <- embedding[, 1:3, ]
    mean_inside <- inside_embedding$mean(dim = 2L)
    max_inside <- torch_maximum(
      torch_maximum(
        inside_embedding[, 1L, ],
        inside_embedding[, 2L, ]
      ),
      inside_embedding[, 3L, ]
    )
    mean_repeated <- torch_stack(
      list(mean_inside, mean_inside, mean_inside, mean_inside),
      dim = 2L
    )
    max_repeated <- torch_stack(
      list(max_inside, max_inside, max_inside, max_inside),
      dim = 2L
    )
    common_repeated <- torch_stack(
      list(common, common, common, common),
      dim = 2L
    )
    combined <- torch_cat(
      list(
        embedding,
        mean_repeated,
        max_repeated,
        embedding - mean_repeated,
        common_repeated
      ),
      dim = 3L
    )
    self$head(
      combined$reshape(c(
        -1L, self$head_input_dim
      ))
    )$reshape(c(n_tasks, 4L))
  }
)

predict_set_context <- function(
    model,
    profile,
    common,
    batch_tasks = 2048L) {
  n_tasks <- nrow(common)
  stopifnot(nrow(profile) == n_tasks * 4L)
  profile_tensor <- torch_tensor(
    profile,
    dtype = torch_float()
  )$reshape(c(
    n_tasks, 4L, ncol(profile)
  ))
  common_tensor <- torch_tensor(
    common,
    dtype = torch_float()
  )
  prediction <- matrix(
    NA_real_, n_tasks, 4L
  )
  model$eval()
  with_no_grad({
    for (
      start in seq.int(
        1L, n_tasks, by = batch_tasks
      )
    ) {
      stop_at <- min(
        n_tasks,
        start + batch_tasks - 1L
      )
      probability <- nnf_softmax(
        model(
          profile_tensor[start:stop_at, , ],
          common_tensor[start:stop_at, ]
        ),
        dim = 2L
      )
      prediction[start:stop_at, ] <-
        as.matrix(as_array(probability))
    }
  })
  validate_probability(prediction, n_tasks)
}

fit_set_context_once <- function(
    train_features,
    validation_features,
    config,
    seed,
    checkpoint = NULL) {
  if (
    !is.null(checkpoint) &&
    file.exists(checkpoint)
  ) {
    saved <- readRDS(checkpoint)
    if (
      identical(saved$experiment_id, experiment_id) &&
      identical(saved$seed, as.integer(seed)) &&
      identical(saved$config, config) &&
      identical(
        as.integer(saved$validation_no),
        as.integer(validation_features$no)
      )
    ) {
      return(saved$result)
    }
  }

  set.seed(as.integer(seed))
  torch_manual_seed(as.integer(seed))
  n_tasks <- nrow(train_features$common)
  stopifnot(
    nrow(train_features$profile) ==
      n_tasks * 4L,
    nrow(train_features$truth) == n_tasks
  )
  profile_tensor <- torch_tensor(
    train_features$profile,
    dtype = torch_float()
  )$reshape(c(
    n_tasks,
    4L,
    ncol(train_features$profile)
  ))
  common_tensor <- torch_tensor(
    train_features$common,
    dtype = torch_float()
  )
  target_tensor <- torch_tensor(
    as.integer(max.col(train_features$truth)),
    dtype = torch_long()
  )
  model <- set_context_net(
    profile_dim =
      ncol(train_features$profile),
    common_dim =
      ncol(train_features$common),
    config = config
  )
  optimizer <- optim_adam(
    model$parameters,
    lr = config$learning_rate,
    weight_decay = config$weight_decay
  )
  trace_rows <- list()
  started <- proc.time()[["elapsed"]]
  for (epoch in seq_len(config$epochs)) {
    model$train()
    task_order <- sample.int(n_tasks)
    total_loss <- 0
    total_tasks <- 0L
    for (
      start in seq.int(
        1L,
        n_tasks,
        by = config$batch_tasks
      )
    ) {
      stop_at <- min(
        n_tasks,
        start + config$batch_tasks - 1L
      )
      task_index <- task_order[start:stop_at]
      optimizer$zero_grad()
      utility <- model(
        profile_tensor[task_index, , ],
        common_tensor[task_index, ]
      )
      loss <- nnf_cross_entropy(
        utility,
        target_tensor[task_index]
      )
      loss$backward()
      optimizer$step()
      count <- length(task_index)
      total_loss <-
        total_loss + loss$item() * count
      total_tasks <- total_tasks + count
    }
    if (
      epoch == 1L ||
      epoch %% 10L == 0L ||
      epoch == config$epochs
    ) {
      trace_rows[[
        length(trace_rows) + 1L
      ]] <- data.frame(
        epoch = epoch,
        training_logloss =
          total_loss / total_tasks
      )
    }
  }
  elapsed <-
    proc.time()[["elapsed"]] - started
  prediction <- predict_set_context(
    model,
    validation_features$profile,
    validation_features$common
  )
  result <- list(
    prediction = prediction,
    validation_no = validation_features$no,
    seed = as.integer(seed),
    elapsed_seconds = elapsed,
    final_training_logloss =
      total_loss / total_tasks,
    trace = do.call(rbind, trace_rows)
  )
  if (!is.null(checkpoint)) {
    saveRDS(
      list(
        experiment_id = experiment_id,
        seed = as.integer(seed),
        config = config,
        validation_no =
          validation_features$no,
        result = result
      ),
      checkpoint
    )
  }
  result
}

network_seeds_for_fold <- function(
    outer_seed,
    outer_fold) {
  seed_registry <- c(
    4821L, additional_seeds
  )
  seed_index <- match(
    as.integer(outer_seed),
    seed_registry
  )
  if (is.na(seed_index)) {
    stop("Outer seed is not pre-registered.")
  }
  as.integer(
    canonical_network_seeds +
      (seed_index - 1L) * 1000L +
      outer_fold * 10L
  )
}

fit_outer_set_context <- function(
    fitting,
    validation,
    outer_seed,
    outer_fold,
    config = network_config,
    seeds = NULL,
    checkpoint_tag = NULL) {
  stopifnot(
    length(intersect(
      unique(fitting$Case),
      unique(validation$Case)
    )) == 0L
  )
  fitting_long <- add_choice_context(
    build_full_long(fitting)
  )
  validation_long <- add_choice_context(
    build_full_long(validation)
  )
  levels_by_attribute <- attribute_levels(
    fitting_long
  )
  profile_scaler <- fit_scaler(
    fitting_long,
    profile_continuous
  )
  common_scaler <- fit_scaler(
    fitting_long,
    common_continuous
  )
  training_features <- build_set_context_features(
    fitting_long,
    levels_by_attribute,
    profile_scaler,
    common_scaler
  )
  validation_features <-
    build_set_context_features(
      validation_long,
      levels_by_attribute,
      profile_scaler,
      common_scaler
    )
  stopifnot(
    identical(
      training_features$profile_columns,
      validation_features$profile_columns
    ),
    identical(
      training_features$common_columns,
      validation_features$common_columns
    )
  )
  if (is.null(seeds)) {
    seeds <- network_seeds_for_fold(
      outer_seed,
      outer_fold
    )
  }
  prediction_array <- array(
    NA_real_,
    dim = c(
      nrow(validation_features$truth),
      4L,
      length(seeds)
    )
  )
  fit_rows <- list()
  for (seed_index in seq_along(seeds)) {
    seed <- seeds[[seed_index]]
    checkpoint <- if (
      is.null(checkpoint_tag)
    ) {
      NULL
    } else {
      file.path(
        output_dir,
        sprintf(
          "network_%s_fold_%d_seed_%d.rds",
          checkpoint_tag,
          outer_fold,
          seed
        )
      )
    }
    fitted <- fit_set_context_once(
      training_features,
      validation_features,
      config,
      seed,
      checkpoint
    )
    prediction_array[, , seed_index] <-
      fitted$prediction
    fit_rows[[seed_index]] <- data.frame(
      outer_seed = outer_seed,
      outer_fold = outer_fold,
      seed = seed,
      profile_features =
        ncol(training_features$profile),
      common_features =
        ncol(training_features$common),
      elapsed_seconds =
        fitted$elapsed_seconds,
      final_training_logloss =
        fitted$final_training_logloss
    )
    cat(sprintf(
      "  seed %d: train %.6f, %.1fs\n",
      seed,
      fitted$final_training_logloss,
      fitted$elapsed_seconds
    ))
    flush.console()
  }
  prediction <- apply(
    prediction_array,
    c(1L, 2L),
    mean
  )
  list(
    validation_no =
      validation_features$no,
    prediction = validate_probability(
      prediction,
      nrow(validation_features$truth)
    ),
    seeds = as.integer(seeds),
    fits = do.call(rbind, fit_rows),
    profile_columns =
      training_features$profile_columns,
    common_columns =
      training_features$common_columns
  )
}

run_set_context_oof <- function(
    train,
    fold_map,
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
      "\n=== %s: set-context outer fold %d/5 ===\n",
      checkpoint_tag,
      outer_fold
    ))
    validation_rows <-
      row_fold == outer_fold
    result <- fit_outer_set_context(
      train[
        !validation_rows,
        ,
        drop = FALSE
      ],
      train[
        validation_rows,
        ,
        drop = FALSE
      ],
      outer_seed,
      outer_fold,
      config = network_config,
      seeds = NULL,
      checkpoint_tag = checkpoint_tag
    )
    rows <- match(
      result$validation_no,
      train$No
    )
    stopifnot(
      !anyNA(rows),
      all(row_fold[rows] == outer_fold),
      length(rows) == sum(validation_rows)
    )
    prediction[rows, ] <-
      result$prediction
    fold_results[[outer_fold]] <- result
    cat(sprintf(
      "  fold %d complete: %d seeds\n",
      outer_fold,
      length(result$seeds)
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
      set_context_weight = best_weight,
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
        n_network_seeds =
          length(result$seeds),
        profile_features =
          length(result$profile_columns),
        common_features =
          length(result$common_columns),
        mean_training_logloss =
          mean(result$fits$final_training_logloss),
        total_elapsed_seconds =
          sum(result$fits$elapsed_seconds),
        set_context_weight =
          weights$set_context_weight[
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

run_smoke_test <- function(train) {
  saved <- readRDS(file.path(
    "data_processed", "oof_ensemble_v10.rds"
  ))
  fold_map <- canonical_fold_map(train, saved)
  outer_fold <- 1L
  fitting_cases <- as.integer(names(
    fold_map[fold_map != outer_fold]
  ))
  validation_cases <- as.integer(names(
    fold_map[fold_map == outer_fold]
  ))
  fitting <- train[
    train$Case %in% fitting_cases, , drop = FALSE
  ]
  validation <- train[
    train$Case %in% validation_cases, , drop = FALSE
  ]
  cat(
    "SMOKE MODE: canonical outer fold 1, ",
    "one network seed, two epochs; no verdict.\n",
    sep = ""
  )
  smoke_config <- network_config
  smoke_config$epochs <- 2L
  result <- fit_outer_set_context(
    fitting,
    validation,
    outer_seed = 4821L,
    outer_fold = 1L,
    config = smoke_config,
    seeds = network_seeds_for_fold(
      4821L, 1L
    )[[1L]],
    checkpoint_tag = "smoke"
  )
  validation_rows <- match(
    result$validation_no,
    validation$No
  )
  stopifnot(!anyNA(validation_rows))
  truth <- as.matrix(
    validation[
      validation_rows,
      paste0("Ch", 1:4),
      drop = FALSE
    ]
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
      "Smoke set-context log loss: %.6f\n"
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
  truth <- as.matrix(
    train[, paste0("Ch", 1:4), drop = FALSE]
  )
  stopifnot(
    nrow(train) == 21565L,
    length(unique(train$Case)) == 1135L,
    all(table(train$Case) == 19L),
    all(rowSums(truth) == 1L)
  )

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
    return(run_smoke_test(train))
  }

  saved <- readRDS(file.path(
    "data_processed", "oof_ensemble_v10.rds"
  ))
  fold_map <- canonical_fold_map(train, saved)
  baseline <- canonical_baseline(
    train, truth, saved
  )
  set_context_oof <- run_set_context_oof(
    train,
    fold_map,
    checkpoint_tag = "seed_4821",
    outer_seed = 4821L
  )
  blended <- crossfit_blend(
    truth,
    baseline$prediction,
    set_context_oof$prediction,
    set_context_oof$row_fold
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
      set_context_component_logloss =
        log_loss_matrix_local(
          truth, set_context_oof$prediction
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
    set_context_oof$fold_results,
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
      set_context_prediction =
        set_context_oof$prediction,
      candidate_prediction = blended$prediction,
      fold_map = fold_map,
      fold_results =
        set_context_oof$fold_results,
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
      paste0(
        "Repeated CV disabled by ",
        "SET_CONTEXT_REPEATED=never."
      )
    } else {
      paste0(
        "Repeated CV not triggered: canonical result ",
        "was neither a pass nor the pre-registered near miss."
      )
    }
    verdict <- paste(
      "REJECT set-context utility network.",
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
      set_context_component_logloss =
        log_loss_matrix_local(
          truth, set_context_oof$prediction
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
    repeat_set_context <-
      run_set_context_oof(
      train,
      repeat_baseline$fold_map,
      checkpoint_tag = paste0("seed_", seed),
      outer_seed = seed
    )
    repeat_blend <- crossfit_blend(
      truth,
      repeat_baseline$prediction,
      repeat_set_context$prediction,
      repeat_set_context$row_fold
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
      set_context_component_logloss =
        log_loss_matrix_local(
          truth,
          repeat_set_context$prediction
        ),
      candidate_logloss = repeat_candidate_loss,
      gain = repeat_baseline$logloss -
        repeat_candidate_loss
    )
    selection_rows[[seed_name]] <-
      fold_selection_table(
        repeat_set_context$fold_results,
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
        set_context_prediction =
          repeat_set_context$prediction,
        candidate_prediction =
          repeat_blend$prediction,
        weights = repeat_blend$weights,
        fold_results =
          repeat_set_context$fold_results,
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
      "PROMOTE set-context utility network for ",
      "a separate full-data build audit. Do not submit until ",
      "that artifact is reproduced and checked."
    )
  } else {
    verdict <- paste0(
      "REJECT set-context utility network; retain ",
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
