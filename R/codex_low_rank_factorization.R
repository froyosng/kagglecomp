# Low-rank demographic-conditioned safety part-worth correction around the
# frozen m8trpg utility, evaluated as a diversity component for v14.
#
# Pre-registration:
#   codex_low_rank_factorization_preregister.md
#
# Smoke test:
#   Sys.setenv(LOW_RANK_SMOKE = "1")
#   source("R/codex_low_rank_factorization.R")
#   Sys.unsetenv("LOW_RANK_SMOKE")
#
# Full run:
#   Sys.unsetenv("LOW_RANK_SMOKE")
#   Sys.setenv(LOW_RANK_REPEATED = "auto")
#   source("R/codex_low_rank_factorization.R")

options(stringsAsFactors = FALSE)

required_packages <- c("torch", "mlogit", "dfidx")
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

experiment_id <- "low_rank_demographic_partworth_v1"
output_dir <- file.path(
  "data_processed", "codex_low_rank_factorization"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

smoke_mode <- identical(
  Sys.getenv("LOW_RANK_SMOKE", "0"), "1"
)
repeated_mode <- tolower(
  Sys.getenv("LOW_RANK_REPEATED", "auto")
)
stopifnot(repeated_mode %in% c("auto", "never", "always"))

canonical_seed <- 4821L
additional_seeds <- c(
  1907L, 2719L, 6151L, 8293L, 104729L
)
all_outer_seeds <- c(canonical_seed, additional_seeds)
near_miss_lower_limit <- -0.00075
blend_weight_grid <- seq(0, 0.25, by = 0.01)
expected_v14_repeated_gain <- 0.0011636195
expected_v14_repeated_lower <- 0.0000105775
v14_metric_tolerance <- 5e-8

low_rank_config <- list(
  rank = 4L,
  learning_rate = 0.002,
  weight_decay = 0.01,
  epochs = 80L,
  batch_tasks = 512L,
  gradient_clip = 5
)
canonical_network_seeds <- 15401:15403
bootstrap_replicates <- if (smoke_mode) {
  200L
} else {
  as.integer(Sys.getenv("LOW_RANK_N_BOOT", "100000"))
}

attrs <- c(
  "CC", "GN", "NS", "BU", "FA", "LD", "BZ", "FC", "FP", "RP",
  "PP", "KA", "SC", "TS", "NV", "MA", "LB", "AF", "HU"
)
respondent_continuous <- c(
  "incomea", "agea", "milesa", "nighta",
  "genderind", "Urbind", "educind"
)
domain_covariates <- c(
  "segmentind", "yearind", "milesind", "milesa",
  "nightind", "nighta", "pparkind", "genderind",
  "ageind", "agea", "educind", "regionind",
  "Urbind", "incomeind", "incomea"
)

v14_dir <- file.path(
  "data_processed", "codex_set_context_network"
)
input_files <- c(
  file.path("csv files", "train.csv"),
  file.path("csv files", "test.csv"),
  file.path("R", "codex_shift_common.R"),
  file.path("R", "codex_modeling_common.R"),
  file.path(v14_dir, "canonical_result.rds")
)
missing_files <- input_files[!file.exists(input_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing required project file(s):\n  ",
    paste(missing_files, collapse = "\n  "),
    "\nRun this script from the kagglecomp repository root after ",
    "the completed v14 set-context experiment."
  )
}

torch::torch_set_num_threads(4L)

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

log_loss_matrix_local <- function(truth, prediction) {
  prediction <- validate_probability(
    prediction, nrow(truth)
  )
  -mean(rowSums(
    as.matrix(truth) * log(pmax(prediction, 1e-15))
  ))
}

row_log_loss_local <- function(truth, prediction) {
  prediction <- validate_probability(
    prediction, nrow(truth)
  )
  -rowSums(
    as.matrix(truth) * log(pmax(prediction, 1e-15))
  )
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
      c(
        floor(length(cases) / folds),
        ceiling(length(cases) / folds)
      ))
  )
  assignment
}

validate_fold_map <- function(train, fold_map) {
  stopifnot(
    !is.null(names(fold_map)),
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

numeric_signature <- function(x) {
  x <- as.numeric(as.matrix(x))
  index <- ((seq_along(x) - 1L) %% 997L) + 1L
  c(
    length = length(x),
    sum = sum(x),
    sum_square = sum(x * x),
    indexed_sum = sum(x * index)
  )
}

same_signature <- function(a, b, tolerance = 1e-10) {
  identical(names(a), names(b)) &&
    length(a) == length(b) &&
    max(abs(as.numeric(a) - as.numeric(b))) <= tolerance
}

m8trpg_helper_environment <- function() {
  helper <- new.env(parent = globalenv())
  helper$source <- function(file, local = TRUE, ...) {
    if (identical(local, FALSE)) {
      stop(
        "A nested m8trpg helper tried to source into the global ",
        "environment."
      )
    }
    base::source(file, local = helper, ...)
  }
  sys.source(
    file.path("R", "codex_shift_common.R"),
    envir = helper
  )
  rm("source", envir = helper)
  required <- c(
    "reshape_choice_long", "choice_scaler",
    "make_m8trpg_features", "m8trpg_formula"
  )
  stopifnot(vapply(
    required,
    exists,
    logical(1),
    envir = helper,
    inherits = FALSE
  ))
  helper
}

softmax_eta <- function(eta) {
  eta <- as.matrix(eta)
  eta <- eta - apply(eta, 1L, max)
  output <- exp(eta)
  validate_probability(
    output / rowSums(output),
    nrow(output)
  )
}

predict_m8trpg_model <- function(
    model,
    features,
    formula) {
  indexed <- dfidx::dfidx(
    features,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen"
  )
  class(indexed) <- c(
    "dfidx_mlogit", class(indexed)
  )
  frame <- model.frame(
    indexed, formula, balanced = TRUE
  )
  design <- model.matrix(
    frame, rhs = 1:3
  )
  beta <- stats::coef(model)
  if (!all(names(beta) %in% colnames(design))) {
    stop(
      "m8trpg prediction matrix is missing fitted columns: ",
      paste(
        setdiff(names(beta), colnames(design)),
        collapse = ", "
      )
    )
  }
  eta <- as.numeric(
    design[, names(beta), drop = FALSE] %*% beta
  )
  task_map <- unique(
    features[, c("chid", "No"), drop = FALSE]
  )
  task_map <- task_map[
    order(task_map$No), , drop = FALSE
  ]
  eta_chid <- as.character(dfidx::idx(frame, 1))
  eta_alt <- as.integer(
    as.character(dfidx::idx(frame, 2))
  )
  eta_order <- order(
    match(eta_chid, task_map$chid),
    eta_alt
  )
  stopifnot(
    identical(
      eta_chid[eta_order],
      rep(task_map$chid, each = 4L)
    ),
    identical(
      eta_alt[eta_order],
      rep(1:4, times = nrow(task_map))
    )
  )
  list(
    prediction = softmax_eta(
      matrix(
        eta[eta_order],
        ncol = 4L,
        byrow = TRUE
      )
    ),
    no = as.integer(task_map$No)
  )
}

fit_m8trpg_offset <- function(
    fitting,
    validation,
    helper) {
  stopifnot(
    length(intersect(
      unique(fitting$Case),
      unique(validation$Case)
    )) == 0L
  )
  fitting_long <- helper$reshape_choice_long(
    fitting
  )
  validation_long <- helper$reshape_choice_long(
    validation
  )
  scaler <- helper$choice_scaler(fitting_long)
  fitting_features <- helper$make_m8trpg_features(
    fitting_long,
    scaler$ctr,
    scaler$scl,
    "none"
  )
  validation_features <-
    helper$make_m8trpg_features(
      validation_long,
      scaler$ctr,
      scaler$scl,
      "none"
    )
  formula <- helper$m8trpg_formula("none")
  environment(formula) <- helper
  model <- mlogit::mlogit(
    formula,
    data = fitting_features,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen"
  )
  fitting_prediction <- predict_m8trpg_model(
    model, fitting_features, formula
  )
  validation_prediction <- predict_m8trpg_model(
    model, validation_features, formula
  )
  list(
    fitting_long = fitting_long,
    validation_long = validation_long,
    fitting_prediction =
      fitting_prediction$prediction,
    validation_prediction =
      validation_prediction$prediction,
    fitting_no = fitting_prediction$no,
    validation_no = validation_prediction$no,
    coefficient_count = length(stats::coef(model))
  )
}

reference_dummies <- function(
    values,
    active_levels,
    prefix) {
  values <- as.integer(values)
  stopifnot(
    all(values %in% c(1L, active_levels))
  )
  output <- vapply(
    active_levels,
    function(level) {
      as.numeric(values == level)
    },
    numeric(length(values))
  )
  if (is.null(dim(output))) {
    output <- matrix(output, ncol = 1L)
  }
  colnames(output) <- paste0(
    prefix, active_levels
  )
  output
}

raw_respondent_matrix <- function(data) {
  output <- cbind(
    reference_dummies(
      data$segmentind, 2:6, "segment_"
    ),
    reference_dummies(
      data$regionind, 2:5, "region_"
    ),
    reference_dummies(
      data$pparkind, 2:5, "ppark_"
    ),
    as.matrix(
      data[
        ,
        respondent_continuous,
        drop = FALSE
      ]
    )
  )
  storage.mode(output) <- "double"
  stopifnot(
    !anyNA(output),
    all(is.finite(output))
  )
  output
}

fit_respondent_scaler <- function(wide) {
  respondents <- wide[
    !duplicated(wide$Case), , drop = FALSE
  ]
  raw <- raw_respondent_matrix(respondents)
  centre <- colMeans(raw)
  scale <- apply(raw, 2L, sd)
  scale[!is.finite(scale) | scale == 0] <- 1
  list(
    centre = centre,
    scale = scale,
    columns = colnames(raw)
  )
}

standardize_respondents <- function(data, scaler) {
  raw <- raw_respondent_matrix(data)
  stopifnot(
    identical(colnames(raw), scaler$columns)
  )
  output <- sweep(
    sweep(raw, 2L, scaler$centre, "-"),
    2L,
    scaler$scale,
    "/"
  )
  storage.mode(output) <- "double"
  stopifnot(all(is.finite(output)))
  output
}

active_attribute_levels <- function(long) {
  output <- lapply(attrs, function(attribute) {
    values <- sort(unique(
      as.integer(long[[attribute]])
    ))
    stopifnot(
      0L %in% values,
      max(values) >= 1L,
      identical(values, 0:max(values))
    )
    seq_len(max(values))
  })
  names(output) <- attrs
  output
}

active_one_hot <- function(
    values,
    levels,
    prefix) {
  values <- as.integer(values)
  stopifnot(all(values %in% c(0L, levels)))
  output <- vapply(
    levels,
    function(level) {
      as.numeric(values == level)
    },
    numeric(length(values))
  )
  if (is.null(dim(output))) {
    output <- matrix(output, ncol = 1L)
  }
  colnames(output) <- paste0(prefix, levels)
  output
}

build_low_rank_features <- function(
    long,
    levels_by_attribute,
    respondent_scaler) {
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
  profile_parts <- lapply(attrs, function(attribute) {
    active_one_hot(
      long[[attribute]],
      levels_by_attribute[[attribute]],
      paste0(attribute, "_")
    )
  })
  names(profile_parts) <- attrs
  profile <- do.call(cbind, profile_parts)
  storage.mode(profile) <- "double"

  first_rows <- seq.int(
    1L, nrow(long), by = 4L
  )
  respondent <- standardize_respondents(
    long[first_rows, , drop = FALSE],
    respondent_scaler
  )
  truth <- matrix(
    as.integer(long$chosen),
    ncol = 4L,
    byrow = TRUE
  )
  no <- as.integer(long$No[first_rows])
  opt_out_rows <- seq.int(
    4L, nrow(profile), by = 4L
  )
  stopifnot(
    nrow(profile) == nrow(respondent) * 4L,
    all(rowSums(truth) == 1L),
    length(unique(no)) == length(no),
    max(abs(profile[opt_out_rows, , drop = FALSE])) == 0,
    all(is.finite(profile)),
    all(is.finite(respondent))
  )
  list(
    profile = profile,
    respondent = respondent,
    truth = truth,
    no = no,
    profile_columns = colnames(profile),
    respondent_columns = colnames(respondent)
  )
}

low_rank_net <- nn_module(
  "low_rank_net",
  initialize = function(
      profile_dim,
      respondent_dim,
      rank) {
    self$profile_dim <- as.integer(profile_dim)
    self$respondent_dim <-
      as.integer(respondent_dim)
    self$rank <- as.integer(rank)
    self$profile_projection <- nn_linear(
      self$profile_dim,
      self$rank,
      bias = FALSE
    )
    self$respondent_projection <- nn_linear(
      self$respondent_dim,
      self$rank,
      bias = FALSE
    )
  },
  forward = function(
      profile,
      respondent,
      offset_log_probability) {
    n_tasks <- profile$size(1L)
    bundle_embedding <- self$profile_projection(
      profile$reshape(c(
        -1L, self$profile_dim
      ))
    )$reshape(c(
      n_tasks, 4L, self$rank
    ))
    respondent_embedding <-
      self$respondent_projection(respondent)
    respondent_repeated <- torch_stack(
      list(
        respondent_embedding,
        respondent_embedding,
        respondent_embedding,
        respondent_embedding
      ),
      dim = 2L
    )
    correction <- (
      respondent_repeated * bundle_embedding
    )$sum(dim = 3L) / sqrt(as.numeric(self$rank))
    offset_log_probability + correction
  }
)

predict_low_rank <- function(
    model,
    features,
    offset_prediction,
    batch_tasks = 2048L) {
  offset_prediction <- validate_probability(
    offset_prediction,
    nrow(features$truth)
  )
  n_tasks <- nrow(features$respondent)
  stopifnot(
    nrow(features$profile) == n_tasks * 4L
  )
  profile_tensor <- torch_tensor(
    features$profile,
    dtype = torch_float()
  )$reshape(c(
    n_tasks,
    4L,
    ncol(features$profile)
  ))
  respondent_tensor <- torch_tensor(
    features$respondent,
    dtype = torch_float()
  )
  offset_tensor <- torch_tensor(
    log(pmax(offset_prediction, 1e-15)),
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
          respondent_tensor[start:stop_at, ],
          offset_tensor[start:stop_at, ]
        ),
        dim = 2L
      )
      prediction[start:stop_at, ] <-
        as.matrix(as_array(probability))
    }
  })
  validate_probability(prediction, n_tasks)
}

fit_low_rank_once <- function(
    training_features,
    validation_features,
    training_offset,
    validation_offset,
    config,
    seed) {
  set.seed(as.integer(seed))
  torch_manual_seed(as.integer(seed))
  n_tasks <- nrow(training_features$respondent)
  stopifnot(
    nrow(training_features$profile) ==
      n_tasks * 4L,
    nrow(training_features$truth) == n_tasks
  )
  training_offset <- validate_probability(
    training_offset, n_tasks
  )
  profile_tensor <- torch_tensor(
    training_features$profile,
    dtype = torch_float()
  )$reshape(c(
    n_tasks,
    4L,
    ncol(training_features$profile)
  ))
  respondent_tensor <- torch_tensor(
    training_features$respondent,
    dtype = torch_float()
  )
  offset_tensor <- torch_tensor(
    log(pmax(training_offset, 1e-15)),
    dtype = torch_float()
  )
  target_tensor <- torch_tensor(
    as.integer(
      max.col(training_features$truth)
    ),
    dtype = torch_long()
  )
  model <- low_rank_net(
    profile_dim =
      ncol(training_features$profile),
    respondent_dim =
      ncol(training_features$respondent),
    rank = config$rank
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
        respondent_tensor[task_index, ],
        offset_tensor[task_index, ]
      )
      loss <- nnf_cross_entropy(
        utility,
        target_tensor[task_index]
      )
      loss$backward()
      nn_utils_clip_grad_norm_(
        model$parameters,
        max_norm = config$gradient_clip
      )
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
  prediction <- predict_low_rank(
    model,
    validation_features,
    validation_offset
  )
  list(
    prediction = prediction,
    seed = as.integer(seed),
    elapsed_seconds = elapsed,
    final_training_logloss =
      total_loss / total_tasks,
    trace = do.call(rbind, trace_rows)
  )
}

network_seeds_for_fold <- function(
    outer_seed,
    outer_fold) {
  seed_index <- match(
    as.integer(outer_seed),
    all_outer_seeds
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

fit_outer_low_rank <- function(
    fitting,
    validation,
    helper,
    outer_seed,
    outer_fold,
    config = low_rank_config,
    seeds = NULL,
    checkpoint_tag = NULL) {
  stopifnot(
    length(intersect(
      unique(fitting$Case),
      unique(validation$Case)
    )) == 0L
  )
  if (is.null(seeds)) {
    seeds <- network_seeds_for_fold(
      outer_seed, outer_fold
    )
  }
  checkpoint <- if (
    is.null(checkpoint_tag)
  ) {
    NULL
  } else {
    file.path(
      output_dir,
      sprintf(
        "outer_%s_fold_%d.rds",
        checkpoint_tag,
        outer_fold
      )
    )
  }
  expected_validation_no <- sort(
    as.integer(validation$No)
  )
  expected_fitting_cases <- sort(
    unique(as.integer(fitting$Case))
  )
  if (
    !is.null(checkpoint) &&
    file.exists(checkpoint)
  ) {
    saved <- readRDS(checkpoint)
    if (
      identical(saved$experiment_id, experiment_id) &&
      identical(saved$config, config) &&
      identical(saved$seeds, as.integer(seeds)) &&
      identical(
        saved$validation_no,
        expected_validation_no
      ) &&
      identical(
        saved$fitting_cases,
        expected_fitting_cases
      )
    ) {
      cat("  using verified outer-fold checkpoint\n")
      return(saved$result)
    }
  }

  offset <- fit_m8trpg_offset(
    fitting, validation, helper
  )
  levels_by_attribute <- active_attribute_levels(
    offset$fitting_long
  )
  respondent_scaler <- fit_respondent_scaler(
    fitting
  )
  training_features <- build_low_rank_features(
    offset$fitting_long,
    levels_by_attribute,
    respondent_scaler
  )
  validation_features <- build_low_rank_features(
    offset$validation_long,
    levels_by_attribute,
    respondent_scaler
  )
  stopifnot(
    identical(
      training_features$profile_columns,
      validation_features$profile_columns
    ),
    identical(
      training_features$respondent_columns,
      validation_features$respondent_columns
    )
  )

  training_order <- match(
    training_features$no,
    offset$fitting_no
  )
  validation_order <- match(
    validation_features$no,
    offset$validation_no
  )
  stopifnot(
    !anyNA(training_order),
    !anyNA(validation_order),
    length(unique(training_order)) ==
      length(training_order),
    length(unique(validation_order)) ==
      length(validation_order)
  )
  training_offset <-
    offset$fitting_prediction[
      training_order, , drop = FALSE
    ]
  validation_offset <-
    offset$validation_prediction[
      validation_order, , drop = FALSE
    ]

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
    fitted <- fit_low_rank_once(
      training_features,
      validation_features,
      training_offset,
      validation_offset,
      config,
      seed
    )
    prediction_array[, , seed_index] <-
      fitted$prediction
    fit_rows[[seed_index]] <- data.frame(
      outer_seed = outer_seed,
      outer_fold = outer_fold,
      seed = seed,
      profile_features =
        ncol(training_features$profile),
      respondent_features =
        ncol(training_features$respondent),
      rank = config$rank,
      m8trpg_coefficients =
        offset$coefficient_count,
      offset_training_logloss =
        log_loss_matrix_local(
          training_features$truth,
          training_offset
        ),
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
  result <- list(
    validation_no =
      validation_features$no,
    prediction = validate_probability(
      prediction,
      nrow(validation_features$truth)
    ),
    offset_prediction =
      validate_probability(
        validation_offset,
        nrow(validation_features$truth)
      ),
    offset_signature =
      numeric_signature(validation_offset),
    seeds = as.integer(seeds),
    fits = do.call(rbind, fit_rows),
    profile_columns =
      training_features$profile_columns,
    respondent_columns =
      training_features$respondent_columns
  )
  if (!is.null(checkpoint)) {
    saveRDS(
      list(
        experiment_id = experiment_id,
        config = config,
        seeds = as.integer(seeds),
        validation_no =
          expected_validation_no,
        fitting_cases =
          expected_fitting_cases,
        result = result
      ),
      checkpoint
    )
  }
  result
}

run_low_rank_oof <- function(
    train,
    fold_map,
    helper,
    checkpoint_tag,
    outer_seed) {
  fold_map <- validate_fold_map(
    train, fold_map
  )
  row_fold <- unname(
    fold_map[as.character(train$Case)]
  )
  stopifnot(!anyNA(row_fold))
  prediction <- matrix(
    NA_real_, nrow(train), 4L
  )
  offset_prediction <- matrix(
    NA_real_, nrow(train), 4L
  )
  fold_results <- vector("list", 5L)
  for (outer_fold in 1:5) {
    cat(sprintf(
      "\n=== %s: low-rank outer fold %d/5 ===\n",
      checkpoint_tag,
      outer_fold
    ))
    validation_rows <- row_fold == outer_fold
    result <- fit_outer_low_rank(
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
      helper,
      outer_seed,
      outer_fold,
      config = low_rank_config,
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
    prediction[rows, ] <- result$prediction
    offset_prediction[rows, ] <-
      result$offset_prediction
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
    offset_prediction =
      validate_probability(
        offset_prediction, nrow(train)
      ),
    fold_results = fold_results,
    fold_map = fold_map,
    row_fold = row_fold
  )
}

load_v14_artifact <- function(
    train,
    truth,
    seed) {
  if (seed == canonical_seed) {
    path <- file.path(
      v14_dir, "canonical_result.rds"
    )
  } else {
    path <- file.path(
      v14_dir,
      sprintf("repeat_result_%d.rds", seed)
    )
  }
  if (!file.exists(path)) {
    stop(
      "Missing verified v14 artifact: ", path
    )
  }
  object <- readRDS(path)
  stopifnot(
    identical(
      object$experiment_id,
      "set_context_utility_network_v1"
    ),
    !is.null(object$fold_map),
    !is.null(object$candidate_prediction)
  )
  fold_map <- validate_fold_map(
    train, object$fold_map
  )
  if (seed != canonical_seed) {
    expected_map <- balanced_fold_map(
      unique(train$Case), 5L, seed
    )
    stopifnot(identical(
      as.integer(
        fold_map[names(expected_map)]
      ),
      as.integer(expected_map)
    ))
  }
  prediction <- validate_probability(
    object$candidate_prediction,
    nrow(train)
  )
  loss <- log_loss_matrix_local(
    truth, prediction
  )
  if (
    seed == canonical_seed &&
    !is.null(object$summary$candidate_logloss)
  ) {
    stopifnot(
      abs(
        loss -
          object$summary$candidate_logloss[[1L]]
      ) < 1e-10
    )
  }
  list(
    seed = seed,
    path = path,
    fold_map = fold_map,
    prediction = prediction,
    logloss = loss
  )
}

load_v14_registry <- function(train, truth) {
  summary_path <- file.path(
    v14_dir, "repeated_cv_summary.csv"
  )
  if (!file.exists(summary_path)) {
    stop(
      "Missing v14 promotion summary: ",
      summary_path
    )
  }
  summary <- read.csv(summary_path)
  stopifnot(
    nrow(summary) == 1L,
    isTRUE(as.logical(summary$promote[[1L]])),
    as.integer(summary$positive_repeats[[1L]]) == 6L,
    abs(
      summary$point_gain[[1L]] -
        expected_v14_repeated_gain
    ) <= v14_metric_tolerance,
    abs(
      summary$lower_95[[1L]] -
        expected_v14_repeated_lower
    ) <= v14_metric_tolerance
  )
  artifacts <- lapply(
    all_outer_seeds,
    function(seed) {
      load_v14_artifact(train, truth, seed)
    }
  )
  names(artifacts) <- as.character(all_outer_seeds)
  list(
    summary = summary,
    artifacts = artifacts
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
    best_weight <-
      blend_weight_grid[[best_index]]
    candidate[validation_rows, ] <-
      (1 - best_weight) *
        baseline[validation_rows, , drop = FALSE] +
        best_weight *
        component[validation_rows, , drop = FALSE]
    weight_rows[[fold]] <- data.frame(
      fold = fold,
      low_rank_weight = best_weight,
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

bootstrap_case_means <- function(
    case_gain,
    replicates,
    seed = canonical_seed,
    chunk_size = 1000L) {
  case_gain <- as.numeric(case_gain)
  stopifnot(
    length(case_gain) > 1L,
    all(is.finite(case_gain)),
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
        length(case_gain),
        length(case_gain) * count,
        replace = TRUE
      ),
      nrow = length(case_gain),
      ncol = count
    )
    output[start:(start + count - 1L)] <-
      colMeans(matrix(
        case_gain[index],
        nrow = length(case_gain),
        ncol = count
      ))
    start <- start + count
  }
  output
}

summarize_case_gain <- function(
    case_gain,
    replicates,
    seed) {
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
    bootstrap = bootstrap
  )
}

bootstrap_summary <- function(
    truth,
    baseline,
    candidate,
    case,
    replicates,
    seed = canonical_seed) {
  case_gain <- respondent_gain(
    truth, baseline, candidate, case
  )
  summarized <- summarize_case_gain(
    case_gain, replicates, seed
  )
  list(
    summary = summarized$summary,
    respondent_gain = case_gain,
    bootstrap = summarized$bootstrap
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
        respondent_features =
          length(result$respondent_columns),
        rank = low_rank_config$rank,
        mean_training_logloss =
          mean(result$fits$final_training_logloss),
        mean_offset_training_logloss =
          mean(result$fits$offset_training_logloss),
        total_elapsed_seconds =
          sum(result$fits$elapsed_seconds),
        low_rank_weight =
          weights$low_rank_weight[
            weights$fold == fold
          ]
      )
    })
  )
}

fit_test_propensity <- function(train, test) {
  train_respondents <- train[
    !duplicated(train$Case),
    c("Case", domain_covariates),
    drop = FALSE
  ]
  test_respondents <- test[
    !duplicated(test$Case),
    c("Case", domain_covariates),
    drop = FALSE
  ]
  train_domain <- train_respondents[
    , domain_covariates, drop = FALSE
  ]
  test_domain <- test_respondents[
    , domain_covariates, drop = FALSE
  ]
  train_domain$is_test <- 0L
  test_domain$is_test <- 1L
  domain <- rbind(train_domain, test_domain)
  classifier <- stats::glm(
    is_test ~ .,
    data = domain,
    family = stats::binomial()
  )
  propensity <- stats::predict(
    classifier,
    newdata = train_respondents[
      , domain_covariates, drop = FALSE
    ],
    type = "response"
  )
  propensity <- pmin(
    pmax(as.numeric(propensity), 1e-8),
    1 - 1e-8
  )
  names(propensity) <- as.character(
    train_respondents$Case
  )
  list(
    classifier = classifier,
    propensity = propensity
  )
}

test_like_diagnostic <- function(
    average_case_gain,
    propensity,
    replicates) {
  stopifnot(
    all(names(average_case_gain) %in%
      names(propensity))
  )
  propensity <- propensity[
    names(average_case_gain)
  ]
  ordering <- order(
    propensity, decreasing = TRUE
  )
  groups <- list(
    all = seq_along(ordering),
    top_50pct = ordering[
      seq_len(ceiling(0.50 * length(ordering)))
    ],
    top_30pct = ordering[
      seq_len(ceiling(0.30 * length(ordering)))
    ]
  )
  rows <- lapply(
    seq_along(groups),
    function(index) {
      group_name <- names(groups)[[index]]
      group_index <- groups[[index]]
      summarized <- summarize_case_gain(
        average_case_gain[group_index],
        replicates,
        seed = 15400L + index
      )$summary
      cbind(
        data.frame(
          group = group_name,
          n_respondents = length(group_index),
          mean_test_propensity =
            mean(propensity[group_index])
        ),
        summarized
      )
    }
  )
  do.call(rbind, rows)
}

run_smoke_test <- function(
    train,
    truth,
    helper) {
  v14 <- load_v14_artifact(
    train, truth, canonical_seed
  )
  outer_fold <- 1L
  row_fold <- unname(
    v14$fold_map[as.character(train$Case)]
  )
  fitting <- train[
    row_fold != outer_fold,
    ,
    drop = FALSE
  ]
  validation <- train[
    row_fold == outer_fold,
    ,
    drop = FALSE
  ]
  smoke_config <- low_rank_config
  smoke_config$epochs <- 2L
  smoke_seed <- network_seeds_for_fold(
    canonical_seed, outer_fold
  )[[1L]]
  cat(
    "SMOKE MODE: canonical outer fold 1, ",
    "one low-rank seed, two epochs; no verdict.\n",
    sep = ""
  )
  result <- fit_outer_low_rank(
    fitting,
    validation,
    helper,
    canonical_seed,
    outer_fold,
    config = smoke_config,
    seeds = smoke_seed,
    checkpoint_tag = "smoke"
  )
  validation_rows <- match(
    result$validation_no,
    train$No
  )
  stopifnot(
    !anyNA(validation_rows),
    all(row_fold[validation_rows] == outer_fold)
  )
  smoke_truth <- truth[
    validation_rows, , drop = FALSE
  ]
  smoke_baseline <- v14$prediction[
    validation_rows, , drop = FALSE
  ]
  result$smoke_component_logloss <-
    log_loss_matrix_local(
      smoke_truth, result$prediction
    )
  result$smoke_v14_logloss <-
    log_loss_matrix_local(
      smoke_truth, smoke_baseline
    )
  path <- file.path(
    output_dir, "smoke_result.rds"
  )
  saveRDS(
    list(
      experiment_id = experiment_id,
      result = result
    ),
    path
  )
  reloaded <- readRDS(path)
  stopifnot(
    identical(
      reloaded$experiment_id,
      experiment_id
    ),
    identical(
      reloaded$result$validation_no,
      result$validation_no
    ),
    same_signature(
      numeric_signature(
        reloaded$result$prediction
      ),
      numeric_signature(result$prediction)
    ),
    same_signature(
      reloaded$result$offset_signature,
      result$offset_signature
    )
  )
  cat(sprintf(
    paste0(
      "\nSmoke test completed successfully: %s\n",
      "Smoke low-rank component log loss: %.6f\n",
      "Smoke v14 baseline log loss: %.6f\n"
    ),
    path,
    result$smoke_component_logloss,
    result$smoke_v14_logloss
  ))
  invisible(result)
}

run_experiment <- function() {
  train <- read.csv(
    file.path("csv files", "train.csv")
  )
  train <- train[
    order(train$No), , drop = FALSE
  ]
  rownames(train) <- NULL
  test <- read.csv(
    file.path("csv files", "test.csv")
  )
  test <- test[
    order(test$No), , drop = FALSE
  ]
  rownames(test) <- NULL
  truth <- as.matrix(
    train[
      , paste0("Ch", 1:4), drop = FALSE
    ]
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
    all(rowSums(truth) == 1L)
  )
  helper <- m8trpg_helper_environment()
  cat("\nRunning", experiment_id, "\n")
  cat(
    "Frozen v14 repeated gain:",
    sprintf("%.10f", expected_v14_repeated_gain),
    "\n"
  )

  if (smoke_mode) {
    return(run_smoke_test(
      train, truth, helper
    ))
  }

  v14_registry <- load_v14_registry(
    train, truth
  )
  canonical_v14 <-
    v14_registry$artifacts[[
      as.character(canonical_seed)
    ]]
  low_rank_oof <- run_low_rank_oof(
    train,
    canonical_v14$fold_map,
    helper,
    checkpoint_tag = "seed_4821",
    outer_seed = canonical_seed
  )
  blended <- crossfit_blend(
    truth,
    canonical_v14$prediction,
    low_rank_oof$prediction,
    low_rank_oof$row_fold
  )
  candidate_loss <- log_loss_matrix_local(
    truth, blended$prediction
  )
  canonical_bootstrap <- bootstrap_summary(
    truth,
    canonical_v14$prediction,
    blended$prediction,
    train$Case,
    bootstrap_replicates,
    seed = canonical_seed
  )
  canonical_summary <- cbind(
    data.frame(
      experiment = experiment_id,
      stage = "canonical",
      baseline_name = "set_context_v14",
      baseline_logloss =
        canonical_v14$logloss,
      m8trpg_offset_logloss =
        log_loss_matrix_local(
          truth,
          low_rank_oof$offset_prediction
        ),
      low_rank_component_logloss =
        log_loss_matrix_local(
          truth, low_rank_oof$prediction
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
    low_rank_oof$fold_results,
    blended$weights,
    canonical_seed
  )

  write.csv(
    canonical_summary,
    file.path(
      output_dir, "canonical_summary.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    canonical_selection,
    file.path(
      output_dir,
      "canonical_fold_selection.csv"
    ),
    row.names = FALSE
  )
  saveRDS(
    list(
      experiment_id = experiment_id,
      summary = canonical_summary,
      baseline_prediction =
        canonical_v14$prediction,
      m8trpg_offset_prediction =
        low_rank_oof$offset_prediction,
      low_rank_prediction =
        low_rank_oof$prediction,
      candidate_prediction =
        blended$prediction,
      fold_map = canonical_v14$fold_map,
      fold_results =
        low_rank_oof$fold_results,
      weights = blended$weights,
      respondent_gain =
        canonical_bootstrap$respondent_gain,
      bootstrap = canonical_bootstrap$bootstrap
    ),
    file.path(
      output_dir, "canonical_result.rds"
    )
  )
  cat("\nCanonical result:\n")
  print(canonical_summary, digits = 9)
  cat("\nFold selections:\n")
  print(canonical_selection, digits = 7)

  run_repeated <- repeated_mode == "always" ||
    (
      repeated_mode == "auto" &&
      (
        isTRUE(
          canonical_summary$canonical_pass
        ) ||
        isTRUE(canonical_summary$near_miss)
      )
    )
  if (!run_repeated) {
    reason <- if (repeated_mode == "never") {
      "Repeated CV disabled by LOW_RANK_REPEATED=never."
    } else {
      paste0(
        "Repeated CV not triggered: canonical result ",
        "was neither a pass nor the pre-registered near miss."
      )
    }
    verdict <- paste(
      "REJECT low-rank factorization.",
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
  case_levels <- sort(unique(train$Case))
  repeat_names <- as.character(
    all_outer_seeds
  )
  case_gain_matrix <- matrix(
    NA_real_,
    nrow = length(case_levels),
    ncol = length(repeat_names),
    dimnames = list(
      as.character(case_levels),
      repeat_names
    )
  )
  case_gain_matrix[, as.character(canonical_seed)] <-
    canonical_bootstrap$respondent_gain
  repeat_rows <- list(
    `4821` = data.frame(
      seed = canonical_seed,
      baseline_logloss =
        canonical_v14$logloss,
      m8trpg_offset_logloss =
        log_loss_matrix_local(
          truth,
          low_rank_oof$offset_prediction
        ),
      low_rank_component_logloss =
        log_loss_matrix_local(
          truth, low_rank_oof$prediction
        ),
      candidate_logloss = candidate_loss,
      gain = canonical_v14$logloss -
        candidate_loss
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
    repeat_v14 <-
      v14_registry$artifacts[[seed_name]]
    repeat_low_rank <- run_low_rank_oof(
      train,
      repeat_v14$fold_map,
      helper,
      checkpoint_tag = paste0("seed_", seed),
      outer_seed = seed
    )
    repeat_blend <- crossfit_blend(
      truth,
      repeat_v14$prediction,
      repeat_low_rank$prediction,
      repeat_low_rank$row_fold
    )
    repeat_candidate_loss <-
      log_loss_matrix_local(
        truth, repeat_blend$prediction
      )
    repeat_case_gain <- respondent_gain(
      truth,
      repeat_v14$prediction,
      repeat_blend$prediction,
      train$Case
    )
    case_gain_matrix[, seed_name] <-
      repeat_case_gain
    repeat_rows[[seed_name]] <- data.frame(
      seed = seed,
      baseline_logloss = repeat_v14$logloss,
      m8trpg_offset_logloss =
        log_loss_matrix_local(
          truth,
          repeat_low_rank$offset_prediction
        ),
      low_rank_component_logloss =
        log_loss_matrix_local(
          truth,
          repeat_low_rank$prediction
        ),
      candidate_logloss =
        repeat_candidate_loss,
      gain = repeat_v14$logloss -
        repeat_candidate_loss
    )
    selection_rows[[seed_name]] <-
      fold_selection_table(
        repeat_low_rank$fold_results,
        repeat_blend$weights,
        seed
      )
    saveRDS(
      list(
        experiment_id = experiment_id,
        seed = seed,
        baseline_source = repeat_v14$path,
        fold_map = repeat_v14$fold_map,
        baseline_prediction =
          repeat_v14$prediction,
        m8trpg_offset_prediction =
          repeat_low_rank$offset_prediction,
        low_rank_prediction =
          repeat_low_rank$prediction,
        candidate_prediction =
          repeat_blend$prediction,
        weights = repeat_blend$weights,
        fold_results =
          repeat_low_rank$fold_results,
        respondent_gain = repeat_case_gain
      ),
      file.path(
        output_dir,
        sprintf("repeat_result_%d.rds", seed)
      )
    )
  }

  repeated_by_seed <- do.call(
    rbind, repeat_rows
  )
  average_case_gain <- rowMeans(
    case_gain_matrix
  )
  pooled_summary <- summarize_case_gain(
    average_case_gain,
    bootstrap_replicates,
    canonical_seed
  )
  repeated_summary <- cbind(
    data.frame(
      experiment = experiment_id,
      stage = "repeated_cv"
    ),
    pooled_summary$summary,
    data.frame(
      positive_repeats =
        sum(repeated_by_seed$gain > 0),
      n_repeats = nrow(repeated_by_seed)
    )
  )

  propensity <- fit_test_propensity(
    train, test
  )
  test_like <- test_like_diagnostic(
    average_case_gain,
    propensity$propensity,
    bootstrap_replicates
  )
  top_30 <- test_like[
    test_like$group == "top_30pct",
    ,
    drop = FALSE
  ]
  repeated_summary$test_like_30_point_gain <-
    top_30$point_gain[[1L]]
  repeated_summary$test_like_30_lower_95 <-
    top_30$lower_95[[1L]]
  repeated_summary$promote <-
    repeated_summary$point_gain > 0 &
    repeated_summary$lower_95 > 0 &
    repeated_summary$positive_repeats >= 5L &
    repeated_summary$test_like_30_point_gain >= 0 &
    repeated_summary$test_like_30_lower_95 >=
      -0.001

  write.csv(
    repeated_by_seed,
    file.path(
      output_dir, "repeated_cv_by_seed.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    repeated_summary,
    file.path(
      output_dir, "repeated_cv_summary.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    do.call(rbind, selection_rows),
    file.path(
      output_dir,
      "repeated_cv_fold_selection.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    test_like,
    file.path(
      output_dir,
      "test_like_respondent_diagnostic.csv"
    ),
    row.names = FALSE
  )
  saveRDS(
    list(
      experiment_id = experiment_id,
      repeated_by_seed = repeated_by_seed,
      repeated_summary = repeated_summary,
      case_gain_matrix = case_gain_matrix,
      average_case_gain = average_case_gain,
      pooled_bootstrap =
        pooled_summary$bootstrap,
      test_propensity =
        propensity$propensity,
      test_like_diagnostic = test_like
    ),
    file.path(
      output_dir, "repeated_cv_result.rds"
    )
  )

  if (isTRUE(repeated_summary$promote)) {
    verdict <- paste0(
      "PROMOTE low-rank factorization for a separate ",
      "full-data build audit. Do not submit until that ",
      "artifact is reproduced and checked."
    )
  } else {
    verdict <- paste0(
      "REJECT low-rank factorization; retain the exact ",
      "set-context v14 submission."
    )
  }
  writeLines(
    verdict,
    file.path(output_dir, "verdict.txt")
  )
  cat("\nRepeated-CV result:\n")
  print(repeated_summary, digits = 9)
  cat("\nTest-like respondent diagnostic:\n")
  print(test_like, digits = 9)
  cat("\n", verdict, "\n", sep = "")
  invisible(list(
    canonical = canonical_summary,
    repeated = repeated_summary,
    test_like = test_like,
    verdict = verdict
  ))
}

run_experiment()
