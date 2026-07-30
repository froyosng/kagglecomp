# Comprehensive optimisation-rescue suite for the respondent-sequence
# Transformer. This runner cannot use test data, generate a submission, or run
# repeated CV.
#
# Pre-registration:
#   codex_transformer_rescue_preregister.md
#
# Smoke test:
#   Sys.setenv(TRANSFORMER_RESCUE_SMOKE = "1")
#   source("R/codex_transformer_rescue_suite.R")
#   Sys.unsetenv("TRANSFORMER_RESCUE_SMOKE")
#
# Full unattended run:
#   Sys.unsetenv("TRANSFORMER_RESCUE_SMOKE")
#   source("R/codex_transformer_rescue_suite.R")

options(stringsAsFactors = FALSE)

rescue_id <- "respondent_sequence_transformer_rescue_v1"
transformer_runner_path <- file.path(
  "R", "codex_respondent_sequence_transformer.R"
)
expected_transformer_runner_md5 <-
  "52cd06dd26d5079e01bcb884990a90e3"
rescue_output_dir <- file.path(
  "data_processed",
  "codex_respondent_sequence_transformer_rescue"
)
dir.create(
  rescue_output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

smoke_mode <- identical(
  Sys.getenv("TRANSFORMER_RESCUE_SMOKE", "0"),
  "1"
)
bootstrap_replicates <- if (smoke_mode) {
  200L
} else {
  as.integer(Sys.getenv(
    "TRANSFORMER_RESCUE_N_BOOT",
    "100000"
  ))
}
if (
  !is.finite(bootstrap_replicates) ||
  bootstrap_replicates < 100L
) {
  stop("TRANSFORMER_RESCUE_N_BOOT must be an integer >= 100.")
}

variant_registry <- data.frame(
  variant_order = 1:6,
  variant = c(
    "zero_no_penalty",
    "zero_no_weight_decay",
    "zero_no_regularizers",
    "small_init_no_regularizers",
    "small_init_low_regularization",
    "small_init_low_reg_fast_lr"
  ),
  output_init_sd = c(
    0, 0, 0, 0.001, 0.001, 0.001
  ),
  correction_penalty = c(
    0, 0.01, 0, 0, 0.001, 0.001
  ),
  weight_decay = c(
    0.002, 0, 0, 0, 0.0002, 0.0002
  ),
  learning_rate = c(
    0.0007, 0.0007, 0.0007,
    0.0007, 0.0007, 0.0015
  ),
  stringsAsFactors = FALSE
)
stopifnot(
  !anyDuplicated(variant_registry$variant),
  identical(
    variant_registry$variant_order,
    seq_len(nrow(variant_registry))
  )
)

if (!file.exists(transformer_runner_path)) {
  stop(
    "Missing ", transformer_runner_path,
    ". Put this suite beside the validated Transformer runner."
  )
}
observed_transformer_runner_md5 <- unname(
  tools::md5sum(transformer_runner_path)
)
if (!identical(
  observed_transformer_runner_md5,
  expected_transformer_runner_md5
)) {
  stop(
    "The Transformer runner does not match the validated telemetry version.",
    "\nExpected MD5: ", expected_transformer_runner_md5,
    "\nObserved MD5: ", observed_transformer_runner_md5
  )
}

load_transformer_definitions <- function() {
  expressions <- parse(
    file = transformer_runner_path,
    keep.source = FALSE
  )
  stopifnot(length(expressions) > 1L)
  final_expression <- paste(
    deparse(expressions[[length(expressions)]]),
    collapse = ""
  )
  if (!identical(final_expression, "run_experiment()")) {
    stop(
      "The validated Transformer runner no longer ends with ",
      "the expected run_experiment() call."
    )
  }
  environment <- new.env(parent = globalenv())
  for (index in seq_len(length(expressions) - 1L)) {
    eval(expressions[[index]], envir = environment)
  }
  required <- c(
    "audited",
    "sequence_config",
    "canonical_seed",
    "network_seeds_for_fold",
    "respondent_sequence_transformer",
    "feature_tensors",
    "predict_sequence_transformer",
    "build_sequence_features",
    "fit_gap_scaler",
    "align_offset",
    "subset_training_respondents",
    "log_loss_matrix_local",
    "validate_probability",
    "numeric_signature",
    "format_duration"
  )
  stopifnot(vapply(
    required,
    exists,
    logical(1),
    envir = environment,
    inherits = FALSE
  ))
  environment
}

safe_save_rds <- function(object, path) {
  temporary <- tempfile(
    pattern = paste0(basename(path), "_"),
    tmpdir = dirname(path)
  )
  on.exit({
    if (file.exists(temporary)) {
      unlink(temporary)
    }
  }, add = TRUE)
  saveRDS(object, temporary)
  if (!file.rename(temporary, path)) {
    if (!file.copy(temporary, path, overwrite = TRUE)) {
      stop("Could not save checkpoint: ", path)
    }
    unlink(temporary)
  }
  invisible(path)
}

append_csv_row <- function(row, path) {
  write.table(
    row,
    file = path,
    sep = ",",
    row.names = FALSE,
    col.names = !file.exists(path),
    append = file.exists(path),
    quote = TRUE,
    na = ""
  )
}

runtime <- new.env(parent = emptyenv())
runtime$run_id <- paste0(
  format(Sys.time(), "%Y%m%dT%H%M%S"),
  "_pid", Sys.getpid()
)
runtime$started_at <- Sys.time()
runtime$status <- "running"
runtime$completed_production_fits <- 0L
runtime$expected_production_fits <- if (smoke_mode) {
  0L
} else {
  6L
}
runtime$production_seconds <- numeric()
runtime_progress_path <- file.path(
  rescue_output_dir, "runtime_progress.csv"
)
runtime_runs_path <- file.path(
  rescue_output_dir, "runtime_runs.csv"
)

record_runtime <- function(
    stage,
    variant,
    outer_fold,
    network_seed,
    source,
    elapsed_seconds,
    production_fit = FALSE) {
  if (isTRUE(production_fit)) {
    runtime$completed_production_fits <-
      runtime$completed_production_fits + 1L
    runtime$production_seconds <- c(
      runtime$production_seconds,
      as.numeric(elapsed_seconds)
    )
  }
  mean_seconds <- if (
    length(runtime$production_seconds) > 0L
  ) {
    mean(runtime$production_seconds)
  } else {
    NA_real_
  }
  remaining <- max(
    0L,
    runtime$expected_production_fits -
      runtime$completed_production_fits
  )
  eta_seconds <- if (is.finite(mean_seconds)) {
    remaining * mean_seconds
  } else {
    NA_real_
  }
  now <- Sys.time()
  row <- data.frame(
    run_id = runtime$run_id,
    event_at = format(
      now, "%Y-%m-%d %H:%M:%S %Z"
    ),
    stage = as.character(stage),
    variant = as.character(variant),
    outer_fold = as.integer(outer_fold),
    network_seed = as.integer(network_seed),
    source = as.character(source),
    fit_elapsed_seconds =
      as.numeric(elapsed_seconds),
    run_elapsed_seconds = as.numeric(
      difftime(
        now, runtime$started_at,
        units = "secs"
      )
    ),
    completed_production_fits =
      runtime$completed_production_fits,
    expected_production_fits =
      runtime$expected_production_fits,
    mean_production_fit_seconds =
      mean_seconds,
    eta_seconds = eta_seconds,
    estimated_finish_at = if (
      is.finite(eta_seconds)
    ) {
      format(
        now + eta_seconds,
        "%Y-%m-%d %H:%M:%S %Z"
      )
    } else {
      NA_character_
    },
    stringsAsFactors = FALSE
  )
  append_csv_row(row, runtime_progress_path)
  if (isTRUE(production_fit)) {
    cat(sprintf(
      paste0(
        "Progress: %d/%d production fits; ",
        "mean %s; ETA %s.\n"
      ),
      runtime$completed_production_fits,
      runtime$expected_production_fits,
      if (is.finite(mean_seconds)) {
        runtime$transformer$
          format_duration(mean_seconds)
      } else {
        "unknown"
      },
      if (is.finite(eta_seconds)) {
        runtime$transformer$
          format_duration(eta_seconds)
      } else {
        "unknown"
      }
    ))
    flush.console()
  }
  invisible(row)
}

finalize_runtime <- function() {
  row <- data.frame(
    run_id = runtime$run_id,
    started_at = format(
      runtime$started_at,
      "%Y-%m-%d %H:%M:%S %Z"
    ),
    ended_at = format(
      Sys.time(),
      "%Y-%m-%d %H:%M:%S %Z"
    ),
    status = runtime$status,
    smoke_mode = smoke_mode,
    elapsed_seconds = as.numeric(
      difftime(
        Sys.time(), runtime$started_at,
        units = "secs"
      )
    ),
    completed_production_fits =
      runtime$completed_production_fits,
    expected_production_fits =
      runtime$expected_production_fits,
    stringsAsFactors = FALSE
  )
  try(
    append_csv_row(row, runtime_runs_path),
    silent = TRUE
  )
  invisible(row)
}

variant_config <- function(
    transformer,
    variant,
    stage = c("production", "mechanism")) {
  stage <- match.arg(stage)
  row <- variant_registry[
    variant_registry$variant == variant,
    ,
    drop = FALSE
  ]
  stopifnot(nrow(row) == 1L)
  config <- transformer$sequence_config
  config$output_init_sd <-
    row$output_init_sd[[1L]]
  config$correction_penalty <-
    row$correction_penalty[[1L]]
  config$weight_decay <-
    row$weight_decay[[1L]]
  config$learning_rate <-
    row$learning_rate[[1L]]
  if (identical(stage, "mechanism")) {
    config$profile_dropout <- 0
    config$transformer_dropout <- 0
    config$utility_dropout <- 0
    config$epochs <- if (smoke_mode) {
      2L
    } else {
      60L
    }
    config$batch_respondents <- 32L
    config$gradient_clip <- 5.0
  } else if (smoke_mode) {
    config$epochs <- 2L
  }
  config
}

original_config <- function(transformer) {
  config <- transformer$sequence_config
  config$output_init_sd <- 0
  config
}

tensor_norm <- function(tensor) {
  if (
    is.null(tensor) ||
    torch::is_undefined_tensor(tensor)
  ) {
    return(NA_real_)
  }
  as.numeric(tensor$detach()$norm()$item())
}

tensor_max_abs <- function(tensor) {
  as.numeric(
    tensor$detach()$abs()$max()$item()
  )
}

scalar_value <- function(value) {
  if (is.numeric(value)) {
    return(as.numeric(value[[1L]]))
  }
  as.numeric(value$item())
}

initialize_rescue_model <- function(
    transformer,
    training_features,
    config,
    seed) {
  set.seed(as.integer(seed))
  torch::torch_manual_seed(as.integer(seed))
  model <- transformer$
    respondent_sequence_transformer(
      profile_dim =
        ncol(training_features$profile),
      common_dim =
        ncol(training_features$common),
      config = config
    )
  if (
    is.finite(config$output_init_sd) &&
    config$output_init_sd > 0
  ) {
    torch::nn_init_normal_(
      model$utility_output$weight,
      mean = 0,
      std = config$output_init_sd
    )
    torch::nn_init_zeros_(
      model$utility_output$bias
    )
  }
  model
}

one_step_check <- function(
    transformer,
    features,
    offset_prediction,
    seed) {
  subset <- transformer$subset_training_respondents(
    features,
    offset_prediction,
    if (smoke_mode) 8L else 32L
  )
  config <- original_config(transformer)
  tensors <- transformer$feature_tensors(
    subset$features, subset$offset
  )
  index <- seq_len(
    subset$features$n_respondents
  )
  model <- initialize_rescue_model(
    transformer,
    subset$features,
    config,
    seed
  )
  optimizer <- torch::optim_adam(
    model$parameters,
    lr = config$learning_rate,
    weight_decay = config$weight_decay
  )
  weight_before <- model$utility_output$
    weight$detach()$clone()
  bias_before <- model$utility_output$
    bias$detach()$clone()

  model$eval()
  before <- torch::with_no_grad({
    output <- model(
      tensors$profile[index, , , ],
      tensors$common[index, , ],
      tensors$offset[index, , ]
    )
    list(
      loss = torch::nnf_cross_entropy(
        output$utility$reshape(c(-1L, 4L)),
        tensors$target[index, ]$reshape(c(-1L))
      )$item(),
      correction = tensor_max_abs(
        output$correction
      )
    )
  })

  model$train()
  optimizer$zero_grad()
  output <- model(
    tensors$profile[index, , , ],
    tensors$common[index, , ],
    tensors$offset[index, , ]
  )
  cross_entropy <- torch::nnf_cross_entropy(
    output$utility$reshape(c(-1L, 4L)),
    tensors$target[index, ]$reshape(c(-1L))
  )
  correction_square <- output$correction$
    pow(2)$mean()
  loss <- cross_entropy +
    config$correction_penalty *
      correction_square
  loss$backward()
  weight_gradient <- tensor_norm(
    model$utility_output$weight$grad
  )
  bias_gradient <- tensor_norm(
    model$utility_output$bias$grad
  )
  global_gradient <- scalar_value(
    torch::nn_utils_clip_grad_norm_(
      model$parameters,
      max_norm = config$gradient_clip
    )
  )
  optimizer$step()
  weight_delta <- tensor_norm(
    model$utility_output$weight$detach() -
      weight_before
  )
  bias_delta <- tensor_norm(
    model$utility_output$bias$detach() -
      bias_before
  )

  model$eval()
  after <- torch::with_no_grad({
    output <- model(
      tensors$profile[index, , , ],
      tensors$common[index, , ],
      tensors$offset[index, , ]
    )
    list(
      loss = torch::nnf_cross_entropy(
        output$utility$reshape(c(-1L, 4L)),
        tensors$target[index, ]$reshape(c(-1L))
      )$item(),
      correction = tensor_max_abs(
        output$correction
      )
    )
  })
  pass <- all(is.finite(c(
    weight_gradient,
    weight_delta,
    after$correction
  ))) &&
    weight_gradient > 1e-10 &&
    weight_delta > 1e-12 &&
    after$correction > 1e-8
  data.frame(
    rescue_id = rescue_id,
    respondents =
      subset$features$n_respondents,
    seed = as.integer(seed),
    loss_before = as.numeric(before$loss),
    loss_after = as.numeric(after$loss),
    loss_change =
      as.numeric(before$loss - after$loss),
    max_correction_before =
      before$correction,
    max_correction_after =
      after$correction,
    output_weight_gradient =
      weight_gradient,
    output_bias_gradient = bias_gradient,
    global_gradient_norm =
      global_gradient,
    output_weight_delta = weight_delta,
    output_bias_delta = bias_delta,
    pass = pass,
    stringsAsFactors = FALSE
  )
}

fit_rescue_once <- function(
    transformer,
    training_features,
    validation_features,
    training_offset,
    validation_offset,
    config,
    variant,
    seed,
    outer_fold,
    checkpoint,
    fitting_cases) {
  validation_no <- as.integer(
    validation_features$no
  )
  if (file.exists(checkpoint)) {
    saved <- readRDS(checkpoint)
    if (
      identical(saved$rescue_id, rescue_id) &&
      identical(
        saved$transformer_runner_md5,
        observed_transformer_runner_md5
      ) &&
      identical(saved$variant, variant) &&
      identical(saved$config, config) &&
      identical(saved$seed, as.integer(seed)) &&
      identical(
        saved$outer_fold,
        as.integer(outer_fold)
      ) &&
      identical(
        saved$validation_no,
        validation_no
      ) &&
      identical(
        saved$fitting_cases,
        fitting_cases
      )
    ) {
      result <- saved$result
      result$prediction <-
        transformer$validate_probability(
          result$prediction,
          length(validation_no)
        )
      stopifnot(
        all(is.finite(
          result$correction_signature
        )),
        is.finite(
          result$maximum_absolute_correction
        ),
        result$maximum_absolute_correction <=
          config$correction_bound + 1e-6
      )
      result$checkpoint_reused <- TRUE
      cat(
        "  using verified rescue checkpoint: ",
        basename(checkpoint), "\n",
        sep = ""
      )
      return(result)
    }
  }

  tensors <- transformer$feature_tensors(
    training_features,
    training_offset
  )
  n_respondents <-
    training_features$n_respondents
  model <- initialize_rescue_model(
    transformer,
    training_features,
    config,
    seed
  )
  optimizer <- torch::optim_adam(
    model$parameters,
    lr = config$learning_rate,
    weight_decay = config$weight_decay
  )

  trace_rows <- list()
  started <- proc.time()[["elapsed"]]
  for (epoch in seq_len(config$epochs)) {
    model$train()
    respondent_order <- sample.int(
      n_respondents
    )
    total_loss <- 0
    total_cross_entropy <- 0
    total_correction_square <- 0
    total_tasks <- 0L
    for (
      start in seq.int(
        1L,
        n_respondents,
        by = config$batch_respondents
      )
    ) {
      stop_at <- min(
        n_respondents,
        start + config$batch_respondents - 1L
      )
      respondent_index <-
        respondent_order[start:stop_at]
      optimizer$zero_grad()
      output <- model(
        tensors$profile[
          respondent_index, , , 
        ],
        tensors$common[
          respondent_index, , 
        ],
        tensors$offset[
          respondent_index, , 
        ]
      )
      cross_entropy <- torch::nnf_cross_entropy(
        output$utility$reshape(c(-1L, 4L)),
        tensors$target[
          respondent_index, 
        ]$reshape(c(-1L))
      )
      correction_square <-
        output$correction$pow(2)$mean()
      loss <- cross_entropy +
        config$correction_penalty *
          correction_square
      loss$backward()
      torch::nn_utils_clip_grad_norm_(
        model$parameters,
        max_norm = config$gradient_clip
      )
      optimizer$step()

      task_count <-
        length(respondent_index) * 19L
      total_loss <- total_loss +
        loss$item() * task_count
      total_cross_entropy <-
        total_cross_entropy +
        cross_entropy$item() * task_count
      total_correction_square <-
        total_correction_square +
        correction_square$item() *
          task_count
      total_tasks <- total_tasks + task_count
    }
    if (
      epoch == 1L ||
      epoch %% 10L == 0L ||
      epoch == config$epochs
    ) {
      epoch_elapsed <-
        proc.time()[["elapsed"]] - started
      row <- data.frame(
        epoch = as.integer(epoch),
        training_objective =
          total_loss / total_tasks,
        training_logloss =
          total_cross_entropy /
            total_tasks,
        mean_correction_square =
          total_correction_square /
            total_tasks,
        elapsed_seconds = epoch_elapsed,
        stringsAsFactors = FALSE
      )
      trace_rows[[length(trace_rows) + 1L]] <-
        row
      if (
        epoch %% 10L == 0L ||
        epoch == config$epochs
      ) {
        eta <- (
          epoch_elapsed / epoch
        ) * (config$epochs - epoch)
        cat(sprintf(
          paste0(
            "    %s fold %d epoch %d/%d: ",
            "train %.7f, rms correction %.7g, ",
            "elapsed %s, ETA %s\n"
          ),
          variant,
          outer_fold,
          epoch,
          config$epochs,
          total_cross_entropy /
            total_tasks,
          sqrt(
            total_correction_square /
              total_tasks
          ),
          transformer$format_duration(
            epoch_elapsed
          ),
          transformer$format_duration(eta)
        ))
        flush.console()
      }
    }
  }
  elapsed <- proc.time()[["elapsed"]] - started

  training_prediction <-
    transformer$predict_sequence_transformer(
      model,
      training_features,
      training_offset
    )
  validation_prediction <-
    transformer$predict_sequence_transformer(
      model,
      validation_features,
      validation_offset
    )
  training_logloss <-
    transformer$log_loss_matrix_local(
      training_features$truth,
      training_prediction$prediction
    )
  training_offset_logloss <-
    transformer$log_loss_matrix_local(
      training_features$truth,
      training_offset
    )
  validation_logloss <-
    transformer$log_loss_matrix_local(
      validation_features$truth,
      validation_prediction$prediction
    )
  correction <- validation_prediction$correction
  result <- list(
    prediction =
      validation_prediction$prediction,
    correction_signature =
      transformer$numeric_signature(
        correction
      ),
    maximum_absolute_correction =
      max(abs(correction)),
    rms_validation_correction =
      sqrt(mean(correction^2)),
    saturation_share = mean(
      abs(correction) >=
        0.95 * config$correction_bound
    ),
    seed = as.integer(seed),
    elapsed_seconds = elapsed,
    final_training_logloss =
      training_logloss,
    training_offset_logloss =
      training_offset_logloss,
    training_logloss_change =
      training_offset_logloss -
        training_logloss,
    training_rms_correction =
      sqrt(mean(
        training_prediction$correction^2
      )),
    validation_logloss =
      validation_logloss,
    trace = do.call(rbind, trace_rows),
    checkpoint_reused = FALSE
  )
  safe_save_rds(
    list(
      rescue_id = rescue_id,
      transformer_runner_md5 =
        observed_transformer_runner_md5,
      variant = variant,
      config = config,
      seed = as.integer(seed),
      outer_fold = as.integer(outer_fold),
      validation_no = validation_no,
      fitting_cases = fitting_cases,
      result = result
    ),
    checkpoint
  )
  result
}

prepare_canonical_fold <- function(
    transformer,
    train,
    v14,
    outer_fold) {
  row_fold <- unname(
    v14$fold_map[as.character(train$Case)]
  )
  stopifnot(
    !anyNA(row_fold),
    all(row_fold %in% 1:5),
    outer_fold %in% 1:5
  )
  validation_rows <- row_fold == outer_fold
  fitting <- train[
    !validation_rows, , drop = FALSE
  ]
  validation <- train[
    validation_rows, , drop = FALSE
  ]
  fitting_cases <- sort(unique(
    as.integer(fitting$Case)
  ))
  validation_cases <- sort(unique(
    as.integer(validation$Case)
  ))
  stopifnot(
    length(intersect(
      fitting_cases, validation_cases
    )) == 0L,
    all(table(fitting$Case) == 19L),
    all(table(validation$Case) == 19L)
  )

  helper <- transformer$audited$
    m8trpg_helper_environment()
  offset <- transformer$audited$
    fit_m8trpg_offset(
      fitting, validation, helper
    )
  levels_by_attribute <- transformer$audited$
    active_attribute_levels(
      offset$fitting_long
    )
  respondent_scaler <- transformer$audited$
    fit_respondent_scaler(fitting)
  gap_scaler <- transformer$fit_gap_scaler(
    offset$fitting_long
  )
  training_features <-
    transformer$build_sequence_features(
      offset$fitting_long,
      levels_by_attribute,
      respondent_scaler,
      gap_scaler
    )
  validation_features <-
    transformer$build_sequence_features(
      offset$validation_long,
      levels_by_attribute,
      respondent_scaler,
      gap_scaler
    )
  training_offset <- transformer$align_offset(
    training_features$no,
    offset$fitting_no,
    offset$fitting_prediction
  )
  validation_offset <- transformer$align_offset(
    validation_features$no,
    offset$validation_no,
    offset$validation_prediction
  )
  validation_index <- match(
    validation_features$no,
    train$No
  )
  stopifnot(
    !anyNA(validation_index),
    all(row_fold[validation_index] ==
      outer_fold),
    identical(
      sort(
        training_features$respondent_cases
      ),
      fitting_cases
    ),
    identical(
      sort(
        validation_features$respondent_cases
      ),
      validation_cases
    ),
    identical(
      training_features$profile_columns,
      validation_features$profile_columns
    ),
    identical(
      training_features$common_columns,
      validation_features$common_columns
    )
  )
  list(
    outer_fold = as.integer(outer_fold),
    fitting_cases = fitting_cases,
    validation_cases = validation_cases,
    validation_index = validation_index,
    training_features = training_features,
    validation_features = validation_features,
    training_offset = training_offset,
    validation_offset = validation_offset,
    v14_prediction =
      transformer$validate_probability(
        v14$prediction[
          validation_index, ,
          drop = FALSE
        ],
        length(validation_index)
      )
  )
}

blend_on_grid <- function(
    transformer,
    truth,
    baseline,
    component) {
  weights <- seq(0, 0.30, by = 0.01)
  losses <- vapply(
    weights,
    function(weight) {
      prediction <- (1 - weight) *
        baseline + weight * component
      transformer$log_loss_matrix_local(
        truth, prediction
      )
    },
    numeric(1)
  )
  best <- which.min(losses)
  list(
    weight = weights[[best]],
    loss = losses[[best]],
    grid = data.frame(
      weight = weights,
      logloss = losses,
      stringsAsFactors = FALSE
    )
  )
}

task_gain <- function(
    truth,
    baseline,
    candidate) {
  selected <- max.col(truth)
  row <- seq_len(nrow(truth))
  -log(pmax(
    baseline[cbind(row, selected)],
    1e-15
  )) +
    log(pmax(
      candidate[cbind(row, selected)],
      1e-15
    ))
}

bootstrap_confirmation <- function(
    case,
    gain,
    n_boot,
    seed) {
  case_gain <- tapply(gain, case, mean)
  case_gain <- as.numeric(case_gain)
  stopifnot(
    length(case_gain) == length(unique(case)),
    all(is.finite(case_gain))
  )
  set.seed(as.integer(seed))
  bootstrap <- numeric(n_boot)
  chunk_size <- 1000L
  start <- 1L
  while (start <= n_boot) {
    stop_at <- min(
      n_boot, start + chunk_size - 1L
    )
    count <- stop_at - start + 1L
    sampled <- matrix(
      sample.int(
        length(case_gain),
        length(case_gain) * count,
        replace = TRUE
      ),
      nrow = length(case_gain),
      ncol = count
    )
    bootstrap[start:stop_at] <-
      colMeans(matrix(
        case_gain[sampled],
        nrow = length(case_gain),
        ncol = count
      ))
    start <- stop_at + 1L
  }
  family_tail <- 0.05 / (2 * 6)
  list(
    respondent_gain = case_gain,
    bootstrap = bootstrap,
    summary = data.frame(
      point_gain = mean(gain),
      respondent_mean_gain =
        mean(case_gain),
      bootstrap_mean = mean(bootstrap),
      bootstrap_sd = sd(bootstrap),
      lower_95 = unname(
        quantile(bootstrap, 0.025)
      ),
      upper_95 = unname(
        quantile(bootstrap, 0.975)
      ),
      bonferroni6_lower = unname(
        quantile(
          bootstrap, family_tail
        )
      ),
      bonferroni6_upper = unname(
        quantile(
          bootstrap,
          1 - family_tail
        )
      ),
      win_rate = mean(bootstrap > 0),
      respondents = length(case_gain),
      n_boot = n_boot,
      stringsAsFactors = FALSE
    )
  )
}

write_verdict <- function(text) {
  writeLines(
    text,
    file.path(rescue_output_dir, "verdict.txt")
  )
  cat("\n", text, "\n", sep = "")
  invisible(text)
}

run_rescue_suite <- function() {
  on.exit({
    if (identical(runtime$status, "running")) {
      runtime$status <- "error_or_interrupted"
    }
    finalize_runtime()
  }, add = TRUE)
  transformer <- load_transformer_definitions()
  runtime$transformer <- transformer
  train <- read.csv(
    file.path("csv files", "train.csv")
  )
  train <- train[
    order(train$No), , drop = FALSE
  ]
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
  v14 <- transformer$audited$load_v14_artifact(
    train,
    truth,
    transformer$canonical_seed
  )
  stopifnot(
    nrow(v14$prediction) == nrow(train),
    length(v14$fold_map) ==
      length(unique(train$Case))
  )

  cat(
    "\nRunning ", rescue_id, "\n",
    "Mode: ", if (smoke_mode) "smoke" else "full",
    "\nExact v14 is frozen; test data and submission generation are unavailable.\n",
    sep = ""
  )
  cat(
    "Preparing canonical fold 1 for diagnosis and screening.\n"
  )
  fold_one <- prepare_canonical_fold(
    transformer, train, v14, 1L
  )
  fold_one_seed <-
    transformer$network_seeds_for_fold(
      transformer$canonical_seed, 1L
    )[[1L]]

  cat("\nStage 1: one-step wiring and optimizer check.\n")
  one_step <- one_step_check(
    transformer,
    fold_one$training_features,
    fold_one$training_offset,
    fold_one_seed
  )
  write.csv(
    one_step,
    file.path(
      rescue_output_dir, "one_step_check.csv"
    ),
    row.names = FALSE
  )
  print(one_step, digits = 9)
  if (!isTRUE(one_step$pass)) {
    runtime$status <- "failed_one_step"
    verdict <- paste(
      "OPTIMISATION/IMPLEMENTATION FAILURE:",
      "the original configuration failed the one-step gradient,",
      "parameter-delta, or correction-movement gate.",
      "No rescue candidates were run."
    )
    write_verdict(verdict)
    return(invisible(list(
      one_step = one_step,
      verdict = verdict
    )))
  }

  cat("\nStage 2a: tiny-sample capacity control.\n")
  tiny <- transformer$subset_training_respondents(
    fold_one$training_features,
    fold_one$training_offset,
    if (smoke_mode) 8L else 32L
  )
  capacity_variant <-
    "small_init_no_regularizers"
  capacity_config <- variant_config(
    transformer,
    capacity_variant,
    stage = "mechanism"
  )
  capacity_config$epochs <- if (smoke_mode) {
    4L
  } else {
    200L
  }
  capacity_checkpoint <- file.path(
    rescue_output_dir,
    sprintf(
      "capacity_%s_seed_%d.rds",
      capacity_variant,
      fold_one_seed + 900000L
    )
  )
  capacity <- fit_rescue_once(
    transformer,
    tiny$features,
    tiny$features,
    tiny$offset,
    tiny$offset,
    capacity_config,
    capacity_variant,
    fold_one_seed + 900000L,
    outer_fold = 1L,
    checkpoint = capacity_checkpoint,
    fitting_cases =
      tiny$features$respondent_cases
  )
  capacity_initial_checkpoint_reused <-
    capacity$checkpoint_reused
  capacity_reloaded <- fit_rescue_once(
    transformer,
    tiny$features,
    tiny$features,
    tiny$offset,
    tiny$offset,
    capacity_config,
    capacity_variant,
    fold_one_seed + 900000L,
    outer_fold = 1L,
    checkpoint = capacity_checkpoint,
    fitting_cases =
      tiny$features$respondent_cases
  )
  stopifnot(
    isTRUE(
      capacity_reloaded$checkpoint_reused
    ),
    identical(
      capacity$correction_signature,
      capacity_reloaded$
        correction_signature
    ),
    max(abs(
      capacity$prediction -
        capacity_reloaded$prediction
    )) <= 1e-6
  )
  capacity <- capacity_reloaded
  capacity_loss_drop <-
    capacity$training_logloss_change
  capacity_pass <- if (smoke_mode) {
    all(is.finite(c(
      capacity_loss_drop,
      capacity$maximum_absolute_correction
    ))) &&
      capacity$maximum_absolute_correction >
        1e-8
  } else {
    is.finite(capacity_loss_drop) &&
      capacity_loss_drop >= 0.02 &&
      is.finite(
        capacity$maximum_absolute_correction
      ) &&
      capacity$maximum_absolute_correction >=
        0.05
  }
  capacity_summary <- data.frame(
    rescue_id = rescue_id,
    variant = capacity_variant,
    respondents =
      tiny$features$n_respondents,
    epochs = capacity_config$epochs,
    offset_logloss =
      capacity$training_offset_logloss,
    trained_logloss =
      capacity$final_training_logloss,
    loss_drop = capacity_loss_drop,
    training_rms_correction =
      capacity$training_rms_correction,
    maximum_absolute_correction =
      capacity$maximum_absolute_correction,
    elapsed_seconds =
      capacity$elapsed_seconds,
    initial_checkpoint_reused =
      capacity_initial_checkpoint_reused,
    reload_verified =
      capacity$checkpoint_reused,
    pass = capacity_pass,
    stringsAsFactors = FALSE
  )
  write.csv(
    capacity_summary,
    file.path(
      rescue_output_dir,
      "tiny_capacity_summary.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    capacity$trace,
    file.path(
      rescue_output_dir,
      "tiny_capacity_trace.csv"
    ),
    row.names = FALSE
  )
  print(capacity_summary, digits = 9)
  record_runtime(
    "capacity",
    capacity_variant,
    1L,
    fold_one_seed + 900000L,
    if (capacity_initial_checkpoint_reused) {
      "checkpoint"
    } else {
      "trained"
    },
    capacity$elapsed_seconds
  )
  if (!isTRUE(capacity_pass)) {
    runtime$status <- "failed_capacity"
    verdict <- paste(
      "OPTIMISATION/IMPLEMENTATION FAILURE:",
      "the small-initialization, unregularized network could not",
      "deliberately overfit the tiny respondent sample.",
      "The production rescue family was not run."
    )
    write_verdict(verdict)
    return(invisible(list(
      one_step = one_step,
      capacity = capacity_summary,
      verdict = verdict
    )))
  }

  cat("\nStage 2b: tiny-sample mechanism matrix.\n")
  mechanism_variants <- if (smoke_mode) {
    capacity_variant
  } else {
    variant_registry$variant
  }
  mechanism_rows <- list()
  for (index in seq_along(mechanism_variants)) {
    variant <- mechanism_variants[[index]]
    config <- variant_config(
      transformer, variant,
      stage = "mechanism"
    )
    checkpoint <- file.path(
      rescue_output_dir,
      sprintf(
        "mechanism_%s_seed_%d.rds",
        variant,
        fold_one_seed + 910000L
      )
    )
    fitted <- fit_rescue_once(
      transformer,
      tiny$features,
      tiny$features,
      tiny$offset,
      tiny$offset,
      config,
      variant,
      fold_one_seed + 910000L,
      outer_fold = 1L,
      checkpoint = checkpoint,
      fitting_cases =
        tiny$features$respondent_cases
    )
    row <- variant_registry[
      variant_registry$variant == variant,
      ,
      drop = FALSE
    ]
    row$epochs <- config$epochs
    row$offset_logloss <-
      fitted$training_offset_logloss
    row$trained_logloss <-
      fitted$final_training_logloss
    row$loss_drop <-
      fitted$training_logloss_change
    row$training_rms_correction <-
      fitted$training_rms_correction
    row$maximum_absolute_correction <-
      fitted$maximum_absolute_correction
    row$elapsed_seconds <-
      fitted$elapsed_seconds
    row$checkpoint_reused <-
      fitted$checkpoint_reused
    mechanism_rows[[index]] <- row
    trace <- fitted$trace
    trace$variant <- variant
    write.csv(
      trace,
      file.path(
        rescue_output_dir,
        paste0(
          "mechanism_trace_", variant,
          ".csv"
        )
      ),
      row.names = FALSE
    )
    record_runtime(
      "mechanism",
      variant,
      1L,
      fold_one_seed + 910000L,
      if (fitted$checkpoint_reused) {
        "checkpoint"
      } else {
        "trained"
      },
      fitted$elapsed_seconds
    )
  }
  mechanism_summary <- do.call(
    rbind, mechanism_rows
  )
  write.csv(
    mechanism_summary,
    file.path(
      rescue_output_dir,
      "mechanism_summary.csv"
    ),
    row.names = FALSE
  )
  print(mechanism_summary, digits = 8)

  if (smoke_mode) {
    runtime$status <- "smoke_passed"
    verdict <- paste(
      "SMOKE PASS:",
      "the exact v14 artifact loaded, the original output layer",
      "received a gradient and moved, the shortened capacity control",
      "produced nonzero corrections, and a rescue checkpoint reloaded",
      "under the frozen artifact contracts.",
      "This is an API/runtime result, not a model verdict."
    )
    write_verdict(verdict)
    cat("\nTransformer rescue smoke test completed successfully.\n")
    return(invisible(list(
      one_step = one_step,
      capacity = capacity_summary,
      mechanism = mechanism_summary,
      verdict = verdict
    )))
  }

  cat("\nStage 3: six-candidate production screen on fold 1.\n")
  screen_rows <- list()
  screen_predictions <- list()
  v14_loss <- transformer$log_loss_matrix_local(
    fold_one$validation_features$truth,
    fold_one$v14_prediction
  )
  offset_validation_loss <-
    transformer$log_loss_matrix_local(
      fold_one$validation_features$truth,
      fold_one$validation_offset
    )
  for (index in seq_len(nrow(variant_registry))) {
    variant <- variant_registry$variant[[index]]
    config <- variant_config(
      transformer, variant,
      stage = "production"
    )
    checkpoint <- file.path(
      rescue_output_dir,
      sprintf(
        "screen_%s_fold_1_seed_%d.rds",
        variant, fold_one_seed
      )
    )
    cat(
      "\nScreen candidate ", index, "/",
      nrow(variant_registry), ": ",
      variant, "\n",
      sep = ""
    )
    fitted <- fit_rescue_once(
      transformer,
      fold_one$training_features,
      fold_one$validation_features,
      fold_one$training_offset,
      fold_one$validation_offset,
      config,
      variant,
      fold_one_seed,
      outer_fold = 1L,
      checkpoint = checkpoint,
      fitting_cases =
        fold_one$fitting_cases
    )
    blend <- blend_on_grid(
      transformer,
      fold_one$validation_features$truth,
      fold_one$v14_prediction,
      fitted$prediction
    )
    write.csv(
      blend$grid,
      file.path(
        rescue_output_dir,
        paste0(
          "screen_blend_grid_", variant,
          ".csv"
        )
      ),
      row.names = FALSE
    )
    mechanically_eligible <-
      all(is.finite(c(
        fitted$maximum_absolute_correction,
        fitted$training_rms_correction,
        fitted$validation_logloss,
        blend$loss
      ))) &&
      fitted$maximum_absolute_correction >=
        1e-4 &&
      fitted$training_rms_correction >=
        1e-5 &&
      blend$weight > 0
    row <- variant_registry[index, , drop = FALSE]
    row$outer_fold <- 1L
    row$network_seed <- fold_one_seed
    row$offset_training_logloss <-
      fitted$training_offset_logloss
    row$final_training_logloss <-
      fitted$final_training_logloss
    row$training_logloss_change <-
      fitted$training_logloss_change
    row$training_rms_correction <-
      fitted$training_rms_correction
    row$offset_validation_logloss <-
      offset_validation_loss
    row$sequence_validation_logloss <-
      fitted$validation_logloss
    row$component_gain_vs_offset <-
      offset_validation_loss -
        fitted$validation_logloss
    row$v14_validation_logloss <- v14_loss
    row$selected_blend_weight <-
      blend$weight
    row$blended_validation_logloss <-
      blend$loss
    row$blended_gain_vs_v14 <-
      v14_loss - blend$loss
    row$rms_validation_correction <-
      fitted$rms_validation_correction
    row$maximum_absolute_correction <-
      fitted$maximum_absolute_correction
    row$saturation_share <-
      fitted$saturation_share
    row$elapsed_seconds <-
      fitted$elapsed_seconds
    row$checkpoint_reused <-
      fitted$checkpoint_reused
    row$mechanically_eligible <-
      mechanically_eligible
    screen_rows[[index]] <- row
    screen_predictions[[variant]] <- list(
      no =
        fold_one$validation_features$no,
      case =
        fold_one$validation_features$case,
      truth =
        fold_one$validation_features$truth,
      v14 = fold_one$v14_prediction,
      offset =
        fold_one$validation_offset,
      sequence = fitted$prediction,
      blended = (1 - blend$weight) *
        fold_one$v14_prediction +
        blend$weight * fitted$prediction
    )
    write.csv(
      do.call(rbind, screen_rows),
      file.path(
        rescue_output_dir,
        "screen_summary_partial.csv"
      ),
      row.names = FALSE
    )
    record_runtime(
      "screen",
      variant,
      1L,
      fold_one_seed,
      if (fitted$checkpoint_reused) {
        "checkpoint"
      } else {
        "trained"
      },
      fitted$elapsed_seconds,
      production_fit = TRUE
    )
  }
  screen_summary <- do.call(
    rbind, screen_rows
  )
  write.csv(
    screen_summary,
    file.path(
      rescue_output_dir, "screen_summary.csv"
    ),
    row.names = FALSE
  )
  safe_save_rds(
    screen_predictions,
    file.path(
      rescue_output_dir,
      "screen_predictions.rds"
    )
  )
  cat("\nFold-1 screen summary:\n")
  print(screen_summary, digits = 9)

  eligible <- screen_summary[
    screen_summary$mechanically_eligible &
      screen_summary$blended_gain_vs_v14 > 0,
    ,
    drop = FALSE
  ]
  if (nrow(eligible) == 0L) {
    runtime$status <- "rejected_at_screen"
    verdict <- paste(
      "REJECT:",
      "none of the six rescue candidates both learned nontrivial",
      "corrections and achieved a positive fold-1 blend gain over exact v14.",
      "Untouched folds 2-5 were not consumed."
    )
    write_verdict(verdict)
    return(invisible(list(
      one_step = one_step,
      capacity = capacity_summary,
      mechanism = mechanism_summary,
      screen = screen_summary,
      verdict = verdict
    )))
  }
  eligible <- eligible[
    order(
      -eligible$blended_gain_vs_v14,
      -eligible$component_gain_vs_offset,
      eligible$variant_order
    ),
    ,
    drop = FALSE
  ]
  finalists <- eligible[
    seq_len(min(2L, nrow(eligible))),
    ,
    drop = FALSE
  ]
  finalists$finalist_rank <-
    seq_len(nrow(finalists))
  write.csv(
    finalists,
    file.path(
      rescue_output_dir, "finalists.csv"
    ),
    row.names = FALSE
  )
  runtime$expected_production_fits <-
    6L + 4L * nrow(finalists)
  cat("\nAdvancing finalists with frozen fold-1 weights:\n")
  print(
    finalists[, c(
      "finalist_rank", "variant",
      "selected_blend_weight",
      "blended_gain_vs_v14"
    )],
    digits = 9
  )

  cat(
    "\nStage 4: untouched confirmation on canonical folds 2-5.\n"
  )
  confirmation_summaries <- list()
  confirmation_fold_rows <- list()
  for (
    finalist_index in seq_len(nrow(finalists))
  ) {
    finalist <- finalists[
      finalist_index, , drop = FALSE
    ]
    variant <- finalist$variant[[1L]]
    weight <-
      finalist$selected_blend_weight[[1L]]
    config <- variant_config(
      transformer, variant,
      stage = "production"
    )
    prediction_rows <- list()
    fold_rows <- list()
    for (outer_fold in 2:5) {
      cat(
        "\nConfirmation finalist ",
        finalist_index, "/", nrow(finalists),
        " (", variant, "), fold ",
        outer_fold, "/5.\n",
        sep = ""
      )
      fold <- prepare_canonical_fold(
        transformer,
        train,
        v14,
        outer_fold
      )
      seed <- transformer$
        network_seeds_for_fold(
          transformer$canonical_seed,
          outer_fold
        )[[1L]]
      checkpoint <- file.path(
        rescue_output_dir,
        sprintf(
          "confirm_%s_fold_%d_seed_%d.rds",
          variant, outer_fold, seed
        )
      )
      fitted <- fit_rescue_once(
        transformer,
        fold$training_features,
        fold$validation_features,
        fold$training_offset,
        fold$validation_offset,
        config,
        variant,
        seed,
        outer_fold,
        checkpoint,
        fold$fitting_cases
      )
      blended <- (1 - weight) *
        fold$v14_prediction +
        weight * fitted$prediction
      blended <- transformer$
        validate_probability(
          blended,
          nrow(fold$validation_features$truth)
        )
      fold_v14_loss <-
        transformer$log_loss_matrix_local(
          fold$validation_features$truth,
          fold$v14_prediction
        )
      fold_candidate_loss <-
        transformer$log_loss_matrix_local(
          fold$validation_features$truth,
          blended
        )
      fold_rows[[outer_fold - 1L]] <-
        data.frame(
          finalist_rank = finalist_index,
          variant = variant,
          outer_fold = outer_fold,
          network_seed = seed,
          frozen_blend_weight = weight,
          v14_logloss = fold_v14_loss,
          candidate_logloss =
            fold_candidate_loss,
          gain = fold_v14_loss -
            fold_candidate_loss,
          component_gain_vs_offset =
            transformer$log_loss_matrix_local(
              fold$validation_features$truth,
              fold$validation_offset
            ) -
            fitted$validation_logloss,
          training_logloss_change =
            fitted$training_logloss_change,
          training_rms_correction =
            fitted$training_rms_correction,
          rms_validation_correction =
            fitted$rms_validation_correction,
          maximum_absolute_correction =
            fitted$maximum_absolute_correction,
          saturation_share =
            fitted$saturation_share,
          elapsed_seconds =
            fitted$elapsed_seconds,
          checkpoint_reused =
            fitted$checkpoint_reused,
          stringsAsFactors = FALSE
        )
      prediction_rows[[outer_fold - 1L]] <-
        list(
          no =
            fold$validation_features$no,
          case =
            fold$validation_features$case,
          fold = rep(
            outer_fold,
            nrow(fold$validation_features$truth)
          ),
          truth =
            fold$validation_features$truth,
          v14 = fold$v14_prediction,
          offset =
            fold$validation_offset,
          sequence = fitted$prediction,
          candidate = blended
        )
      partial <- do.call(
        rbind, fold_rows
      )
      write.csv(
        partial,
        file.path(
          rescue_output_dir,
          paste0(
            "confirmation_folds_partial_",
            variant, ".csv"
          )
        ),
        row.names = FALSE
      )
      record_runtime(
        "confirmation",
        variant,
        outer_fold,
        seed,
        if (fitted$checkpoint_reused) {
          "checkpoint"
        } else {
          "trained"
        },
        fitted$elapsed_seconds,
        production_fit = TRUE
      )
    }

    combined <- list(
      no = unlist(lapply(
        prediction_rows, `[[`, "no"
      )),
      case = unlist(lapply(
        prediction_rows, `[[`, "case"
      )),
      fold = unlist(lapply(
        prediction_rows, `[[`, "fold"
      )),
      truth = do.call(
        rbind, lapply(
          prediction_rows, `[[`, "truth"
        )
      ),
      v14 = do.call(
        rbind, lapply(
          prediction_rows, `[[`, "v14"
        )
      ),
      offset = do.call(
        rbind, lapply(
          prediction_rows, `[[`, "offset"
        )
      ),
      sequence = do.call(
        rbind, lapply(
          prediction_rows, `[[`, "sequence"
        )
      ),
      candidate = do.call(
        rbind, lapply(
          prediction_rows, `[[`, "candidate"
        )
      )
    )
    order <- order(combined$no)
    combined$no <- combined$no[order]
    combined$case <- combined$case[order]
    combined$fold <- combined$fold[order]
    combined$truth <- combined$truth[
      order, , drop = FALSE
    ]
    combined$v14 <- combined$v14[
      order, , drop = FALSE
    ]
    combined$offset <- combined$offset[
      order, , drop = FALSE
    ]
    combined$sequence <- combined$sequence[
      order, , drop = FALSE
    ]
    combined$candidate <- combined$candidate[
      order, , drop = FALSE
    ]
    stopifnot(
      !anyDuplicated(combined$no),
      all(combined$fold %in% 2:5),
      all(rowSums(combined$truth) == 1L)
    )
    gain <- task_gain(
      combined$truth,
      combined$v14,
      combined$candidate
    )
    bootstrap <- bootstrap_confirmation(
      combined$case,
      gain,
      bootstrap_replicates,
      seed = 740200L + finalist_index
    )
    folds <- do.call(rbind, fold_rows)
    summary <- cbind(
      data.frame(
        finalist_rank = finalist_index,
        variant = variant,
        frozen_blend_weight = weight,
        screen_gain =
          finalist$blended_gain_vs_v14[[1L]],
        confirmation_v14_logloss =
          transformer$log_loss_matrix_local(
            combined$truth,
            combined$v14
          ),
        confirmation_candidate_logloss =
          transformer$log_loss_matrix_local(
            combined$truth,
            combined$candidate
          ),
        positive_confirmation_folds =
          sum(folds$gain > 0),
        n_confirmation_folds = 4L,
        maximum_absolute_correction =
          max(
            folds$maximum_absolute_correction
          ),
        maximum_saturation_share =
          max(folds$saturation_share),
        stringsAsFactors = FALSE
      ),
      bootstrap$summary
    )
    summary$rescue_pass <-
      summary$point_gain >= 0.001 &&
      summary$positive_confirmation_folds >=
        3L &&
      summary$lower_95 > 0 &&
      summary$bonferroni6_lower > 0 &&
      summary$maximum_saturation_share <
        0.01
    summary$promising_unconfirmed <-
      !summary$rescue_pass &&
      summary$point_gain > 0 &&
      summary$positive_confirmation_folds >=
        3L
    confirmation_summaries[[finalist_index]] <-
      summary
    confirmation_fold_rows[[finalist_index]] <-
      folds
    safe_save_rds(
      list(
        rescue_id = rescue_id,
        transformer_runner_md5 =
          observed_transformer_runner_md5,
        variant = variant,
        fold_one_selected_weight = weight,
        predictions = combined,
        task_gain = gain,
        respondent_gain =
          bootstrap$respondent_gain,
        bootstrap = bootstrap$bootstrap,
        summary = summary,
        fold_summary = folds
      ),
      file.path(
        rescue_output_dir,
        paste0(
          "confirmation_result_", variant,
          ".rds"
        )
      )
    )
  }

  confirmation_summary <- do.call(
    rbind, confirmation_summaries
  )
  confirmation_folds <- do.call(
    rbind, confirmation_fold_rows
  )
  write.csv(
    confirmation_summary,
    file.path(
      rescue_output_dir,
      "confirmation_summary.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    confirmation_folds,
    file.path(
      rescue_output_dir,
      "confirmation_by_fold.csv"
    ),
    row.names = FALSE
  )
  cat("\nUntouched folds 2-5 confirmation summary:\n")
  print(confirmation_summary, digits = 9)

  if (any(confirmation_summary$rescue_pass)) {
    runtime$status <- "rescue_pass"
    winners <- paste(
      confirmation_summary$variant[
        confirmation_summary$rescue_pass
      ],
      collapse = ", "
    )
    verdict <- paste(
      "RESCUE PASS:",
      winners,
      "cleared the pre-registered effect-size, fold-consistency,",
      "ordinary interval, Bonferroni-six interval, and saturation gates",
      "on untouched canonical folds 2-5.",
      "This does not generate or authorize a submission;",
      "audit the artifacts before any full-data build."
    )
  } else if (
    any(
      confirmation_summary$
        promising_unconfirmed
    )
  ) {
    runtime$status <-
      "promising_but_unconfirmed"
    names <- paste(
      confirmation_summary$variant[
        confirmation_summary$
          promising_unconfirmed
      ],
      collapse = ", "
    )
    verdict <- paste(
      "PROMISING BUT UNCONFIRMED:",
      names,
      "had positive pooled gain with at least 3/4 untouched",
      "confirmation folds improving, but failed at least one",
      "pre-registered size or uncertainty gate.",
      "Do not generate a submission from this runner."
    )
  } else {
    runtime$status <- "rejected_confirmation"
    verdict <- paste(
      "REJECT:",
      "no finalist produced a positive, fold-consistent confirmation",
      "result on untouched canonical folds 2-5.",
      "The respondent-sequence rescue route is closed."
    )
  }
  write_verdict(verdict)
  cat(
    "\nComprehensive Transformer rescue suite completed.\n",
    "Primary result: confirmation_summary.csv\n",
    "Monitoring: runtime_progress.csv and runtime_runs.csv\n",
    sep = ""
  )
  invisible(list(
    one_step = one_step,
    capacity = capacity_summary,
    mechanism = mechanism_summary,
    screen = screen_summary,
    finalists = finalists,
    confirmation_by_fold =
      confirmation_folds,
    confirmation = confirmation_summary,
    verdict = verdict
  ))
}

run_rescue_suite()
