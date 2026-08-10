# Respondent-sequence residual Transformer around the fold-fitted m8trpg
# utility, evaluated as a diversity component for exact v14.
#
# Pre-registration:
#   codex_respondent_sequence_transformer_preregister.md
#
# Smoke test:
#   Sys.setenv(SEQUENCE_TRANSFORMER_SMOKE = "1")
#   source("R/codex_respondent_sequence_transformer.R")
#   Sys.unsetenv("SEQUENCE_TRANSFORMER_SMOKE")
#
# Full run:
#   Sys.unsetenv("SEQUENCE_TRANSFORMER_SMOKE")
#   Sys.setenv(SEQUENCE_TRANSFORMER_REPEATED = "auto")
#   source("R/codex_respondent_sequence_transformer.R")

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

experiment_id <- "respondent_sequence_transformer_v1"
output_dir <- file.path(
  "data_processed",
  "codex_respondent_sequence_transformer"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

smoke_mode <- identical(
  Sys.getenv("SEQUENCE_TRANSFORMER_SMOKE", "0"),
  "1"
)
repeated_mode <- tolower(
  Sys.getenv("SEQUENCE_TRANSFORMER_REPEATED", "auto")
)
stopifnot(
  repeated_mode %in% c("auto", "never", "always")
)

canonical_seed <- 4821L
additional_seeds <- c(
  1907L, 2719L, 6151L, 8293L, 104729L
)
all_outer_seeds <- c(
  canonical_seed, additional_seeds
)
near_miss_lower_limit <- -0.00075
blend_weight_grid <- seq(0, 0.30, by = 0.01)
expected_v14_repeated_gain <- 0.0011636195
expected_v14_repeated_lower <- 0.0000105775
v14_metric_tolerance <- 5e-8

sequence_config <- list(
  profile_hidden = 128L,
  profile_embedding = 64L,
  profile_dropout = 0.10,
  model_width = 128L,
  attention_heads = 8L,
  transformer_layers = 4L,
  feedforward_width = 384L,
  transformer_dropout = 0.15,
  utility_hidden = c(128L, 64L),
  utility_dropout = 0.15,
  correction_bound = 0.35,
  correction_penalty = 0.01,
  learning_rate = 0.0007,
  weight_decay = 0.002,
  epochs = 120L,
  batch_respondents = 64L,
  gradient_clip = 1.0
)
canonical_network_seeds <- 18701:18703
bootstrap_replicates <- if (smoke_mode) {
  200L
} else {
  as.integer(Sys.getenv(
    "SEQUENCE_TRANSFORMER_N_BOOT",
    "100000"
  ))
}
progress_every_epochs <- 10L
models_per_outer_seed <-
  5L * length(canonical_network_seeds)
maximum_full_models <-
  length(all_outer_seeds) *
  models_per_outer_seed

runtime_state <- new.env(parent = emptyenv())
runtime_state$run_id <- paste0(
  format(Sys.time(), "%Y%m%dT%H%M%S"),
  "_pid",
  Sys.getpid()
)
runtime_state$started_at <- Sys.time()
runtime_state$stage <- if (smoke_mode) {
  "smoke"
} else {
  "canonical"
}
runtime_state$target_models <- if (smoke_mode) {
  1L
} else {
  models_per_outer_seed
}
runtime_state$final_status <- "running"
runtime_state$jobs <- data.frame(
  job_key = character(),
  model_elapsed_seconds = numeric(),
  stringsAsFactors = FALSE
)
runtime_progress_path <- file.path(
  output_dir, "runtime_progress.csv"
)
runtime_runs_path <- file.path(
  output_dir, "runtime_runs.csv"
)

format_duration <- function(seconds) {
  if (
    length(seconds) != 1L ||
    !is.finite(seconds)
  ) {
    return("unknown")
  }
  seconds <- max(0L, as.integer(round(seconds)))
  days <- seconds %/% 86400L
  seconds <- seconds %% 86400L
  hours <- seconds %/% 3600L
  seconds <- seconds %% 3600L
  minutes <- seconds %/% 60L
  seconds <- seconds %% 60L
  parts <- character()
  if (days > 0L) {
    parts <- c(parts, sprintf("%dd", days))
  }
  if (hours > 0L || days > 0L) {
    parts <- c(parts, sprintf("%dh", hours))
  }
  if (
    minutes > 0L ||
    hours > 0L ||
    days > 0L
  ) {
    parts <- c(parts, sprintf("%dm", minutes))
  }
  if (days == 0L && hours == 0L) {
    parts <- c(parts, sprintf("%ds", seconds))
  }
  paste(parts, collapse = " ")
}

append_runtime_row <- function(row, path) {
  tryCatch(
    {
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
      TRUE
    },
    error = function(error) {
      warning(
        "Could not write timing telemetry to ",
        path,
        ": ",
        conditionMessage(error),
        call. = FALSE
      )
      FALSE
    }
  )
}

set_runtime_stage <- function(stage, target_models) {
  runtime_state$stage <- as.character(stage)
  runtime_state$target_models <-
    as.integer(target_models)
}

runtime_snapshot <- function() {
  completed_models <- nrow(runtime_state$jobs)
  cumulative_model_seconds <- if (
    completed_models > 0L
  ) {
    sum(
      runtime_state$jobs$
        model_elapsed_seconds
    )
  } else {
    0
  }
  mean_model_seconds <- if (
    completed_models > 0L
  ) {
    mean(
      runtime_state$jobs$
        model_elapsed_seconds
    )
  } else {
    NA_real_
  }
  remaining_models <- max(
    0L,
    runtime_state$target_models -
      completed_models
  )
  eta_seconds <- if (
    is.finite(mean_model_seconds)
  ) {
    remaining_models * mean_model_seconds
  } else {
    NA_real_
  }
  full_remaining_models <- max(
    0L,
    maximum_full_models -
      completed_models
  )
  conditional_full_eta_seconds <- if (
    !smoke_mode &&
    is.finite(mean_model_seconds)
  ) {
    full_remaining_models *
      mean_model_seconds
  } else {
    NA_real_
  }
  list(
    completed_models = completed_models,
    cumulative_model_seconds =
      cumulative_model_seconds,
    mean_model_seconds = mean_model_seconds,
    remaining_models = remaining_models,
    eta_seconds = eta_seconds,
    conditional_full_eta_seconds =
      conditional_full_eta_seconds
  )
}

record_runtime_progress <- function(
    outer_seed,
    outer_fold,
    network_seed,
    source,
    model_elapsed_seconds) {
  job_key <- paste(
    outer_seed,
    outer_fold,
    network_seed,
    sep = ":"
  )
  existing <- match(
    job_key, runtime_state$jobs$job_key
  )
  job_row <- data.frame(
    job_key = job_key,
    model_elapsed_seconds =
      as.numeric(model_elapsed_seconds),
    stringsAsFactors = FALSE
  )
  if (is.na(existing)) {
    runtime_state$jobs <- rbind(
      runtime_state$jobs, job_row
    )
  } else {
    runtime_state$jobs[existing, ] <- job_row
  }
  snapshot <- runtime_snapshot()
  now <- Sys.time()
  run_elapsed_seconds <- as.numeric(
    difftime(
      now,
      runtime_state$started_at,
      units = "secs"
    )
  )
  estimated_finish_at <- if (
    is.finite(snapshot$eta_seconds)
  ) {
    format(
      now + snapshot$eta_seconds,
      "%Y-%m-%d %H:%M:%S %Z"
    )
  } else {
    NA_character_
  }
  conditional_full_finish_at <- if (
    is.finite(
      snapshot$
        conditional_full_eta_seconds
    )
  ) {
    format(
      now +
        snapshot$
          conditional_full_eta_seconds,
      "%Y-%m-%d %H:%M:%S %Z"
    )
  } else {
    NA_character_
  }
  row <- data.frame(
    run_id = runtime_state$run_id,
    event_at = format(
      now, "%Y-%m-%d %H:%M:%S %Z"
    ),
    stage = runtime_state$stage,
    outer_seed = as.integer(outer_seed),
    outer_fold = as.integer(outer_fold),
    network_seed =
      as.integer(network_seed),
    source = as.character(source),
    model_elapsed_seconds =
      as.numeric(model_elapsed_seconds),
    run_elapsed_seconds =
      run_elapsed_seconds,
    completed_models =
      snapshot$completed_models,
    target_models =
      runtime_state$target_models,
    cumulative_model_seconds =
      snapshot$cumulative_model_seconds,
    mean_model_seconds =
      snapshot$mean_model_seconds,
    eta_seconds = snapshot$eta_seconds,
    estimated_finish_at =
      estimated_finish_at,
    conditional_full_eta_seconds =
      snapshot$
        conditional_full_eta_seconds,
    conditional_full_finish_at =
      conditional_full_finish_at,
    stringsAsFactors = FALSE
  )
  append_runtime_row(
    row, runtime_progress_path
  )
  cat(sprintf(
    paste0(
      "  progress: %s %d/%d models | ",
      "model compute %s | this source %s | ",
      "mean %s/model | ",
      "ETA %s (about %s)\n"
    ),
    runtime_state$stage,
    snapshot$completed_models,
    runtime_state$target_models,
    format_duration(
      snapshot$cumulative_model_seconds
    ),
    format_duration(run_elapsed_seconds),
    format_duration(
      snapshot$mean_model_seconds
    ),
    format_duration(snapshot$eta_seconds),
    ifelse(
      is.na(estimated_finish_at),
      "unknown",
      estimated_finish_at
    )
  ))
  if (
    identical(runtime_state$stage, "canonical") &&
    repeated_mode != "never" &&
    is.finite(
      snapshot$
        conditional_full_eta_seconds
    )
  ) {
    cat(sprintf(
      paste0(
        "  if repeated CV activates: ",
        "%s remaining (about %s)\n"
      ),
      format_duration(
        snapshot$
          conditional_full_eta_seconds
      ),
      conditional_full_finish_at
    ))
  }
  flush.console()
  invisible(row)
}

finalize_runtime <- function() {
  ended_at <- Sys.time()
  snapshot <- runtime_snapshot()
  status <- runtime_state$final_status
  if (identical(status, "running")) {
    status <- "stopped_or_error"
  }
  row <- data.frame(
    run_id = runtime_state$run_id,
    started_at = format(
      runtime_state$started_at,
      "%Y-%m-%d %H:%M:%S %Z"
    ),
    ended_at = format(
      ended_at, "%Y-%m-%d %H:%M:%S %Z"
    ),
    status = status,
    stage = runtime_state$stage,
    elapsed_seconds = as.numeric(
      difftime(
        ended_at,
        runtime_state$started_at,
        units = "secs"
      )
    ),
    completed_models =
      snapshot$completed_models,
    target_models =
      runtime_state$target_models,
    cumulative_model_seconds =
      snapshot$cumulative_model_seconds,
    stringsAsFactors = FALSE
  )
  append_runtime_row(row, runtime_runs_path)
  cat(sprintf(
    paste0(
      "\nRun timing: %s (%s). ",
      "Model checkpoints observed: %d/%d ",
      "(%s model compute).\n",
      "Timing files: %s and %s\n"
    ),
    format_duration(row$elapsed_seconds),
    status,
    snapshot$completed_models,
    runtime_state$target_models,
    format_duration(
      snapshot$cumulative_model_seconds
    ),
    runtime_progress_path,
    runtime_runs_path
  ))
  flush.console()
  invisible(row)
}

attrs <- c(
  "CC", "GN", "NS", "BU", "FA", "LD", "BZ", "FC", "FP", "RP",
  "PP", "KA", "SC", "TS", "NV", "MA", "LB", "AF", "HU"
)
low_rank_helper_path <- file.path(
  "R", "codex_low_rank_factorization.R"
)
expected_low_rank_helper_md5 <-
  "1e48e15c7e096e6c16b3742a6eb978f9"
v14_dir <- file.path(
  "data_processed", "codex_set_context_network"
)
input_files <- c(
  file.path("csv files", "train.csv"),
  file.path("csv files", "test.csv"),
  file.path("R", "codex_shift_common.R"),
  file.path("R", "codex_modeling_common.R"),
  low_rank_helper_path,
  file.path(v14_dir, "canonical_result.rds"),
  file.path(v14_dir, "repeated_cv_summary.csv")
)
missing_files <- input_files[!file.exists(input_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing required project file(s):\n  ",
    paste(missing_files, collapse = "\n  "),
    "\nRun this script from the kagglecomp repository root after ",
    "the completed v14 experiment."
  )
}
observed_helper_md5 <- unname(
  tools::md5sum(low_rank_helper_path)
)
if (!identical(
  observed_helper_md5,
  expected_low_rank_helper_md5
)) {
  stop(
    "The audited low-rank helper script does not match the ",
    "version used to build this runner.\nExpected MD5: ",
    expected_low_rank_helper_md5,
    "\nObserved MD5: ",
    observed_helper_md5
  )
}

torch::torch_set_num_threads(4L)

load_audited_helpers <- function() {
  expressions <- parse(
    file = low_rank_helper_path,
    keep.source = FALSE
  )
  stopifnot(length(expressions) > 1L)
  final_expression <- paste(
    deparse(expressions[[length(expressions)]]),
    collapse = ""
  )
  if (!identical(final_expression, "run_experiment()")) {
    stop(
      "The audited helper no longer ends with the expected ",
      "run_experiment() call."
    )
  }
  helper <- new.env(parent = globalenv())
  for (
    index in seq_len(length(expressions) - 1L)
  ) {
    eval(expressions[[index]], envir = helper)
  }
  required <- c(
    "validate_probability",
    "log_loss_matrix_local",
    "row_log_loss_local",
    "numeric_signature",
    "same_signature",
    "m8trpg_helper_environment",
    "fit_m8trpg_offset",
    "active_attribute_levels",
    "active_one_hot",
    "fit_respondent_scaler",
    "standardize_respondents",
    "load_v14_artifact",
    "load_v14_registry",
    "bootstrap_summary",
    "bootstrap_case_means",
    "respondent_gain",
    "fit_test_propensity",
    "test_like_diagnostic"
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

audited <- load_audited_helpers()
validate_probability <- audited$validate_probability
log_loss_matrix_local <- audited$log_loss_matrix_local
row_log_loss_local <- audited$row_log_loss_local
numeric_signature <- audited$numeric_signature
same_signature <- audited$same_signature

active_one_hot <- function(
    values,
    levels,
    prefix) {
  audited$active_one_hot(
    values, levels, prefix
  )
}

price_one_hot <- function(values) {
  active_one_hot(
    values,
    1:12,
    "Price_"
  )
}

choice_context <- function(long) {
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
  n_tasks <- nrow(long) %/% 4L
  price <- matrix(
    as.numeric(long$Price),
    nrow = n_tasks,
    ncol = 4L,
    byrow = TRUE
  )
  inside_price <- price[, 1:3, drop = FALSE]
  price_min <- apply(inside_price, 1L, min)
  price_max <- apply(inside_price, 1L, max)
  gap_min <- cbind(
    sweep(inside_price, 1L, price_min, "-"),
    0
  )
  gap_max <- cbind(
    sweep(inside_price, 1L, price_max, function(x, y) y - x),
    0
  )
  cheapest <- cbind(
    sweep(inside_price, 1L, price_min, "==") * 1,
    0
  )
  dearest <- cbind(
    sweep(inside_price, 1L, price_max, "==") * 1,
    0
  )
  long$price_gap_min_raw <- as.numeric(t(gap_min))
  long$price_gap_max_raw <- as.numeric(t(gap_max))
  long$is_cheapest_sequence <- as.numeric(t(cheapest))
  long$is_dearest_sequence <- as.numeric(t(dearest))
  stopifnot(
    all(is.finite(long$price_gap_min_raw)),
    all(is.finite(long$price_gap_max_raw))
  )
  long
}

fit_gap_scaler <- function(long) {
  long <- choice_context(long)
  fields <- c(
    "price_gap_min_raw",
    "price_gap_max_raw"
  )
  centre <- vapply(
    long[, fields, drop = FALSE],
    mean,
    numeric(1)
  )
  scale <- vapply(
    long[, fields, drop = FALSE],
    sd,
    numeric(1)
  )
  scale[!is.finite(scale) | scale == 0] <- 1
  list(
    centre = centre,
    scale = scale,
    fields = fields
  )
}

build_sequence_features <- function(
    long,
    levels_by_attribute,
    respondent_scaler,
    gap_scaler) {
  long <- choice_context(long)
  first_rows <- seq.int(
    1L, nrow(long), by = 4L
  )
  n_tasks <- length(first_rows)
  stopifnot(
    nrow(long) == n_tasks * 4L,
    identical(
      as.integer(long$alt),
      rep(1:4, n_tasks)
    )
  )

  attribute_parts <- lapply(
    attrs,
    function(attribute) {
      active_one_hot(
        long[[attribute]],
        levels_by_attribute[[attribute]],
        paste0(attribute, "_")
      )
    }
  )
  profile <- do.call(
    cbind,
    c(
      attribute_parts,
      list(
        price_one_hot(long$Price),
        inside = as.numeric(
          as.integer(long$alt) <= 3L
        ),
        d2 = as.numeric(
          as.integer(long$alt) == 2L
        ),
        d3 = as.numeric(
          as.integer(long$alt) == 3L
        ),
        price_gap_min = (
          long$price_gap_min_raw -
            gap_scaler$centre[["price_gap_min_raw"]]
        ) /
          gap_scaler$scale[["price_gap_min_raw"]],
        price_gap_max = (
          long$price_gap_max_raw -
            gap_scaler$centre[["price_gap_max_raw"]]
        ) /
          gap_scaler$scale[["price_gap_max_raw"]],
        is_cheapest =
          long$is_cheapest_sequence,
        is_dearest =
          long$is_dearest_sequence
      )
    )
  )
  storage.mode(profile) <- "double"

  respondent <- audited$standardize_respondents(
    long[first_rows, , drop = FALSE],
    respondent_scaler
  )
  task_c <- (
    as.numeric(long$Task[first_rows]) - 10
  ) / 9
  common <- cbind(
    respondent,
    Task_c = task_c
  )
  storage.mode(common) <- "double"

  truth <- matrix(
    as.integer(long$chosen),
    ncol = 4L,
    byrow = TRUE
  )
  no <- as.integer(long$No[first_rows])
  case <- as.integer(long$Case[first_rows])
  task <- as.integer(long$Task[first_rows])
  respondent_cases <- unique(case)
  stopifnot(
    all(rowSums(truth) == 1L),
    length(unique(no)) == n_tasks,
    all(table(case) == 19L),
    length(respondent_cases) * 19L == n_tasks,
    identical(
      task,
      rep(1:19, times = length(respondent_cases))
    ),
    identical(
      case,
      rep(respondent_cases, each = 19L)
    ),
    all(is.finite(profile)),
    all(is.finite(common)),
    !any(grepl(
      "chosen|^Ch[1-4]$|^Case$|^No$|version",
      c(colnames(profile), colnames(common)),
      ignore.case = TRUE
    ))
  )

  opt_out_rows <- seq.int(
    4L, nrow(profile), by = 4L
  )
  attr_profile_columns <- unlist(
    lapply(
      attribute_parts,
      colnames
    ),
    use.names = FALSE
  )
  price_profile_columns <- paste0(
    "Price_", 1:12
  )
  zero_reference_columns <- c(
    attr_profile_columns,
    price_profile_columns,
    "inside", "d2", "d3",
    "is_cheapest", "is_dearest"
  )
  stopifnot(
    max(abs(
      profile[
        opt_out_rows,
        zero_reference_columns,
        drop = FALSE
      ]
    )) == 0
  )

  list(
    profile = profile,
    common = common,
    truth = truth,
    no = no,
    case = case,
    task = task,
    respondent_cases = respondent_cases,
    n_respondents = length(respondent_cases),
    profile_columns = colnames(profile),
    common_columns = colnames(common)
  )
}

align_offset <- function(
    feature_no,
    offset_no,
    offset_prediction) {
  order <- match(feature_no, offset_no)
  stopifnot(
    !anyNA(order),
    length(unique(order)) == length(order)
  )
  validate_probability(
    offset_prediction[order, , drop = FALSE],
    length(feature_no)
  )
}

sequence_transformer_block <- nn_module(
  "sequence_transformer_block",
  initialize = function(
      model_width,
      attention_heads,
      feedforward_width,
      dropout) {
    self$norm_attention <- nn_layer_norm(
      as.integer(model_width)
    )
    self$attention <- nn_multihead_attention(
      embed_dim = as.integer(model_width),
      num_heads = as.integer(attention_heads),
      dropout = dropout,
      batch_first = TRUE
    )
    self$attention_dropout <- nn_dropout(
      p = dropout
    )
    self$norm_feedforward <- nn_layer_norm(
      as.integer(model_width)
    )
    self$feedforward <- nn_sequential(
      nn_linear(
        as.integer(model_width),
        as.integer(feedforward_width)
      ),
      nn_gelu(),
      nn_dropout(p = dropout),
      nn_linear(
        as.integer(feedforward_width),
        as.integer(model_width)
      )
    )
    self$feedforward_dropout <- nn_dropout(
      p = dropout
    )
  },
  forward = function(x) {
    normalized <- self$norm_attention(x)
    attention_output <- self$attention(
      normalized,
      normalized,
      normalized
    )[[1L]]
    x <- x + self$attention_dropout(
      attention_output
    )
    normalized <- self$norm_feedforward(x)
    x + self$feedforward_dropout(
      self$feedforward(normalized)
    )
  }
)

respondent_sequence_transformer <- nn_module(
  "respondent_sequence_transformer",
  initialize = function(
      profile_dim,
      common_dim,
      config) {
    self$profile_dim <- as.integer(profile_dim)
    self$common_dim <- as.integer(common_dim)
    self$profile_embedding_dim <-
      as.integer(config$profile_embedding)
    self$model_width <-
      as.integer(config$model_width)
    self$correction_bound <-
      as.numeric(config$correction_bound)

    self$profile_encoder <- nn_sequential(
      nn_linear(
        self$profile_dim,
        as.integer(config$profile_hidden)
      ),
      nn_gelu(),
      nn_dropout(p = config$profile_dropout),
      nn_linear(
        as.integer(config$profile_hidden),
        self$profile_embedding_dim
      ),
      nn_gelu()
    )

    task_input_dim <-
      2L * self$profile_embedding_dim +
      self$common_dim
    self$task_projection <- nn_sequential(
      nn_linear(
        as.integer(task_input_dim),
        self$model_width
      ),
      nn_gelu(),
      nn_dropout(
        p = config$transformer_dropout
      )
    )
    self$position_embedding <- nn_embedding(
      num_embeddings = 19L,
      embedding_dim = self$model_width
    )
    self$blocks <- nn_module_list(lapply(
      seq_len(config$transformer_layers),
      function(index) {
        sequence_transformer_block(
          model_width = config$model_width,
          attention_heads =
            config$attention_heads,
          feedforward_width =
            config$feedforward_width,
          dropout =
            config$transformer_dropout
        )
      }
    ))
    self$final_norm <- nn_layer_norm(
      self$model_width
    )

    utility_input_dim <-
      4L * self$profile_embedding_dim +
      self$model_width +
      self$common_dim
    self$utility_input_dim <-
      as.integer(utility_input_dim)
    self$utility_body <- nn_sequential(
      nn_linear(
        self$utility_input_dim,
        as.integer(config$utility_hidden[[1L]])
      ),
      nn_gelu(),
      nn_dropout(p = config$utility_dropout),
      nn_linear(
        as.integer(config$utility_hidden[[1L]]),
        as.integer(config$utility_hidden[[2L]])
      ),
      nn_gelu()
    )
    self$utility_output <- nn_linear(
      as.integer(config$utility_hidden[[2L]]),
      1L
    )
    nn_init_zeros_(self$utility_output$weight)
    nn_init_zeros_(self$utility_output$bias)
  },
  forward = function(
      profile,
      common,
      offset_log_probability) {
    batch_size <- profile$size(1L)
    task_count <- profile$size(2L)
    profile_embedding <- self$profile_encoder(
      profile$reshape(c(
        -1L, self$profile_dim
      ))
    )$reshape(c(
      batch_size,
      task_count,
      4L,
      self$profile_embedding_dim
    ))
    inside <- profile_embedding[
      , , 1:3, 
    ]
    inside_mean <- inside$mean(dim = 3L)
    inside_max <- torch_maximum(
      torch_maximum(
        inside[, , 1L, ],
        inside[, , 2L, ]
      ),
      inside[, , 3L, ]
    )

    task_token <- self$task_projection(
      torch_cat(
        list(
          inside_mean,
          inside_max,
          common
        ),
        dim = 3L
      )
    )
    positions <- torch_tensor(
      seq_len(19L),
      dtype = torch_long()
    )
    positional <- self$position_embedding(
      positions
    )$unsqueeze(1L)
    task_token <- task_token + positional
    for (index in seq_along(self$blocks)) {
      task_token <- self$blocks[[index]](
        task_token
      )
    }
    task_token <- self$final_norm(task_token)

    inside_mean_repeated <- torch_stack(
      list(
        inside_mean, inside_mean,
        inside_mean, inside_mean
      ),
      dim = 3L
    )
    inside_max_repeated <- torch_stack(
      list(
        inside_max, inside_max,
        inside_max, inside_max
      ),
      dim = 3L
    )
    task_repeated <- torch_stack(
      list(
        task_token, task_token,
        task_token, task_token
      ),
      dim = 3L
    )
    common_repeated <- torch_stack(
      list(common, common, common, common),
      dim = 3L
    )
    utility_input <- torch_cat(
      list(
        profile_embedding,
        inside_mean_repeated,
        inside_max_repeated,
        profile_embedding -
          inside_mean_repeated,
        task_repeated,
        common_repeated
      ),
      dim = 4L
    )
    raw_correction <- self$utility_output(
      self$utility_body(
        utility_input$reshape(c(
          -1L, self$utility_input_dim
        ))
      )
    )$reshape(c(
      batch_size, task_count, 4L
    ))
    correction <- self$correction_bound *
      torch_tanh(raw_correction)
    list(
      utility =
        offset_log_probability + correction,
      correction = correction
    )
  }
)

feature_tensors <- function(
    features,
    offset_prediction) {
  n_tasks <- nrow(features$truth)
  n_respondents <- features$n_respondents
  stopifnot(
    n_tasks == n_respondents * 19L,
    nrow(features$profile) == n_tasks * 4L
  )
  offset_prediction <- validate_probability(
    offset_prediction, n_tasks
  )
  list(
    profile = torch_tensor(
      features$profile,
      dtype = torch_float()
    )$reshape(c(
      n_respondents,
      19L,
      4L,
      ncol(features$profile)
    )),
    common = torch_tensor(
      features$common,
      dtype = torch_float()
    )$reshape(c(
      n_respondents,
      19L,
      ncol(features$common)
    )),
    offset = torch_tensor(
      log(pmax(offset_prediction, 1e-15)),
      dtype = torch_float()
    )$reshape(c(
      n_respondents, 19L, 4L
    )),
    target = torch_tensor(
      as.integer(max.col(features$truth)),
      dtype = torch_long()
    )$reshape(c(
      n_respondents, 19L
    ))
  )
}

predict_sequence_transformer <- function(
    model,
    features,
    offset_prediction,
    batch_respondents = 128L) {
  tensors <- feature_tensors(
    features, offset_prediction
  )
  n_respondents <- features$n_respondents
  prediction <- matrix(
    NA_real_,
    nrow(features$truth),
    4L
  )
  correction <- matrix(
    NA_real_,
    nrow(features$truth),
    4L
  )
  model$eval()
  with_no_grad({
    for (
      start in seq.int(
        1L,
        n_respondents,
        by = batch_respondents
      )
    ) {
      stop_at <- min(
        n_respondents,
        start + batch_respondents - 1L
      )
      output <- model(
        tensors$profile[start:stop_at, , , ],
        tensors$common[start:stop_at, , ],
        tensors$offset[start:stop_at, , ]
      )
      probability <- nnf_softmax(
        output$utility,
        dim = 3L
      )
      task_rows <- seq.int(
        (start - 1L) * 19L + 1L,
        stop_at * 19L
      )
      prediction[task_rows, ] <-
        as.matrix(as_array(
          probability$reshape(c(-1L, 4L))
        ))
      correction[task_rows, ] <-
        as.matrix(as_array(
          output$correction$reshape(
            c(-1L, 4L)
          )
        ))
    }
  })
  stopifnot(
    all(is.finite(correction)),
    max(abs(correction)) <=
      sequence_config$correction_bound + 1e-6
  )
  list(
    prediction = validate_probability(
      prediction, nrow(features$truth)
    ),
    correction = correction
  )
}

fit_sequence_once <- function(
    training_features,
    validation_features,
    training_offset,
    validation_offset,
    config,
    seed,
    checkpoint = NULL,
    fitting_cases = NULL) {
  validation_no <- as.integer(
    validation_features$no
  )
  if (
    !is.null(checkpoint) &&
    file.exists(checkpoint)
  ) {
    saved <- readRDS(checkpoint)
    if (
      identical(saved$experiment_id, experiment_id) &&
      identical(saved$config, config) &&
      identical(saved$seed, as.integer(seed)) &&
      identical(saved$validation_no, validation_no) &&
      identical(saved$fitting_cases, fitting_cases)
    ) {
      saved$result$prediction <-
        validate_probability(
          saved$result$prediction,
          length(validation_no)
        )
      stopifnot(
        all(is.finite(
          saved$result$correction_signature
        )),
        is.finite(
          saved$result$
            maximum_absolute_correction
        ),
        saved$result$
          maximum_absolute_correction <=
          config$correction_bound + 1e-6
      )
      cat(
        "  using verified network-seed checkpoint\n"
      )
      saved$result$checkpoint_reused <- TRUE
      return(saved$result)
    }
  }

  set.seed(as.integer(seed))
  torch_manual_seed(as.integer(seed))
  tensors <- feature_tensors(
    training_features,
    training_offset
  )
  n_respondents <-
    training_features$n_respondents
  model <- respondent_sequence_transformer(
    profile_dim =
      ncol(training_features$profile),
    common_dim =
      ncol(training_features$common),
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
    respondent_order <- sample.int(
      n_respondents
    )
    total_loss <- 0
    total_cross_entropy <- 0
    total_correction_penalty <- 0
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
      cross_entropy <- nnf_cross_entropy(
        output$utility$reshape(c(-1L, 4L)),
        tensors$target[
          respondent_index, 
        ]$reshape(c(-1L))
      )
      correction_penalty <-
        output$correction$pow(2)$mean()
      loss <- cross_entropy +
        config$correction_penalty *
        correction_penalty
      loss$backward()
      nn_utils_clip_grad_norm_(
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
      total_correction_penalty <-
        total_correction_penalty +
        correction_penalty$item() * task_count
      total_tasks <- total_tasks + task_count
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
        training_objective =
          total_loss / total_tasks,
        training_logloss =
          total_cross_entropy / total_tasks,
        mean_correction_square =
          total_correction_penalty /
          total_tasks
      )
      if (
        epoch %% progress_every_epochs == 0L ||
        epoch == config$epochs
      ) {
        epoch_elapsed <-
          proc.time()[["elapsed"]] - started
        epoch_eta <- (
          epoch_elapsed / epoch
        ) * (config$epochs - epoch)
        cat(sprintf(
          paste0(
            "    epoch %d/%d: train %.6f, ",
            "seed elapsed %s, seed ETA %s\n"
          ),
          epoch,
          config$epochs,
          total_cross_entropy /
            total_tasks,
          format_duration(epoch_elapsed),
          format_duration(epoch_eta)
        ))
        flush.console()
      }
    }
  }
  elapsed <-
    proc.time()[["elapsed"]] - started
  predicted <- predict_sequence_transformer(
    model,
    validation_features,
    validation_offset
  )
  result <- list(
    prediction = predicted$prediction,
    correction_signature =
      numeric_signature(predicted$correction),
    maximum_absolute_correction =
      max(abs(predicted$correction)),
    seed = as.integer(seed),
    elapsed_seconds = elapsed,
    final_training_logloss =
      total_cross_entropy / total_tasks,
    final_training_objective =
      total_loss / total_tasks,
    trace = do.call(rbind, trace_rows),
    checkpoint_reused = FALSE
  )
  if (!is.null(checkpoint)) {
    saveRDS(
      list(
        experiment_id = experiment_id,
        config = config,
        seed = as.integer(seed),
        validation_no = validation_no,
        fitting_cases = fitting_cases,
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
      as.integer(outer_fold) * 10L
  )
}

subset_training_respondents <- function(
    features,
    offset_prediction,
    maximum_respondents) {
  if (
    is.null(maximum_respondents) ||
    features$n_respondents <=
      maximum_respondents
  ) {
    return(list(
      features = features,
      offset = offset_prediction
    ))
  }
  keep_cases <- features$respondent_cases[
    seq_len(maximum_respondents)
  ]
  keep_tasks <- features$case %in% keep_cases
  keep_profile <- rep(
    keep_tasks, each = 4L
  )
  subset <- features
  subset$profile <- features$profile[
    keep_profile, , drop = FALSE
  ]
  subset$common <- features$common[
    keep_tasks, , drop = FALSE
  ]
  subset$truth <- features$truth[
    keep_tasks, , drop = FALSE
  ]
  subset$no <- features$no[keep_tasks]
  subset$case <- features$case[keep_tasks]
  subset$task <- features$task[keep_tasks]
  subset$respondent_cases <- keep_cases
  subset$n_respondents <- length(keep_cases)
  list(
    features = subset,
    offset = offset_prediction[
      keep_tasks, , drop = FALSE
    ]
  )
}

fit_outer_sequence <- function(
    fitting,
    validation,
    m8trpg_helper,
    outer_seed,
    outer_fold,
    config = sequence_config,
    seeds = NULL,
    checkpoint_tag = NULL,
    smoke_max_respondents = NULL) {
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
  if (is.null(seeds)) {
    seeds <- network_seeds_for_fold(
      outer_seed, outer_fold
    )
  }
  expected_validation_no <- sort(
    as.integer(validation$No)
  )
  outer_checkpoint <- if (
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
  if (
    !is.null(outer_checkpoint) &&
    file.exists(outer_checkpoint)
  ) {
    saved <- readRDS(outer_checkpoint)
    if (
      identical(saved$experiment_id, experiment_id) &&
      identical(saved$config, config) &&
      identical(saved$seeds, as.integer(seeds)) &&
      identical(
        saved$validation_no,
        expected_validation_no
      ) &&
      identical(saved$fitting_cases, fitting_cases)
    ) {
      saved$result$prediction <-
        validate_probability(
          saved$result$prediction,
          length(expected_validation_no)
        )
      saved$result$offset_prediction <-
        validate_probability(
          saved$result$offset_prediction,
          length(expected_validation_no)
        )
      stopifnot(
        identical(
          sort(saved$result$validation_cases),
          validation_cases
        ),
        length(intersect(
          saved$result$fitting_cases,
          saved$result$validation_cases
        )) == 0L
      )
      cat(
        "  using verified outer-fold checkpoint\n"
      )
      if (
        is.data.frame(saved$result$fits) &&
        nrow(saved$result$fits) ==
          length(seeds)
      ) {
        for (
          fit_index in seq_len(
            nrow(saved$result$fits)
          )
        ) {
          record_runtime_progress(
            outer_seed = outer_seed,
            outer_fold = outer_fold,
            network_seed =
              saved$result$fits$
                seed[[fit_index]],
            source = "outer_checkpoint",
            model_elapsed_seconds =
              saved$result$fits$
                elapsed_seconds[[fit_index]]
          )
        }
      }
      return(saved$result)
    }
  }

  offset <- audited$fit_m8trpg_offset(
    fitting, validation, m8trpg_helper
  )
  levels_by_attribute <-
    audited$active_attribute_levels(
      offset$fitting_long
    )
  respondent_scaler <-
    audited$fit_respondent_scaler(fitting)
  gap_scaler <- fit_gap_scaler(
    offset$fitting_long
  )
  training_features <- build_sequence_features(
    offset$fitting_long,
    levels_by_attribute,
    respondent_scaler,
    gap_scaler
  )
  validation_features <-
    build_sequence_features(
      offset$validation_long,
      levels_by_attribute,
      respondent_scaler,
      gap_scaler
    )
  stopifnot(
    identical(
      training_features$profile_columns,
      validation_features$profile_columns
    ),
    identical(
      training_features$common_columns,
      validation_features$common_columns
    ),
    identical(
      sort(training_features$respondent_cases),
      fitting_cases
    ),
    identical(
      sort(validation_features$respondent_cases),
      validation_cases
    )
  )

  training_offset <- align_offset(
    training_features$no,
    offset$fitting_no,
    offset$fitting_prediction
  )
  validation_offset <- align_offset(
    validation_features$no,
    offset$validation_no,
    offset$validation_prediction
  )
  smoke_subset <- subset_training_respondents(
    training_features,
    training_offset,
    smoke_max_respondents
  )
  network_training_features <-
    smoke_subset$features
  network_training_offset <-
    smoke_subset$offset

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
    seed <- as.integer(seeds[[seed_index]])
    seed_checkpoint <- if (
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
    fitted <- fit_sequence_once(
      network_training_features,
      validation_features,
      network_training_offset,
      validation_offset,
      config,
      seed,
      checkpoint = seed_checkpoint,
      fitting_cases =
        network_training_features$respondent_cases
    )
    prediction_array[, , seed_index] <-
      fitted$prediction
    fit_rows[[seed_index]] <- data.frame(
      outer_seed = as.integer(outer_seed),
      outer_fold = as.integer(outer_fold),
      seed = seed,
      training_respondents =
        network_training_features$n_respondents,
      validation_respondents =
        validation_features$n_respondents,
      profile_features =
        ncol(training_features$profile),
      common_features =
        ncol(training_features$common),
      transformer_layers =
        config$transformer_layers,
      model_width = config$model_width,
      attention_heads =
        config$attention_heads,
      m8trpg_coefficients =
        offset$coefficient_count,
      offset_training_logloss =
        log_loss_matrix_local(
          network_training_features$truth,
          network_training_offset
        ),
      elapsed_seconds =
        fitted$elapsed_seconds,
      final_training_logloss =
        fitted$final_training_logloss,
      maximum_absolute_correction =
        fitted$maximum_absolute_correction
    )
    record_runtime_progress(
      outer_seed = outer_seed,
      outer_fold = outer_fold,
      network_seed = seed,
      source = if (
        isTRUE(fitted$checkpoint_reused)
      ) {
        "network_checkpoint"
      } else {
        "trained"
      },
      model_elapsed_seconds =
        fitted$elapsed_seconds
    )
    cat(sprintf(
      paste0(
        "  seed %d: train %.6f, ",
        "max correction %.4f, %.1fs\n"
      ),
      seed,
      fitted$final_training_logloss,
      fitted$maximum_absolute_correction,
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
    common_columns =
      training_features$common_columns,
    fitting_cases = fitting_cases,
    validation_cases = validation_cases
  )
  if (!is.null(outer_checkpoint)) {
    saveRDS(
      list(
        experiment_id = experiment_id,
        config = config,
        seeds = as.integer(seeds),
        validation_no =
          expected_validation_no,
        fitting_cases = fitting_cases,
        result = result
      ),
      outer_checkpoint
    )
  }
  result
}

run_sequence_oof <- function(
    train,
    fold_map,
    m8trpg_helper,
    checkpoint_tag,
    outer_seed) {
  fold_map <- audited$validate_fold_map(
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
      paste0(
        "\n=== %s: sequence Transformer ",
        "outer fold %d/5 ===\n"
      ),
      checkpoint_tag,
      outer_fold
    ))
    validation_rows <- row_fold == outer_fold
    result <- fit_outer_sequence(
      train[
        !validation_rows, , drop = FALSE
      ],
      train[
        validation_rows, , drop = FALSE
      ],
      m8trpg_helper,
      outer_seed,
      outer_fold,
      config = sequence_config,
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
      length(rows) == sum(validation_rows),
      length(intersect(
        result$fitting_cases,
        result$validation_cases
      )) == 0L
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
        baseline[
          validation_rows, , drop = FALSE
        ] +
      best_weight *
        component[
          validation_rows, , drop = FALSE
        ]
    weight_rows[[fold]] <- data.frame(
      fold = fold,
      sequence_weight = best_weight,
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

fold_selection_table <- function(
    fold_results,
    weights,
    seed) {
  do.call(rbind, lapply(
    seq_along(fold_results),
    function(fold) {
      result <- fold_results[[fold]]
      data.frame(
        seed = as.integer(seed),
        fold = as.integer(fold),
        n_network_seeds =
          length(result$seeds),
        training_respondents =
          result$fits$training_respondents[[1L]],
        validation_respondents =
          result$fits$validation_respondents[[1L]],
        profile_features =
          length(result$profile_columns),
        common_features =
          length(result$common_columns),
        transformer_layers =
          sequence_config$transformer_layers,
        model_width =
          sequence_config$model_width,
        attention_heads =
          sequence_config$attention_heads,
        mean_training_logloss =
          mean(result$fits$final_training_logloss),
        mean_offset_training_logloss =
          mean(result$fits$offset_training_logloss),
        maximum_absolute_correction =
          max(
            result$fits$
              maximum_absolute_correction
          ),
        total_elapsed_seconds =
          sum(result$fits$elapsed_seconds),
        sequence_weight =
          weights$sequence_weight[
            weights$fold == fold
          ]
      )
    }
  ))
}

run_smoke_test <- function(
    train,
    truth,
    m8trpg_helper) {
  v14 <- audited$load_v14_artifact(
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
  smoke_config <- sequence_config
  smoke_config$epochs <- 2L
  smoke_seed <- network_seeds_for_fold(
    canonical_seed, outer_fold
  )[[1L]]
  cat(
    "SMOKE MODE: canonical outer fold 1, ",
    "one sequence-network seed, two epochs, ",
    "96 training respondents; no verdict.\n",
    sep = ""
  )
  result <- fit_outer_sequence(
    fitting,
    validation,
    m8trpg_helper,
    canonical_seed,
    outer_fold,
    config = smoke_config,
    seeds = smoke_seed,
    checkpoint_tag = "smoke",
    smoke_max_respondents = 96L
  )
  validation_rows <- match(
    result$validation_no,
    train$No
  )
  stopifnot(
    !anyNA(validation_rows),
    all(row_fold[validation_rows] == outer_fold),
    length(intersect(
      result$fitting_cases,
      result$validation_cases
    )) == 0L
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
      "Smoke sequence component log loss: %.6f\n",
      "Smoke v14 baseline log loss: %.6f\n"
    ),
    path,
    result$smoke_component_logloss,
    result$smoke_v14_logloss
  ))
  invisible(result)
}

run_experiment <- function() {
  on.exit(finalize_runtime(), add = TRUE)
  train <- read.csv(
    file.path("csv files", "train.csv")
  )
  test <- read.csv(
    file.path("csv files", "test.csv")
  )
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
    all(table(test$Case) == 19L),
    all(rowSums(truth) == 1L)
  )
  m8trpg_helper <-
    audited$m8trpg_helper_environment()

  cat("\nRunning", experiment_id, "\n")
  cat(
    "Architecture: 4 layers, width 128, 8 heads, ",
    "120 epochs, 3 seeds/fold.\n",
    sep = ""
  )
  cat(
    "Frozen v14 repeated gain:",
    sprintf("%.10f", expected_v14_repeated_gain),
    "\n"
  )
  cat(
    "Timing telemetry:",
    runtime_progress_path,
    "\n"
  )
  if (smoke_mode) {
    set_runtime_stage("smoke", 1L)
    result <- run_smoke_test(
      train, truth, m8trpg_helper
    )
    runtime_state$final_status <- "completed"
    return(result)
  }

  set_runtime_stage(
    "canonical", models_per_outer_seed
  )
  v14_registry <- audited$load_v14_registry(
    train, truth
  )
  canonical_v14 <-
    v14_registry$artifacts[[
      as.character(canonical_seed)
    ]]
  canonical_sequence <- run_sequence_oof(
    train,
    canonical_v14$fold_map,
    m8trpg_helper,
    checkpoint_tag = paste0(
      "seed_", canonical_seed
    ),
    outer_seed = canonical_seed
  )
  canonical_blend <- crossfit_blend(
    truth,
    canonical_v14$prediction,
    canonical_sequence$prediction,
    canonical_sequence$row_fold
  )
  canonical_candidate_loss <-
    log_loss_matrix_local(
      truth,
      canonical_blend$prediction
    )
  canonical_bootstrap <-
    audited$bootstrap_summary(
      truth,
      canonical_v14$prediction,
      canonical_blend$prediction,
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
          canonical_sequence$
            offset_prediction
        ),
      sequence_component_logloss =
        log_loss_matrix_local(
          truth,
          canonical_sequence$prediction
        ),
      candidate_logloss =
        canonical_candidate_loss
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
    canonical_sequence$fold_results,
    canonical_blend$weights,
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
      sequence_prediction =
        canonical_sequence$prediction,
      candidate_prediction =
        canonical_blend$prediction,
      offset_prediction =
        canonical_sequence$offset_prediction,
      fold_map = canonical_v14$fold_map,
      fold_results =
        canonical_sequence$fold_results,
      weights = canonical_blend$weights,
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
      paste0(
        "Repeated CV disabled by ",
        "SEQUENCE_TRANSFORMER_REPEATED=never."
      )
    } else {
      paste0(
        "Repeated CV not triggered: canonical ",
        "result was neither a pass nor the ",
        "pre-registered near miss."
      )
    }
    verdict <- paste(
      "REJECT respondent-sequence Transformer;",
      "retain exact v14.",
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
    runtime_state$final_status <- "completed"
    return(invisible(list(
      canonical = canonical_summary,
      repeated = NULL,
      verdict = verdict
    )))
  }

  cat(
    "\nRepeated-CV escalation activated (",
    repeated_mode,
    "). Completed fold/seed checkpoints ",
    "will be reused automatically.\n",
    sep = ""
  )
  set_runtime_stage(
    "repeated_cv", maximum_full_models
  )
  repeat_names <- as.character(
    all_outer_seeds
  )
  case_levels <- sort(unique(train$Case))
  case_gain_matrix <- matrix(
    NA_real_,
    nrow = length(case_levels),
    ncol = length(repeat_names),
    dimnames = list(
      as.character(case_levels),
      repeat_names
    )
  )
  case_gain_matrix[
    , as.character(canonical_seed)
  ] <- canonical_bootstrap$respondent_gain
  repeat_rows <- list()
  repeat_rows[[as.character(canonical_seed)]] <-
    data.frame(
      seed = canonical_seed,
      baseline_logloss =
        canonical_v14$logloss,
      sequence_component_logloss =
        log_loss_matrix_local(
          truth,
          canonical_sequence$prediction
        ),
      candidate_logloss =
        canonical_candidate_loss,
      gain = canonical_v14$logloss -
        canonical_candidate_loss
    )
  selection_rows <- list()
  selection_rows[[as.character(canonical_seed)]] <-
    canonical_selection

  for (seed_index in seq_along(
    additional_seeds
  )) {
    seed <- additional_seeds[[seed_index]]
    seed_name <- as.character(seed)
    cat(sprintf(
      paste0(
        "\n######## repeated CV seed %d ",
        "(%d/5) ########\n"
      ),
      seed, seed_index
    ))
    repeat_v14 <-
      v14_registry$artifacts[[seed_name]]
    repeat_sequence <- run_sequence_oof(
      train,
      repeat_v14$fold_map,
      m8trpg_helper,
      checkpoint_tag = paste0(
        "seed_", seed
      ),
      outer_seed = seed
    )
    repeat_blend <- crossfit_blend(
      truth,
      repeat_v14$prediction,
      repeat_sequence$prediction,
      repeat_sequence$row_fold
    )
    repeat_candidate_loss <-
      log_loss_matrix_local(
        truth,
        repeat_blend$prediction
      )
    repeat_case_gain <-
      audited$respondent_gain(
        truth,
        repeat_v14$prediction,
        repeat_blend$prediction,
        train$Case
      )
    case_gain_matrix[, seed_name] <-
      repeat_case_gain
    repeat_rows[[seed_name]] <- data.frame(
      seed = seed,
      baseline_logloss =
        repeat_v14$logloss,
      sequence_component_logloss =
        log_loss_matrix_local(
          truth,
          repeat_sequence$prediction
        ),
      candidate_logloss =
        repeat_candidate_loss,
      gain = repeat_v14$logloss -
        repeat_candidate_loss
    )
    selection_rows[[seed_name]] <-
      fold_selection_table(
        repeat_sequence$fold_results,
        repeat_blend$weights,
        seed
      )
    saveRDS(
      list(
        experiment_id = experiment_id,
        seed = seed,
        baseline_source =
          repeat_v14$path,
        fold_map = repeat_v14$fold_map,
        baseline_prediction =
          repeat_v14$prediction,
        sequence_prediction =
          repeat_sequence$prediction,
        candidate_prediction =
          repeat_blend$prediction,
        offset_prediction =
          repeat_sequence$offset_prediction,
        weights = repeat_blend$weights,
        fold_results =
          repeat_sequence$fold_results,
        respondent_gain = repeat_case_gain
      ),
      file.path(
        output_dir,
        sprintf(
          "repeat_result_%d.rds", seed
        )
      )
    )
  }

  repeated_by_seed <- do.call(
    rbind, repeat_rows
  )
  average_case_gain <- rowMeans(
    case_gain_matrix
  )
  pooled_bootstrap <-
    audited$bootstrap_case_means(
      average_case_gain,
      bootstrap_replicates,
      seed = canonical_seed
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

  propensity <- audited$fit_test_propensity(
    train, test
  )$propensity
  test_like <- audited$test_like_diagnostic(
    average_case_gain,
    propensity,
    bootstrap_replicates
  )
  top_30 <- test_like[
    test_like$group == "top_30pct",
    ,
    drop = FALSE
  ]
  stopifnot(nrow(top_30) == 1L)
  repeated_summary$test_like_30_gain <-
    top_30$point_gain
  repeated_summary$test_like_30_lower_95 <-
    top_30$lower_95
  repeated_summary$promote <-
    repeated_summary$point_gain > 0 &
    repeated_summary$lower_95 > 0 &
    repeated_summary$positive_repeats >= 5L &
    repeated_summary$test_like_30_gain >= 0 &
    repeated_summary$
      test_like_30_lower_95 >= -0.001

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
      pooled_bootstrap = pooled_bootstrap,
      test_like = test_like
    ),
    file.path(
      output_dir, "repeated_cv_result.rds"
    )
  )

  if (isTRUE(repeated_summary$promote)) {
    verdict <- paste0(
      "PROMOTE respondent-sequence Transformer ",
      "for a separate full-data build audit. ",
      "Do not submit until that artifact is ",
      "reproduced and checked."
    )
  } else {
    verdict <- paste0(
      "REJECT respondent-sequence Transformer; ",
      "retain exact set-context v14."
    )
  }
  writeLines(
    verdict,
    file.path(output_dir, "verdict.txt")
  )
  cat("\nRepeated-CV result:\n")
  print(repeated_summary, digits = 9)
  cat("\nTest-like diagnostic:\n")
  print(test_like, digits = 9)
  cat("\n", verdict, "\n", sep = "")
  runtime_state$final_status <- "completed"
  invisible(list(
    canonical = canonical_summary,
    repeated = repeated_summary,
    test_like = test_like,
    verdict = verdict
  ))
}

run_experiment()
