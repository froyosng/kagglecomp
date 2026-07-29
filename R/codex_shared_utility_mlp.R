# Shared-alternative-utility neural network: a genuinely new combination not
# yet tried in this project. Earlier rounds tested (a) a shared-utility exact
# 4-way-softmax model with a TREE function class (custom xgboost objective,
# codex_shared_utility_screen.R) -- failed cold-start because m8trpg's
# hand-built interactions were doing the real work, not the loss function --
# and (b) a non-shared (flat, alternative-position-specific) neural net (the
# already-adopted shallow/deep MLP components). This script is the missing
# cell: the SAME shared-weight function, applied identically to every
# alternative's own feature row (including the opt-out's fixed all-zero
# profile), trained on the exact 4-way cross-entropy via a neural network
# instead of trees -- a function class that has already shown real (if
# modest) ensemble-diversity value in this project.
#
# Feature treatment mirrors m8trpg's own established choices as closely as
# possible for a fair comparison, not an arbitrary new encoding:
#   - all 19 attribute codes and Price: one-hot (m8trpg's own price-as-factor
#     finding: non-linearity in levels matters).
#   - segment/region/ppark: one-hot (m8trpg treats these as categorical
#     dummies too).
#   - income/age/miles/night/gender/urbanicity/education: standardized
#     continuous (m8trpg's own shift_scaler_vars treatment, not one-hot --
#     avoids the sparse high-cardinality income/miles/night dummy problem
#     this project already found harmful for binned covariates).
#   - price_gap_min/max, is_cheapest/is_dearest, Task_c, d2/d3: the same
#     choice-set-context engineering already confirmed to carry signal.
#
# Alternative position (d2/d3) is included as an ordinary input feature (not
# dropped for "true" exchangeability) because m8trpg itself found a small
# real position effect; the shared FUNCTION is still applied identically to
# every row regardless of position, it just receives position as one input.
#
# Stages:
#   CODEX_SHARED_MLP_STAGE=define - load reusable functions only
#   CODEX_SHARED_MLP_STAGE=screen - single-split (seed 7402) architecture screen

options(stringsAsFactors = FALSE)
suppressPackageStartupMessages(library(torch))
source("R/codex_modeling_common.R")

stage <- Sys.getenv("CODEX_SHARED_MLP_STAGE", "screen")
stopifnot(stage %in% c("define", "screen"))

output_dir <- "data_processed/codex_shared_utility_mlp"
checkpoint_dir <- file.path(output_dir, "checkpoints")
dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)

torch_set_num_threads(4L)

## ---- Feature construction ----

context_columns <- function(df) {
  df <- sort_long_tasks(df)
  df$inside <- as.integer(df$alt != 4L)
  df$d2 <- as.integer(df$alt == 2L)
  df$d3 <- as.integer(df$alt == 3L)
  df$Task_c <- (as.numeric(df$Task) - 10) / 9
  price_num <- as.numeric(df$Price)
  price_mat <- matrix(price_num, ncol = 4, byrow = TRUE)
  inside_price <- price_mat[, 1:3, drop = FALSE]
  pmin3 <- apply(inside_price, 1, min)
  pmax3 <- apply(inside_price, 1, max)
  df$price_min <- rep(pmin3, each = 4)
  df$price_max <- rep(pmax3, each = 4)
  df$is_cheapest <- as.integer(df$inside == 1L & price_num == df$price_min)
  df$is_dearest <- as.integer(df$inside == 1L & price_num == df$price_max)
  df$price_gap_min <- ifelse(df$inside == 1L, price_num - df$price_min, 0)
  df$price_gap_max <- ifelse(df$inside == 1L, df$price_max - price_num, 0)
  df
}

shared_utility_cont_vars <- c(
  "incomea", "agea", "milesa", "nighta",
  "genderind", "Urbind", "educind",
  "price_gap_min", "price_gap_max"
)

shared_utility_scaler <- function(long_df) {
  long_df <- context_columns(long_df)
  ctr <- vapply(
    long_df[, shared_utility_cont_vars, drop = FALSE], mean, numeric(1)
  )
  scl <- vapply(
    long_df[, shared_utility_cont_vars, drop = FALSE], sd, numeric(1)
  )
  scl[scl == 0] <- 1
  list(ctr = ctr, scl = scl)
}

compute_attr_levels <- function(full_long) {
  levels_list <- lapply(attrs, function(a) {
    0:max(as.integer(full_long[[a]]))
  })
  names(levels_list) <- attrs
  levels_list
}

one_hot_matrix <- function(values, levels) {
  values <- as.integer(values)
  n <- length(values)
  m <- matrix(0, n, length(levels))
  colnames(m) <- as.character(levels)
  idx <- match(values, levels)
  stopifnot(!anyNA(idx))
  m[cbind(seq_len(n), idx)] <- 1
  m
}

build_shared_utility_features <- function(long_df, attr_levels, scaler) {
  long_df <- context_columns(long_df)
  parts <- list()
  for (a in attrs) {
    parts[[paste0("attr_", a)]] <- one_hot_matrix(
      long_df[[a]], attr_levels[[a]]
    )
  }
  parts[["price"]] <- one_hot_matrix(long_df$Price, 0:12)
  parts[["segment"]] <- one_hot_matrix(long_df$segmentind, 1:6)
  parts[["region"]] <- one_hot_matrix(long_df$regionind, 1:5)
  parts[["ppark"]] <- one_hot_matrix(long_df$pparkind, 1:5)

  cont_raw <- as.matrix(long_df[, shared_utility_cont_vars, drop = FALSE])
  storage.mode(cont_raw) <- "double"
  cont_z <- sweep(sweep(cont_raw, 2, scaler$ctr, "-"), 2, scaler$scl, "/")
  colnames(cont_z) <- shared_utility_cont_vars
  parts[["cont"]] <- cont_z

  ctx <- as.matrix(
    long_df[, c("d2", "d3", "Task_c", "is_cheapest", "is_dearest"),
            drop = FALSE]
  )
  storage.mode(ctx) <- "double"
  parts[["ctx"]] <- ctx
  parts[["inside"]] <- matrix(
    as.double(long_df$inside), ncol = 1,
    dimnames = list(NULL, "inside")
  )

  x <- do.call(cbind, parts)
  storage.mode(x) <- "double"
  stopifnot(nrow(x) %% 4L == 0L)
  list(x = x, no = unique(long_df$No), truth = long_truth_matrix(long_df))
}

## ---- Model ----

shared_utility_net <- nn_module(
  "shared_utility_net",
  initialize = function(input_dim, hidden, dropout) {
    layers <- list()
    previous <- as.integer(input_dim)
    for (width in hidden) {
      layers[[length(layers) + 1L]] <-
        nn_linear(previous, as.integer(width))
      layers[[length(layers) + 1L]] <- nn_relu()
      layers[[length(layers) + 1L]] <- nn_dropout(p = dropout)
      previous <- as.integer(width)
    }
    layers[[length(layers) + 1L]] <- nn_linear(previous, 1L)
    self$network <- do.call(nn_sequential, layers)
  },
  forward = function(x) {
    self$network(x)
  }
)

predict_shared_utility <- function(model, x, batch_size = 8192L) {
  model$eval()
  n <- nrow(x)
  stopifnot(n %% 4L == 0L)
  utility <- numeric(n)
  with_no_grad({
    for (start in seq.int(1L, n, by = batch_size)) {
      stop_at <- min(n, start + batch_size - 1L)
      xb <- torch_tensor(x[start:stop_at, , drop = FALSE], dtype = torch_float())
      utility[start:stop_at] <- as.numeric(model(xb)$squeeze(2))
    }
  })
  softmax_margins(utility)
}

fit_shared_utility_once <- function(x_train, truth_train, x_valid, config,
                                    seed, checkpoint = NULL) {
  # Checkpoints store only the plain-R prediction matrix and scalar metrics,
  # never a torch model/state_dict -- torch external pointers are not valid
  # across process restarts, so an RDS'd state_dict loaded in a fresh
  # Rscript invocation crashes ("external pointer is not valid"). Matching
  # R/codex_torch_deep_mlp.R's established fit_torch_once pattern exactly.
  if (!is.null(checkpoint) && file.exists(checkpoint)) {
    return(readRDS(checkpoint))
  }
  set.seed(seed)
  torch_manual_seed(seed)

  n_tasks <- nrow(x_train) %/% 4L
  stopifnot(nrow(x_train) == n_tasks * 4L, nrow(truth_train) == n_tasks)
  y_task <- as.integer(max.col(truth_train))
  x_tensor <- torch_tensor(x_train, dtype = torch_float())
  y_tensor <- torch_tensor(y_task, dtype = torch_long())

  model <- shared_utility_net(
    input_dim = ncol(x_train), hidden = config$hidden,
    dropout = config$dropout
  )
  optimizer <- optim_adam(
    model$parameters, lr = config$learning_rate,
    weight_decay = config$weight_decay
  )

  trace_rows <- list()
  started <- proc.time()[["elapsed"]]
  for (epoch in seq_len(config$epochs)) {
    model$train()
    task_order <- sample.int(n_tasks)
    total_loss <- 0
    total_tasks <- 0L
    for (start in seq.int(1L, n_tasks, by = config$batch_tasks)) {
      stop_at <- min(n_tasks, start + config$batch_tasks - 1L)
      task_idx <- task_order[start:stop_at]
      row_idx <- as.vector(rbind(
        4L * task_idx - 3L, 4L * task_idx - 2L,
        4L * task_idx - 1L, 4L * task_idx
      ))
      optimizer$zero_grad()
      utility <- model(x_tensor[row_idx, ])$view(c(-1L, 4L))
      loss <- nnf_cross_entropy(utility, y_tensor[task_idx])
      loss$backward()
      optimizer$step()
      n_batch <- length(task_idx)
      total_loss <- total_loss + loss$item() * n_batch
      total_tasks <- total_tasks + n_batch
    }
    if (epoch == 1L || epoch %% 10L == 0L || epoch == config$epochs) {
      trace_rows[[length(trace_rows) + 1L]] <- data.frame(
        epoch = epoch, training_logloss = total_loss / total_tasks
      )
    }
  }
  elapsed <- proc.time()[["elapsed"]] - started
  prediction <- predict_shared_utility(model, x_valid)
  stopifnot(
    nrow(x_valid) %% 4L == 0L,
    identical(dim(prediction), c(nrow(x_valid) %/% 4L, 4L)),
    !anyNA(prediction), all(prediction > 0),
    max(abs(rowSums(prediction) - 1)) < 1e-6
  )
  result <- list(
    prediction = prediction,
    seed = seed,
    trace = do.call(rbind, trace_rows),
    elapsed_seconds = elapsed,
    final_training_logloss = total_loss / total_tasks
  )
  if (!is.null(checkpoint)) saveRDS(result, checkpoint)
  result
}

fit_shared_utility_average <- function(x_train, truth_train, x_valid,
                                       config, seeds,
                                       checkpoint_prefix = NULL) {
  prediction_array <- array(
    NA_real_, dim = c(nrow(x_valid) %/% 4L, 4L, length(seeds))
  )
  fit_rows <- list()
  for (i in seq_along(seeds)) {
    seed <- as.integer(seeds[[i]])
    checkpoint <- if (is.null(checkpoint_prefix)) {
      NULL
    } else {
      sprintf("%s_seed_%d.rds", checkpoint_prefix, seed)
    }
    fitted <- fit_shared_utility_once(
      x_train, truth_train, x_valid, config, seed, checkpoint = checkpoint
    )
    prediction_array[, , i] <- fitted$prediction
    fit_rows[[i]] <- data.frame(
      seed = seed, elapsed_seconds = fitted$elapsed_seconds,
      final_training_logloss = fitted$final_training_logloss
    )
    cat(sprintf(
      "  seed %d: train %.6f, %.1fs\n",
      seed, fitted$final_training_logloss, fitted$elapsed_seconds
    ))
    flush.console()
  }
  list(
    prediction = apply(prediction_array, c(1L, 2L), mean),
    fits = do.call(rbind, fit_rows)
  )
}

shared_utility_configs <- list(
  shared_64_32 = list(
    hidden = c(64L, 32L), dropout = 0.10, weight_decay = 1e-4,
    learning_rate = 1e-3, epochs = 60L, batch_tasks = 256L
  ),
  shared_128_64 = list(
    hidden = c(128L, 64L), dropout = 0.20, weight_decay = 1e-4,
    learning_rate = 1e-3, epochs = 60L, batch_tasks = 256L
  ),
  shared_32 = list(
    hidden = 32L, dropout = 0.10, weight_decay = 1e-4,
    learning_rate = 1e-3, epochs = 60L, batch_tasks = 256L
  )
)

if (stage == "screen") {
  split <- readRDS("data_processed/train_val_split.rds")
  full_train <- read.csv("csv files/train.csv")
  full_long_pattern <- paste0(
    "^(", paste(c(attrs, "Price", "Ch"), collapse = "|"), ")([1-4])$"
  )
  fixed_names <- names(full_train)[!grepl(full_long_pattern, names(full_train))]
  full_long_parts <- lapply(1:4, function(a) {
    out <- full_train[, fixed_names, drop = FALSE]
    for (v in c(attrs, "Price", "Ch")) out[[v]] <- full_train[[paste0(v, a)]]
    out$alt <- a
    out$chosen <- as.integer(out$Ch == 1)
    out
  })
  full_long <- do.call(rbind, full_long_parts)
  attr_levels <- compute_attr_levels(full_long)

  tr_long <- split$train_long_tr
  va_long <- split$train_long_val
  scaler <- shared_utility_scaler(tr_long)
  train_features <- build_shared_utility_features(tr_long, attr_levels, scaler)
  valid_features <- build_shared_utility_features(va_long, attr_levels, scaler)
  cat(sprintf(
    "feature dim = %d, train tasks = %d, valid tasks = %d\n",
    ncol(train_features$x), nrow(train_features$x) %/% 4L,
    nrow(valid_features$x) %/% 4L
  ))

  validation_order <- order(split$train_wide_val$No)
  truth <- as.matrix(
    split$train_wide_val[validation_order, paste0("Ch", 1:4)]
  )
  stopifnot(identical(
    as.integer(valid_features$no), as.integer(split$train_wide_val$No[validation_order])
  ))

  saved_screen <- readRDS(
    "data_processed/codex_behavioral_round/mlp_screen.rds"
  )
  stopifnot(identical(
    as.integer(valid_features$no), as.integer(saved_screen$validation_no)
  ))
  v11 <- saved_screen$v11_screen
  shallow <- saved_screen$predictions[["h08_d0.100"]]
  current <- 0.85 * v11 + 0.15 * shallow
  stopifnot(
    abs(log_loss_matrix(truth, v11) - 1.160568) < 1e-5,
    abs(log_loss_matrix(truth, current) - 1.160412) < 1e-5
  )

  seeds <- c(9401L, 9402L, 9403L)
  weight_grid <- seq(0, 0.40, by = 0.01)
  result_rows <- list()
  prediction_list <- list()
  fit_rows <- list()

  for (config_name in names(shared_utility_configs)) {
    cat(sprintf("=== config %s ===\n", config_name))
    flush.console()
    config <- shared_utility_configs[[config_name]]
    prefix <- file.path(checkpoint_dir, paste0("screen_", config_name))
    fitted <- fit_shared_utility_average(
      train_features$x, train_features$truth, valid_features$x,
      config, seeds, checkpoint_prefix = prefix
    )
    prediction <- fitted$prediction
    component_loss <- log_loss_matrix(truth, prediction)

    replacement_curve <- vapply(weight_grid, function(w) {
      log_loss_matrix(truth, (1 - w) * v11 + w * prediction)
    }, numeric(1))
    incremental_curve <- vapply(weight_grid, function(w) {
      log_loss_matrix(truth, (1 - w) * current + w * prediction)
    }, numeric(1))
    replacement_best <- which.min(replacement_curve)
    incremental_best <- which.min(incremental_curve)

    result_rows[[config_name]] <- data.frame(
      config = config_name,
      hidden = paste(config$hidden, collapse = "-"),
      dropout = config$dropout,
      component_logloss = component_loss,
      v11_logloss = log_loss_matrix(truth, v11),
      current_logloss = log_loss_matrix(truth, current),
      replacement_weight = weight_grid[[replacement_best]],
      replacement_logloss = replacement_curve[[replacement_best]],
      replacement_gain_vs_v11 =
        log_loss_matrix(truth, v11) - replacement_curve[[replacement_best]],
      incremental_weight = weight_grid[[incremental_best]],
      incremental_logloss = incremental_curve[[incremental_best]],
      incremental_gain_vs_current =
        log_loss_matrix(truth, current) - incremental_curve[[incremental_best]]
    )
    prediction_list[[config_name]] <- prediction
    fit_rows[[config_name]] <- cbind(
      data.frame(config = config_name), fitted$fits
    )
    cat(sprintf(
      "%s: component %.6f; replacement %.6f (w=%.2f); incremental %.6f (w=%.2f)\n",
      config_name, component_loss,
      replacement_curve[[replacement_best]], weight_grid[[replacement_best]],
      incremental_curve[[incremental_best]], weight_grid[[incremental_best]]
    ))
    flush.console()
  }

  result <- do.call(rbind, result_rows)
  write.csv(
    result, file.path(output_dir, "shared_utility_mlp_screen.csv"),
    row.names = FALSE
  )
  write.csv(
    do.call(rbind, fit_rows),
    file.path(output_dir, "shared_utility_mlp_screen_fits.csv"),
    row.names = FALSE
  )
  saveRDS(
    list(
      result = result, predictions = prediction_list,
      validation_no = valid_features$no, truth = truth,
      v11 = v11, current = current,
      feature_dim = ncol(train_features$x)
    ),
    file.path(output_dir, "shared_utility_mlp_screen.rds")
  )
  cat("\n=== rank:ndcg xgboost screen reference: 1.193073 (component alone) ===\n")
  print(result, digits = 9)
}
