# Deep feed-forward choice model using R torch.
#
# This is a targeted extension of R/codex_mlp_ensemble.R. It deliberately
# reuses that script's exact feature matrix and changes only the learner:
# two/three hidden layers with dropout and Adam weight decay. Architecture
# screening is confined to the canonical seed-7402 split. The chosen
# architecture is then frozen before respondent-grouped five-fold CV.
#
# Stages:
#   CODEX_DEEP_STAGE=define - load reusable functions without running a stage
#   CODEX_DEEP_STAGE=smoke  - two-epoch API/runtime check
#   CODEX_DEEP_STAGE=screen - architecture single-split screen
#   CODEX_DEEP_STAGE=cv     - canonical five-fold OOF confirmation

options(stringsAsFactors = FALSE)

suppressPackageStartupMessages(library(torch))

old_stage <- Sys.getenv("CODEX_STAGE", unset = NA_character_)
Sys.setenv(CODEX_STAGE = "define")
source("R/codex_mlp_ensemble.R")
if (is.na(old_stage)) {
  Sys.unsetenv("CODEX_STAGE")
} else {
  Sys.setenv(CODEX_STAGE = old_stage)
}

stage <- Sys.getenv("CODEX_DEEP_STAGE", "screen")
stopifnot(stage %in% c("define", "smoke", "screen", "cv"))

output_dir <- "data_processed/codex_deep_stack"
checkpoint_dir <- file.path(output_dir, "torch_checkpoints")
dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)

torch_set_num_threads(4L)

deep_configs <- list(
  d64_32_do10 = list(
    hidden = c(64L, 32L),
    dropout = 0.10,
    weight_decay = 1e-4,
    learning_rate = 1e-3,
    epochs = 60L,
    batch_size = 256L
  ),
  d128_64_do20 = list(
    hidden = c(128L, 64L),
    dropout = 0.20,
    weight_decay = 1e-4,
    learning_rate = 1e-3,
    epochs = 60L,
    batch_size = 256L
  ),
  d128_64_32_do20 = list(
    hidden = c(128L, 64L, 32L),
    dropout = 0.20,
    weight_decay = 5e-4,
    learning_rate = 1e-3,
    epochs = 60L,
    batch_size = 256L
  ),
  d64_32_do20_e15 = list(
    hidden = c(64L, 32L),
    dropout = 0.20,
    weight_decay = 5e-4,
    learning_rate = 1e-3,
    epochs = 15L,
    batch_size = 256L
  ),
  d128_64_do30_e12 = list(
    hidden = c(128L, 64L),
    dropout = 0.30,
    weight_decay = 1e-3,
    learning_rate = 1e-3,
    epochs = 12L,
    batch_size = 256L
  )
)

dense_choice_net <- nn_module(
  "dense_choice_net",
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
    layers[[length(layers) + 1L]] <- nn_linear(previous, 4L)
    self$network <- do.call(nn_sequential, layers)
  },
  forward = function(x) {
    self$network(x)
  }
)

predict_torch <- function(model, x, batch_size = 2048L) {
  model$eval()
  prediction <- matrix(NA_real_, nrow(x), 4L)
  with_no_grad({
    for (start in seq.int(1L, nrow(x), by = batch_size)) {
      stop_at <- min(nrow(x), start + batch_size - 1L)
      xb <- torch_tensor(
        x[start:stop_at, , drop = FALSE],
        dtype = torch_float()
      )
      probability <- nnf_softmax(model(xb), dim = 2L)
      prediction[start:stop_at, ] <- as_array(probability)
    }
  })
  prediction <- pmax(prediction, 1e-15)
  prediction / rowSums(prediction)
}

fit_torch_once <- function(x_train, y_train, x_valid, config, seed,
                           epochs_override = NULL) {
  set.seed(seed)
  torch_manual_seed(seed)

  epochs <- if (is.null(epochs_override)) {
    config$epochs
  } else {
    as.integer(epochs_override)
  }
  model <- dense_choice_net(
    input_dim = ncol(x_train),
    hidden = config$hidden,
    dropout = config$dropout
  )
  optimizer <- optim_adam(
    model$parameters,
    lr = config$learning_rate,
    weight_decay = config$weight_decay
  )
  x_tensor <- torch_tensor(x_train, dtype = torch_float())
  y_tensor <- torch_tensor(
    as.integer(max.col(y_train)),
    dtype = torch_long()
  )

  trace_rows <- list()
  started <- proc.time()[["elapsed"]]
  n <- nrow(x_train)
  for (epoch in seq_len(epochs)) {
    model$train()
    row_order <- sample.int(n)
    total_loss <- 0
    total_rows <- 0L
    for (start in seq.int(1L, n, by = config$batch_size)) {
      stop_at <- min(n, start + config$batch_size - 1L)
      index <- row_order[start:stop_at]
      optimizer$zero_grad()
      logits <- model(x_tensor[index, ])
      loss <- nnf_cross_entropy(logits, y_tensor[index])
      loss$backward()
      optimizer$step()
      batch_n <- length(index)
      total_loss <- total_loss + loss$item() * batch_n
      total_rows <- total_rows + batch_n
    }
    if (epoch == 1L || epoch %% 10L == 0L || epoch == epochs) {
      trace_rows[[length(trace_rows) + 1L]] <- data.frame(
        epoch = epoch,
        training_logloss = total_loss / total_rows
      )
    }
  }
  elapsed <- proc.time()[["elapsed"]] - started
  prediction <- predict_torch(model, x_valid)
  list(
    prediction = prediction,
    trace = do.call(rbind, trace_rows),
    elapsed_seconds = elapsed,
    final_training_logloss = total_loss / total_rows
  )
}

fit_torch_average <- function(x_train, y_train, x_valid, config,
                              seeds, checkpoint_prefix = NULL,
                              epochs_override = NULL) {
  prediction_array <- array(
    NA_real_,
    dim = c(nrow(x_valid), 4L, length(seeds))
  )
  fit_rows <- list()
  traces <- list()

  for (seed_index in seq_along(seeds)) {
    seed <- as.integer(seeds[[seed_index]])
    checkpoint <- if (is.null(checkpoint_prefix)) {
      NULL
    } else {
      sprintf("%s_seed_%d.rds", checkpoint_prefix, seed)
    }
    if (!is.null(checkpoint) && file.exists(checkpoint)) {
      fitted <- readRDS(checkpoint)
      source_status <- "checkpoint"
    } else {
      fitted <- fit_torch_once(
        x_train, y_train, x_valid, config, seed,
        epochs_override = epochs_override
      )
      if (!is.null(checkpoint)) saveRDS(fitted, checkpoint)
      source_status <- "fitted"
    }
    stopifnot(
      identical(dim(fitted$prediction), c(nrow(x_valid), 4L)),
      !anyNA(fitted$prediction),
      all(fitted$prediction > 0),
      max(abs(rowSums(fitted$prediction) - 1)) < 1e-6
    )
    prediction_array[, , seed_index] <- fitted$prediction
    fit_rows[[seed_index]] <- data.frame(
      seed = seed,
      elapsed_seconds = fitted$elapsed_seconds,
      final_training_logloss = fitted$final_training_logloss,
      source_status = source_status
    )
    traces[[as.character(seed)]] <- fitted$trace
    cat(sprintf(
      "torch seed %d complete [%s]: train %.6f, %.1fs\n",
      seed, source_status, fitted$final_training_logloss,
      fitted$elapsed_seconds
    ))
    flush.console()
  }
  list(
    prediction = apply(prediction_array, c(1L, 2L), mean),
    fits = do.call(rbind, fit_rows),
    traces = traces
  )
}

crossfit_two_way <- function(truth, baseline, component, row_fold,
                             weight_grid) {
  prediction <- matrix(NA_real_, nrow(truth), 4L)
  weights <- numeric(5L)
  for (fold in 1:5) {
    fit_rows <- row_fold != fold
    validation_rows <- row_fold == fold
    losses <- vapply(weight_grid, function(weight) {
      log_loss_matrix(
        truth[fit_rows, , drop = FALSE],
        (1 - weight) * baseline[fit_rows, , drop = FALSE] +
          weight * component[fit_rows, , drop = FALSE]
      )
    }, numeric(1))
    weights[[fold]] <- weight_grid[[which.min(losses)]]
    prediction[validation_rows, ] <-
      (1 - weights[[fold]]) *
        baseline[validation_rows, , drop = FALSE] +
      weights[[fold]] *
        component[validation_rows, , drop = FALSE]
  }
  stopifnot(!anyNA(prediction))
  list(
    prediction = prediction,
    weights = weights,
    logloss = log_loss_matrix(truth, prediction)
  )
}

softmax_weights <- function(theta, n_components) {
  value <- c(theta, 0)
  value <- value - max(value)
  exp(value) / sum(exp(value))
}

blend_components <- function(components, weights) {
  Reduce(
    `+`,
    Map(function(prediction, weight) prediction * weight,
        components, weights)
  )
}

fit_arithmetic_weights <- function(truth, components, rows) {
  n_components <- length(components)
  chosen <- do.call(cbind, lapply(components, function(prediction) {
    rowSums(
      truth[rows, , drop = FALSE] *
        prediction[rows, , drop = FALSE]
    )
  }))
  evaluate <- function(theta) {
    weights <- softmax_weights(theta, n_components)
    mixed <- as.numeric(chosen %*% weights)
    value <- -mean(log(pmax(mixed, 1e-15)))
    gradient_weight <- -colMeans(chosen / mixed)
    weighted_gradient <- sum(weights * gradient_weight)
    gradient <- weights[seq_len(n_components - 1L)] *
      (
        gradient_weight[seq_len(n_components - 1L)] -
          weighted_gradient
      )
    list(value = value, gradient = gradient)
  }
  fitted <- optim(
    rep(0, n_components - 1L),
    function(theta) evaluate(theta)$value,
    gr = function(theta) evaluate(theta)$gradient,
    method = "BFGS",
    control = list(maxit = 1000, reltol = 1e-11)
  )
  stopifnot(fitted$convergence == 0L)
  softmax_weights(fitted$par, n_components)
}

crossfit_three_way <- function(truth, v11, shallow, deep, row_fold) {
  components <- list(v11 = v11, shallow = shallow, deep = deep)
  prediction <- matrix(NA_real_, nrow(truth), 4L)
  weights <- matrix(
    NA_real_, 5L, 3L,
    dimnames = list(paste0("fold", 1:5), names(components))
  )
  for (fold in 1:5) {
    fit_rows <- which(row_fold != fold)
    validation_rows <- row_fold == fold
    weights[fold, ] <- fit_arithmetic_weights(
      truth, components, fit_rows
    )
    prediction[validation_rows, ] <- blend_components(
      lapply(
        components,
        function(x) x[validation_rows, , drop = FALSE]
      ),
      weights[fold, ]
    )
  }
  list(
    prediction = prediction,
    weights = weights,
    logloss = log_loss_matrix(truth, prediction)
  )
}

bootstrap_comparisons <- function(truth, case, predictions,
                                  n_boot = 100000L,
                                  family_size = 3L) {
  row_loss <- function(prediction) {
    -rowSums(
      truth * log(pmax(prediction / rowSums(prediction), 1e-15))
    )
  }
  comparisons <- list(
    replacement_vs_v11 =
      row_loss(predictions$v11) -
      row_loss(predictions$replacement),
    incremental_vs_current =
      row_loss(predictions$current) -
      row_loss(predictions$incremental),
    joint_vs_current =
      row_loss(predictions$current) -
      row_loss(predictions$joint)
  )
  case_gain <- do.call(cbind, lapply(comparisons, function(value) {
    unname(tapply(value, case, mean))
  }))
  set.seed(4821L)
  n_case <- nrow(case_gain)
  boot <- matrix(
    NA_real_, n_boot, ncol(case_gain),
    dimnames = list(NULL, colnames(case_gain))
  )
  for (start in seq.int(1L, n_boot, by = 1000L)) {
    stop_at <- min(n_boot, start + 999L)
    n_this <- stop_at - start + 1L
    sampled <- matrix(
      sample.int(n_case, n_case * n_this, replace = TRUE),
      nrow = n_case
    )
    for (comparison in seq_len(ncol(case_gain))) {
      boot[start:stop_at, comparison] <- colMeans(matrix(
        case_gain[sampled, comparison],
        nrow = n_case
      ))
    }
  }
  family_alpha <- 0.05 / family_size
  summary <- do.call(rbind, lapply(seq_len(ncol(boot)), function(index) {
    value <- boot[, index]
    data.frame(
      comparison = colnames(boot)[[index]],
      point_gain = mean(case_gain[, index]),
      bootstrap_sd = sd(value),
      lower_95 = unname(quantile(value, 0.025)),
      upper_95 = unname(quantile(value, 0.975)),
      lower_99 = unname(quantile(value, 0.005)),
      upper_99 = unname(quantile(value, 0.995)),
      lower_bonferroni = unname(
        quantile(value, family_alpha / 2)
      ),
      upper_bonferroni = unname(
        quantile(value, 1 - family_alpha / 2)
      ),
      win_rate = mean(value > 0),
      n_boot = n_boot,
      family_size = family_size
    )
  }))
  list(summary = summary, bootstrap = boot, case_gain = case_gain)
}

if (stage %in% c("smoke", "screen")) {
  split_data <- readRDS("data_processed/train_val_split.rds")
  full_train <- read.csv("csv files/train.csv")
  reference <- feature_reference(full_train)
  scaler <- continuous_scaler(split_data$train_wide_tr)
  train_matrix <- build_mlp_matrix(
    split_data$train_wide_tr, reference, scaler
  )
  validation_matrix <- build_mlp_matrix(
    split_data$train_wide_val, reference, scaler,
    keep_columns = train_matrix$keep_columns
  )
  train_order <- order(split_data$train_wide_tr$No)
  validation_order <- order(split_data$train_wide_val$No)
  x_train <- train_matrix$x[train_order, , drop = FALSE]
  x_valid <- validation_matrix$x[
    validation_order, , drop = FALSE
  ]
  y_train <- as.matrix(
    split_data$train_wide_tr[
      train_order, paste0("Ch", 1:4)
    ]
  )
  truth <- as.matrix(
    split_data$train_wide_val[
      validation_order, paste0("Ch", 1:4)
    ]
  )
  saved_screen <- readRDS(
    "data_processed/codex_behavioral_round/mlp_screen.rds"
  )
  validation_no <- split_data$train_wide_val$No[validation_order]
  stopifnot(
    identical(validation_no, saved_screen$validation_no),
    identical(dim(saved_screen$v11_screen), dim(truth))
  )
  v11 <- saved_screen$v11_screen
  shallow <- saved_screen$predictions[["h08_d0.100"]]
  current <- 0.85 * v11 + 0.15 * shallow

  if (stage == "smoke") {
    fitted <- fit_torch_once(
      x_train[seq_len(min(2048L, nrow(x_train))), , drop = FALSE],
      y_train[seq_len(min(2048L, nrow(y_train))), , drop = FALSE],
      x_valid[seq_len(min(512L, nrow(x_valid))), , drop = FALSE],
      deep_configs[[1]],
      seed = 9101L,
      epochs_override = 2L
    )
    stopifnot(!anyNA(fitted$prediction))
    print(fitted$trace)
    cat("torch smoke test passed\n")
  } else {
    seeds <- c(9101L, 9102L)
    weight_grid <- seq(0, 0.40, by = 0.01)
    result_rows <- list()
    prediction_list <- list()
    fit_rows <- list()
    trace_list <- list()
    for (config_name in names(deep_configs)) {
      config <- deep_configs[[config_name]]
      prefix <- file.path(
        checkpoint_dir,
        paste0("screen_", config_name)
      )
      fitted <- fit_torch_average(
        x_train, y_train, x_valid, config, seeds,
        checkpoint_prefix = prefix
      )
      prediction <- fitted$prediction
      replacement_curve <- vapply(weight_grid, function(weight) {
        log_loss_matrix(
          truth, (1 - weight) * v11 + weight * prediction
        )
      }, numeric(1))
      incremental_curve <- vapply(weight_grid, function(weight) {
        log_loss_matrix(
          truth, (1 - weight) * current + weight * prediction
        )
      }, numeric(1))
      replacement_best <- which.min(replacement_curve)
      incremental_best <- which.min(incremental_curve)
      result_rows[[config_name]] <- data.frame(
        config = config_name,
        hidden = paste(config$hidden, collapse = "-"),
        dropout = config$dropout,
        weight_decay = config$weight_decay,
        learning_rate = config$learning_rate,
        epochs = config$epochs,
        n_seeds = length(seeds),
        n_features = ncol(x_train),
        component_logloss = log_loss_matrix(truth, prediction),
        v11_logloss = log_loss_matrix(truth, v11),
        current_logloss = log_loss_matrix(truth, current),
        replacement_weight =
          weight_grid[[replacement_best]],
        replacement_logloss =
          replacement_curve[[replacement_best]],
        replacement_gain_vs_v11 =
          log_loss_matrix(truth, v11) -
          replacement_curve[[replacement_best]],
        incremental_weight =
          weight_grid[[incremental_best]],
        incremental_logloss =
          incremental_curve[[incremental_best]],
        incremental_gain_vs_current =
          log_loss_matrix(truth, current) -
          incremental_curve[[incremental_best]]
      )
      prediction_list[[config_name]] <- prediction
      fit_rows[[config_name]] <- cbind(
        data.frame(config = config_name),
        fitted$fits
      )
      trace_list[[config_name]] <- fitted$traces
      cat(sprintf(
        "%s: component %.6f; replacement %.6f; incremental %.6f\n",
        config_name,
        log_loss_matrix(truth, prediction),
        replacement_curve[[replacement_best]],
        incremental_curve[[incremental_best]]
      ))
      flush.console()
    }
    result <- do.call(rbind, result_rows)
    write.csv(
      result,
      file.path(output_dir, "torch_deep_screen.csv"),
      row.names = FALSE
    )
    write.csv(
      do.call(rbind, fit_rows),
      file.path(output_dir, "torch_deep_screen_fits.csv"),
      row.names = FALSE
    )
    saveRDS(
      list(
        result = result,
        predictions = prediction_list,
        traces = trace_list,
        validation_no = validation_no,
        truth = truth,
        v11 = v11,
        current = current,
        feature_columns = train_matrix$keep_columns
      ),
      file.path(output_dir, "torch_deep_screen.rds")
    )
    print(result, digits = 9)
  }
}

if (stage == "cv") {
  screen <- read.csv(
    file.path(output_dir, "torch_deep_screen.csv")
  )
  requested <- Sys.getenv("CODEX_DEEP_CONFIG", "")
  if (nzchar(requested)) {
    config_name <- requested
  } else {
    viable <- screen[
      screen$incremental_gain_vs_current > 0,
      ,
      drop = FALSE
    ]
    if (nrow(viable) == 0L) {
      cat("No deep architecture improved the current screen model; CV skipped.\n")
      quit(save = "no", status = 0)
    }
    config_name <- viable$config[[
      which.min(viable$incremental_logloss)
    ]]
  }
  stopifnot(config_name %in% names(deep_configs))
  config <- deep_configs[[config_name]]

  train <- read.csv("csv files/train.csv")
  truth <- as.matrix(train[, paste0("Ch", 1:4)])
  reference <- feature_reference(train)
  saved_base <- readRDS("data_processed/oof_ensemble_v10.rds")
  fold_map <- saved_base$fold_of_case
  row_fold <- unname(fold_map[as.character(train$Case)])
  v11 <- 0.8 * saved_base$oof_mlogit + 0.2 * saved_base$oof_xgb
  shallow <- readRDS(
    "data_processed/codex_behavioral_round/mlp_oof.rds"
  )$oof[["h08_d0.100"]]
  current <- readRDS(
    "data_processed/codex_behavioral_round/mlp_precision.rds"
  )$crossfit_prediction
  stopifnot(
    abs(log_loss_matrix(truth, v11) -
      1.14509421298673) < 1e-8,
    abs(log_loss_matrix(truth, shallow) -
      1.19054334979533) < 1e-8,
    abs(log_loss_matrix(truth, current) -
      1.14378944178118) < 1e-8
  )

  deep_oof <- matrix(NA_real_, nrow(train), 4L)
  fit_rows <- list()
  seeds <- c(9201L, 9202L, 9203L)
  for (fold in 1:5) {
    validation_cases <- as.integer(
      names(fold_map)[fold_map == fold]
    )
    train_rows <- !(train$Case %in% validation_cases)
    validation_rows <- train$Case %in% validation_cases
    tr <- train[train_rows, , drop = FALSE]
    va <- train[validation_rows, , drop = FALSE]
    scaler <- continuous_scaler(tr)
    train_matrix <- build_mlp_matrix(tr, reference, scaler)
    validation_matrix <- build_mlp_matrix(
      va, reference, scaler,
      keep_columns = train_matrix$keep_columns
    )
    prefix <- file.path(
      checkpoint_dir,
      sprintf("cv_%s_fold_%d", config_name, fold)
    )
    fitted <- fit_torch_average(
      train_matrix$x,
      as.matrix(tr[, paste0("Ch", 1:4)]),
      validation_matrix$x,
      config,
      seeds = seeds + fold * 100L,
      checkpoint_prefix = prefix
    )
    index <- match(va$No, train$No)
    stopifnot(!anyNA(index))
    deep_oof[index, ] <- fitted$prediction
    fit_rows[[fold]] <- cbind(
      data.frame(config = config_name, fold = fold),
      fitted$fits
    )
    cat(sprintf(
      "deep CV fold %d: %.9f\n",
      fold,
      log_loss_matrix(truth[index, ], fitted$prediction)
    ))
    flush.console()
  }
  stopifnot(
    !anyNA(deep_oof),
    max(abs(rowSums(deep_oof) - 1)) < 1e-6
  )

  replacement <- crossfit_two_way(
    truth, v11, deep_oof, row_fold,
    seq(0, 0.50, by = 0.01)
  )
  incremental <- crossfit_two_way(
    truth, current, deep_oof, row_fold,
    seq(0, 0.40, by = 0.01)
  )
  joint <- crossfit_three_way(
    truth, v11, shallow, deep_oof, row_fold
  )
  predictions <- list(
    v11 = v11,
    current = current,
    replacement = replacement$prediction,
    incremental = incremental$prediction,
    joint = joint$prediction
  )
  bootstrap <- bootstrap_comparisons(
    truth, train$Case, predictions,
    n_boot = 100000L,
    family_size = length(deep_configs)
  )
  summary <- data.frame(
    config = config_name,
    component_logloss = log_loss_matrix(truth, deep_oof),
    v11_logloss = log_loss_matrix(truth, v11),
    current_logloss = log_loss_matrix(truth, current),
    replacement_logloss = replacement$logloss,
    incremental_logloss = incremental$logloss,
    joint_logloss = joint$logloss,
    replacement_gain_vs_v11 =
      log_loss_matrix(truth, v11) - replacement$logloss,
    incremental_gain_vs_current =
      log_loss_matrix(truth, current) - incremental$logloss,
    joint_gain_vs_current =
      log_loss_matrix(truth, current) - joint$logloss
  )
  weight_rows <- rbind(
    data.frame(
      blend = "replacement", fold = 1:5,
      component = "deep", weight = replacement$weights
    ),
    data.frame(
      blend = "incremental", fold = 1:5,
      component = "deep", weight = incremental$weights
    ),
    do.call(rbind, lapply(1:5, function(fold) {
      data.frame(
        blend = "joint",
        fold = fold,
        component = colnames(joint$weights),
        weight = as.numeric(joint$weights[fold, ])
      )
    }))
  )
  write.csv(
    summary,
    file.path(output_dir, "torch_deep_cv.csv"),
    row.names = FALSE
  )
  write.csv(
    do.call(rbind, fit_rows),
    file.path(output_dir, "torch_deep_cv_fits.csv"),
    row.names = FALSE
  )
  write.csv(
    weight_rows,
    file.path(output_dir, "torch_deep_cv_weights.csv"),
    row.names = FALSE
  )
  write.csv(
    bootstrap$summary,
    file.path(output_dir, "torch_deep_cv_bootstrap.csv"),
    row.names = FALSE
  )
  saveRDS(
    list(
      config = config_name,
      specification = config,
      deep_oof = deep_oof,
      predictions = predictions,
      replacement = replacement,
      incremental = incremental,
      joint = joint,
      summary = summary,
      bootstrap = bootstrap
    ),
    file.path(output_dir, "torch_deep_oof.rds")
  )
  print(summary, digits = 9)
  print(bootstrap$summary, digits = 9)
  print(weight_rows, digits = 7)
}
