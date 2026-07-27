## Non-tree ensemble member using the base-recommended nnet package. Neural
## backends with embedding layers (keras/tensorflow/torch) are unavailable, so
## categorical inputs are one-hot encoded for a small single-hidden-layer MLP.

suppressPackageStartupMessages({
  library(nnet)
  library(mlogit)
  library(dfidx)
  library(xgboost)
})
source("R/codex_shift_common.R")

stage <- Sys.getenv("CODEX_STAGE", "screen")
stopifnot(stage %in% c("define", "screen", "refine", "cv"))

output_dir <- "data_processed/codex_behavioral_round"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

mlp_configs <- expand.grid(
  size = c(4L, 8L, 12L),
  decay = c(0.001, 0.01, 0.1),
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)
mlp_configs$config <- sprintf(
  "h%02d_d%s",
  mlp_configs$size,
  format(mlp_configs$decay, scientific = FALSE)
)

respondent_factor_vars <- c(
  "segmentind", "yearind", "milesind", "nightind",
  "pparkind", "genderind", "ageind", "educind",
  "regionind", "Urbind", "incomeind"
)
respondent_continuous_vars <- c(
  "milesa", "nighta", "agea", "incomea"
)

prepare_mlp_long <- function(df) {
  df <- as.data.frame(df)
  if (!("chid" %in% names(df))) {
    df$chid <- paste(df$Case, df$Task, sep = "_")
  }
  if (!("d2" %in% names(df))) df$d2 <- as.integer(df$alt == 2L)
  if (!("d3" %in% names(df))) df$d3 <- as.integer(df$alt == 3L)
  df[order(df$No, df$alt), , drop = FALSE]
}

feature_reference <- function(full_wide) {
  list(
    attr_max = as.list(vapply(
      attrs,
      function(attribute) {
        max(full_wide[, paste0(attribute, 1:4)])
      },
      numeric(1)
    )),
    factor_levels = lapply(
      respondent_factor_vars,
      function(variable) sort(unique(full_wide[[variable]]))
    )
  )
}

continuous_scaler <- function(wide) {
  ctr <- vapply(
    wide[, respondent_continuous_vars, drop = FALSE],
    mean,
    numeric(1)
  )
  scl <- vapply(
    wide[, respondent_continuous_vars, drop = FALSE],
    sd,
    numeric(1)
  )
  scl[scl == 0] <- 1
  list(ctr = ctr, scl = scl)
}

build_mlp_matrix <- function(wide, reference, scaler,
                             keep_columns = NULL) {
  features <- list()

  # Dummy-code every non-reference attribute level separately by alternative.
  for (attribute in attrs) {
    for (alternative in 1:4) {
      values <- wide[[paste0(attribute, alternative)]]
      for (level in seq_len(reference$attr_max[[attribute]])) {
        features[[
          paste0(attribute, alternative, "_L", level)
        ]] <- as.numeric(values == level)
      }
    }
  }

  # Price is categorical in the strongest logit, so expose its levels to the
  # MLP while also adding normalized choice-set price context below.
  for (alternative in 1:4) {
    values <- wide[[paste0("Price", alternative)]]
    for (level in 1:12) {
      features[[
        paste0("Price", alternative, "_L", level)
      ]] <- as.numeric(values == level)
    }
  }

  for (index in seq_along(respondent_factor_vars)) {
    variable <- respondent_factor_vars[[index]]
    levels <- reference$factor_levels[[index]]
    if (length(levels) > 1L) {
      for (level in levels[-1]) {
        features[[
          paste0(variable, "_L", level)
        ]] <- as.numeric(wide[[variable]] == level)
      }
    }
  }

  for (variable in respondent_continuous_vars) {
    features[[paste0("z_", variable)]] <-
      (
        as.numeric(wide[[variable]]) - scaler$ctr[[variable]]
      ) / scaler$scl[[variable]]
  }
  features$Task_c <- (as.numeric(wide$Task) - 10) / 9

  price <- as.matrix(wide[, paste0("Price", 1:4)])
  inside_price <- price[, 1:3, drop = FALSE]
  price_min <- apply(inside_price, 1, min)
  price_max <- apply(inside_price, 1, max)
  features$task_price_mean <- rowMeans(inside_price) / 12
  features$task_price_range <- (price_max - price_min) / 11
  for (alternative in 1:3) {
    features[[paste0("is_cheapest_", alternative)]] <-
      as.numeric(price[, alternative] == price_min)
    features[[paste0("is_dearest_", alternative)]] <-
      as.numeric(price[, alternative] == price_max)
    features[[paste0("price_gap_min_", alternative)]] <-
      (price[, alternative] - price_min) / 11
    features[[paste0("price_gap_max_", alternative)]] <-
      (price_max - price[, alternative]) / 11
  }

  x <- do.call(cbind, features)
  storage.mode(x) <- "double"
  if (is.null(keep_columns)) {
    keep_columns <- colnames(x)[
      apply(x, 2, function(column) {
        all(is.finite(column)) && sd(column) > 0
      })
    ]
  }
  stopifnot(all(keep_columns %in% colnames(x)))
  list(
    x = x[, keep_columns, drop = FALSE],
    keep_columns = keep_columns
  )
}

fit_mlp_average <- function(x_train, y_train, x_valid,
                            size, decay, seeds,
                            max_iterations = 200L) {
  predictions <- array(
    NA_real_,
    dim = c(nrow(x_valid), 4L, length(seeds))
  )
  fit_rows <- list()
  models <- vector("list", length(seeds))

  for (seed_index in seq_along(seeds)) {
    seed <- seeds[[seed_index]]
    set.seed(seed)
    started <- proc.time()[["elapsed"]]
    model <- nnet(
      x = x_train,
      y = y_train,
      size = size,
      softmax = TRUE,
      decay = decay,
      maxit = max_iterations,
      trace = FALSE,
      rang = 0.10,
      MaxNWts = 100000,
      abstol = 1e-5,
      reltol = 1e-8
    )
    elapsed <- proc.time()[["elapsed"]] - started
    pred <- predict(model, x_valid, type = "raw")
    pred <- as.matrix(pred)
    pred <- pred / rowSums(pred)
    predictions[, , seed_index] <- pred
    fit_rows[[seed_index]] <- data.frame(
      seed = seed,
      convergence = model$convergence,
      training_objective = model$value,
      elapsed_seconds = elapsed
    )
    models[[seed_index]] <- model
    cat(sprintf(
      "MLP h=%d decay=%g seed=%d: objective %.3f, convergence %d, %.1fs\n",
      size, decay, seed, model$value,
      model$convergence, elapsed
    ))
    flush.console()
  }

  list(
    pred = apply(predictions, c(1, 2), mean),
    fits = do.call(rbind, fit_rows),
    models = models
  )
}

xgb_screen_prediction <- function(split_data) {
  tr <- split_data$train_wide_tr
  va <- split_data$train_wide_val
  tr$Choice <- max.col(tr[, paste0("Ch", 1:4)])
  model <- xgb.train(
    params = list(
      objective = "multi:softprob",
      num_class = 4,
      eval_metric = "mlogloss",
      eta = 0.1,
      max_depth = 4,
      subsample = 0.8,
      colsample_bytree = 0.8,
      seed = 4821,
      nthread = 1
    ),
    data = xgb.DMatrix(
      wide_feature_matrix(tr),
      label = tr$Choice - 1L
    ),
    nrounds = 73,
    verbose = 0
  )
  pred <- predict(
    model,
    xgb.DMatrix(wide_feature_matrix(va))
  )
  pred[order(va$No), , drop = FALSE]
}

respondent_bootstrap_gain <- function(truth, baseline, candidate,
                                      case, seed = 4821,
                                      replicates = 2000L) {
  row_loss <- function(pred) {
    pred <- pmin(pmax(pred / rowSums(pred), 1e-15), 1 - 1e-15)
    -rowSums(truth * log(pred))
  }
  respondent_gain <- tapply(
    row_loss(baseline) - row_loss(candidate),
    case,
    mean
  )
  set.seed(seed)
  boot <- replicate(
    replicates,
    mean(sample(
      respondent_gain, length(respondent_gain), replace = TRUE
    ))
  )
  data.frame(
    point_gain = mean(respondent_gain),
    bootstrap_mean = mean(boot),
    bootstrap_sd = sd(boot),
    lower_95 = unname(quantile(boot, 0.025)),
    upper_95 = unname(quantile(boot, 0.975)),
    win_rate = mean(boot > 0)
  )
}

if (stage == "screen") {
  split_data <- readRDS("data_processed/train_val_split.rds")
  full_train <- read.csv("csv files/train.csv")
  reference <- feature_reference(full_train)
  scaler <- continuous_scaler(split_data$train_wide_tr)
  tr_matrix <- build_mlp_matrix(
    split_data$train_wide_tr, reference, scaler
  )
  va_matrix <- build_mlp_matrix(
    split_data$train_wide_val, reference, scaler,
    keep_columns = tr_matrix$keep_columns
  )
  tr_wide <- split_data$train_wide_tr
  va_wide <- split_data$train_wide_val
  tr_order <- order(tr_wide$No)
  va_order <- order(va_wide$No)
  x_train <- tr_matrix$x[tr_order, , drop = FALSE]
  x_valid <- va_matrix$x[va_order, , drop = FALSE]
  truth_train <- as.matrix(
    tr_wide[tr_order, paste0("Ch", 1:4)]
  )
  truth_valid <- as.matrix(
    va_wide[va_order, paste0("Ch", 1:4)]
  )

  tr_long <- prepare_mlp_long(split_data$train_long_tr)
  va_long <- prepare_mlp_long(split_data$train_long_val)
  baseline_fit <- fit_predict_m8trpg(tr_long, va_long)
  baseline <- baseline_fit$pred[
    match(va_wide$No[va_order], baseline_fit$no), ,
    drop = FALSE
  ]
  baseline_loss <- log_loss_matrix(truth_valid, baseline)
  stopifnot(abs(baseline_loss - 1.15968144721113) < 1e-8)
  xgb <- xgb_screen_prediction(split_data)
  v11_screen <- 0.8 * baseline + 0.2 * xgb
  v11_screen_loss <- log_loss_matrix(truth_valid, v11_screen)

  seeds <- c(4821L, 4822L, 4823L)
  weight_grid <- seq(0, 0.30, by = 0.01)
  result_rows <- list()
  fit_rows <- list()
  predictions <- list()

  for (row in seq_len(nrow(mlp_configs))) {
    config <- mlp_configs[row, , drop = FALSE]
    fitted <- fit_mlp_average(
      x_train, truth_train, x_valid,
      size = config$size,
      decay = config$decay,
      seeds = seeds,
      max_iterations = as.integer(
        Sys.getenv("CODEX_MLP_MAXIT", "200")
      )
    )
    component_loss <- log_loss_matrix(
      truth_valid, fitted$pred
    )
    blend_losses <- vapply(weight_grid, function(weight) {
      log_loss_matrix(
        truth_valid,
        (1 - weight) * v11_screen +
          weight * fitted$pred
      )
    }, numeric(1))
    best_index <- which.min(blend_losses)
    result_rows[[config$config]] <- data.frame(
      config = config$config,
      size = config$size,
      decay = config$decay,
      n_features = ncol(x_train),
      n_seeds = length(seeds),
      component_logloss = component_loss,
      v11_screen_logloss = v11_screen_loss,
      best_mlp_weight = weight_grid[[best_index]],
      best_blend_logloss = blend_losses[[best_index]],
      blend_gain =
        v11_screen_loss - blend_losses[[best_index]]
    )
    fit_rows[[config$config]] <- cbind(
      data.frame(config = config$config),
      fitted$fits
    )
    predictions[[config$config]] <- fitted$pred
    cat(sprintf(
      "%s: component %.6f; best w %.2f, blend %.6f, gain %+.6f\n",
      config$config, component_loss,
      weight_grid[[best_index]],
      blend_losses[[best_index]],
      v11_screen_loss - blend_losses[[best_index]]
    ))
    flush.console()
  }

  result <- do.call(rbind, result_rows)
  write_result_csv(
    result, file.path(output_dir, "mlp_screen.csv")
  )
  write_result_csv(
    do.call(rbind, fit_rows),
    file.path(output_dir, "mlp_screen_fits.csv")
  )
  saveRDS(
    list(
      result = result,
      predictions = predictions,
      validation_no = va_wide$No[va_order],
      truth = truth_valid,
      v11_screen = v11_screen,
      feature_columns = tr_matrix$keep_columns
    ),
    file.path(output_dir, "mlp_screen.rds")
  )
  print(result, digits = 7)
}

if (stage == "refine") {
  screen <- read.csv(
    file.path(output_dir, "mlp_screen.csv"),
    stringsAsFactors = FALSE
  )
  requested <- Sys.getenv("CODEX_CANDIDATES", "")
  if (nzchar(requested)) {
    candidates <- strsplit(requested, ",", fixed = TRUE)[[1]]
  } else {
    candidates <- screen$config[which.min(screen$best_blend_logloss)]
  }
  stopifnot(all(candidates %in% mlp_configs$config))

  split_data <- readRDS("data_processed/train_val_split.rds")
  full_train <- read.csv("csv files/train.csv")
  reference <- feature_reference(full_train)
  scaler <- continuous_scaler(split_data$train_wide_tr)
  tr_matrix <- build_mlp_matrix(
    split_data$train_wide_tr, reference, scaler
  )
  va_matrix <- build_mlp_matrix(
    split_data$train_wide_val, reference, scaler,
    keep_columns = tr_matrix$keep_columns
  )
  tr_wide <- split_data$train_wide_tr
  va_wide <- split_data$train_wide_val
  tr_order <- order(tr_wide$No)
  va_order <- order(va_wide$No)
  x_train <- tr_matrix$x[tr_order, , drop = FALSE]
  x_valid <- va_matrix$x[va_order, , drop = FALSE]
  truth_train <- as.matrix(
    tr_wide[tr_order, paste0("Ch", 1:4)]
  )
  truth_valid <- as.matrix(
    va_wide[va_order, paste0("Ch", 1:4)]
  )

  tr_long <- prepare_mlp_long(split_data$train_long_tr)
  va_long <- prepare_mlp_long(split_data$train_long_val)
  baseline_fit <- fit_predict_m8trpg(tr_long, va_long)
  baseline <- baseline_fit$pred[
    match(va_wide$No[va_order], baseline_fit$no), ,
    drop = FALSE
  ]
  stopifnot(
    abs(log_loss_matrix(truth_valid, baseline) -
      1.15968144721113) < 1e-8
  )
  xgb <- xgb_screen_prediction(split_data)
  v11_screen <- 0.8 * baseline + 0.2 * xgb
  v11_screen_loss <- log_loss_matrix(truth_valid, v11_screen)

  max_iterations <- as.integer(
    Sys.getenv("CODEX_MLP_MAXIT", "600")
  )
  seeds <- c(4821L, 4822L, 4823L)
  weight_grid <- seq(0, 0.30, by = 0.01)
  result_rows <- list()
  fit_rows <- list()
  predictions <- list()

  for (candidate in candidates) {
    config <- mlp_configs[
      mlp_configs$config == candidate, , drop = FALSE
    ]
    fitted <- fit_mlp_average(
      x_train, truth_train, x_valid,
      size = config$size,
      decay = config$decay,
      seeds = seeds,
      max_iterations = max_iterations
    )
    component_loss <- log_loss_matrix(truth_valid, fitted$pred)
    blend_losses <- vapply(weight_grid, function(weight) {
      log_loss_matrix(
        truth_valid,
        (1 - weight) * v11_screen + weight * fitted$pred
      )
    }, numeric(1))
    best_index <- which.min(blend_losses)
    result_rows[[candidate]] <- data.frame(
      config = candidate,
      size = config$size,
      decay = config$decay,
      max_iterations = max_iterations,
      n_features = ncol(x_train),
      n_seeds = length(seeds),
      all_converged = all(fitted$fits$convergence == 0),
      component_logloss = component_loss,
      v11_screen_logloss = v11_screen_loss,
      best_mlp_weight = weight_grid[[best_index]],
      best_blend_logloss = blend_losses[[best_index]],
      blend_gain = v11_screen_loss - blend_losses[[best_index]]
    )
    fit_rows[[candidate]] <- cbind(
      data.frame(config = candidate),
      fitted$fits
    )
    predictions[[candidate]] <- fitted$pred
  }

  result <- do.call(rbind, result_rows)
  fits <- do.call(rbind, fit_rows)
  write_result_csv(
    result, file.path(output_dir, "mlp_refine.csv")
  )
  write_result_csv(
    fits, file.path(output_dir, "mlp_refine_fits.csv")
  )
  saveRDS(
    list(
      result = result,
      fits = fits,
      predictions = predictions,
      validation_no = va_wide$No[va_order],
      truth = truth_valid,
      v11_screen = v11_screen,
      feature_columns = tr_matrix$keep_columns
    ),
    file.path(output_dir, "mlp_refine.rds")
  )
  print(result, digits = 7)
  print(fits, digits = 7)
}

if (stage == "cv") {
  screen <- read.csv(
    file.path(output_dir, "mlp_screen.csv"),
    stringsAsFactors = FALSE
  )
  requested <- Sys.getenv("CODEX_CANDIDATES", "")
  if (nzchar(requested)) {
    candidates <- strsplit(requested, ",", fixed = TRUE)[[1]]
  } else {
    viable <- screen[
      screen$best_mlp_weight > 0 &
        screen$blend_gain > 0,
      ,
      drop = FALSE
    ]
    if (nrow(viable) == 0L) {
      cat("No MLP configuration improved the screen blend; CV skipped.\n")
      quit(save = "no", status = 0)
    }
    candidates <- viable$config[
      which.min(viable$best_blend_logloss)
    ]
  }
  stopifnot(all(candidates %in% mlp_configs$config))

  train <- read.csv("csv files/train.csv")
  truth <- as.matrix(train[, paste0("Ch", 1:4)])
  reference <- feature_reference(train)
  saved <- readRDS("data_processed/oof_ensemble_v10.rds")
  fold_map <- saved$fold_of_case
  baseline <- saved$oof_mlogit
  xgb <- saved$oof_xgb
  v11 <- 0.8 * baseline + 0.2 * xgb
  stopifnot(
    abs(log_loss_matrix(truth, baseline) - 1.1470212110518) < 1e-8,
    abs(log_loss_matrix(truth, v11) - 1.145094) < 5e-6
  )

  candidate_oof <- lapply(
    candidates,
    function(x) matrix(NA_real_, nrow(train), 4)
  )
  names(candidate_oof) <- candidates
  fit_rows <- list()
  seeds <- 4821L + 0:4

  for (candidate in candidates) {
    config <- mlp_configs[
      mlp_configs$config == candidate, , drop = FALSE
    ]
    for (fold in 1:5) {
      val_cases <- as.integer(names(fold_map)[fold_map == fold])
      train_rows <- !(train$Case %in% val_cases)
      valid_rows <- train$Case %in% val_cases
      tr <- train[train_rows, , drop = FALSE]
      va <- train[valid_rows, , drop = FALSE]
      scaler <- continuous_scaler(tr)
      tr_matrix <- build_mlp_matrix(tr, reference, scaler)
      va_matrix <- build_mlp_matrix(
        va, reference, scaler,
        keep_columns = tr_matrix$keep_columns
      )
      fitted <- fit_mlp_average(
        tr_matrix$x,
        as.matrix(tr[, paste0("Ch", 1:4)]),
        va_matrix$x,
        size = config$size,
        decay = config$decay,
        seeds = seeds + fold * 100L,
        max_iterations = as.integer(
          Sys.getenv("CODEX_MLP_MAXIT", "600")
        )
      )
      idx <- match(va$No, train$No)
      candidate_oof[[candidate]][idx, ] <- fitted$pred
      fit_rows[[length(fit_rows) + 1L]] <- cbind(
        data.frame(
          candidate = candidate,
          fold = fold
        ),
        fitted$fits
      )
      saveRDS(
        list(
          candidate = candidate,
          fold = fold,
          prediction = fitted$pred,
          validation_index = idx,
          fits = fitted$fits,
          feature_columns = tr_matrix$keep_columns
        ),
        file.path(
          output_dir,
          sprintf("mlp_%s_fold_%d.rds", candidate, fold)
        )
      )
      cat(sprintf(
        "%s fold %d complete: component %.6f\n",
        candidate, fold,
        log_loss_matrix(truth[idx, ], fitted$pred)
      ))
      flush.console()
    }
    stopifnot(!anyNA(candidate_oof[[candidate]]))
  }

  weight_grid <- seq(0, 0.30, by = 0.01)
  row_fold <- unname(
    fold_map[as.character(train$Case)]
  )
  summary_rows <- list()
  weight_rows <- list()
  bootstrap_rows <- list()

  for (candidate in candidates) {
    pred <- candidate_oof[[candidate]]
    global_losses <- vapply(weight_grid, function(weight) {
      log_loss_matrix(
        truth, (1 - weight) * v11 + weight * pred
      )
    }, numeric(1))
    global_index <- which.min(global_losses)

    crossfit <- matrix(NA_real_, nrow(train), 4)
    for (fold in 1:5) {
      train_weight_rows <- row_fold != fold
      validation_rows <- row_fold == fold
      training_losses <- vapply(
        weight_grid,
        function(weight) {
          log_loss_matrix(
            truth[train_weight_rows, ],
            (1 - weight) * v11[train_weight_rows, ] +
              weight * pred[train_weight_rows, ]
          )
        },
        numeric(1)
      )
      best_index <- which.min(training_losses)
      best_weight <- weight_grid[[best_index]]
      crossfit[validation_rows, ] <-
        (1 - best_weight) * v11[validation_rows, ] +
        best_weight * pred[validation_rows, ]
      weight_rows[[length(weight_rows) + 1L]] <- data.frame(
        candidate = candidate,
        fold = fold,
        mlp_weight = best_weight,
        training_logloss = training_losses[[best_index]]
      )
    }
    stopifnot(!anyNA(crossfit))

    summary_rows[[candidate]] <- data.frame(
      candidate = candidate,
      component_logloss = log_loss_matrix(truth, pred),
      v11_logloss = log_loss_matrix(truth, v11),
      global_best_weight = weight_grid[[global_index]],
      global_best_blend_logloss =
        global_losses[[global_index]],
      global_optimistic_gain =
        log_loss_matrix(truth, v11) -
        global_losses[[global_index]],
      crossfit_blend_logloss =
        log_loss_matrix(truth, crossfit),
      crossfit_blend_gain =
        log_loss_matrix(truth, v11) -
        log_loss_matrix(truth, crossfit)
    )
    bootstrap_rows[[candidate]] <- cbind(
      data.frame(candidate = candidate),
      respondent_bootstrap_gain(
        truth, v11, crossfit, train$Case
      )
    )
  }

  summary <- do.call(rbind, summary_rows)
  bootstrap <- do.call(rbind, bootstrap_rows)
  write_result_csv(
    do.call(rbind, fit_rows),
    file.path(output_dir, "mlp_cv_fits.csv")
  )
  write_result_csv(
    do.call(rbind, weight_rows),
    file.path(output_dir, "mlp_cv_weights.csv")
  )
  write_result_csv(
    summary,
    file.path(output_dir, "mlp_cv.csv")
  )
  write_result_csv(
    bootstrap,
    file.path(output_dir, "mlp_bootstrap.csv")
  )
  saveRDS(
    list(
      candidates = candidates,
      oof = candidate_oof,
      summary = summary,
      bootstrap = bootstrap
    ),
    file.path(output_dir, "mlp_oof.rds")
  )
  print(summary, digits = 7)
  print(bootstrap, digits = 7)
}
