# LightGBM as a distinct tree-based ensemble component.
#
# Native categorical splits are used for attribute levels, Price levels, and
# categorical respondent covariates. This differs materially from the existing
# xgboost models, which treat their integer codes as ordinary numeric inputs.
# A three-configuration screen precedes canonical respondent-grouped CV.

options(stringsAsFactors = FALSE)

suppressPackageStartupMessages(library(lightgbm))
source("R/codex_modeling_common.R")

stage <- Sys.getenv("CODEX_LGB_STAGE", "screen")
stopifnot(stage %in% c("screen", "cv"))

output_dir <- "data_processed/codex_deep_stack"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

lgb_configs <- list(
  leaves15 = list(
    num_leaves = 15L,
    max_depth = 5L,
    min_data_in_leaf = 120L,
    feature_fraction = 0.85,
    bagging_fraction = 0.85,
    lambda_l2 = 8,
    learning_rate = 0.03,
    nrounds = 180L
  ),
  leaves31 = list(
    num_leaves = 31L,
    max_depth = 6L,
    min_data_in_leaf = 100L,
    feature_fraction = 0.80,
    bagging_fraction = 0.85,
    lambda_l2 = 10,
    learning_rate = 0.025,
    nrounds = 200L
  ),
  conservative = list(
    num_leaves = 15L,
    max_depth = 4L,
    min_data_in_leaf = 200L,
    feature_fraction = 0.90,
    bagging_fraction = 0.90,
    lambda_l2 = 15,
    learning_rate = 0.03,
    nrounds = 180L
  )
)

wide_columns <- c(
  paste0(rep(attrs, each = 4), rep(1:4, times = length(attrs))),
  paste0("Price", 1:4),
  xgb_covariates
)
categorical_covariates <- c(
  "segmentind", "yearind", "milesind", "nightind",
  "pparkind", "genderind", "ageind", "educind",
  "regionind", "Urbind", "incomeind"
)
categorical_columns <- c(
  paste0(rep(attrs, each = 4), rep(1:4, times = length(attrs))),
  paste0("Price", 1:4),
  categorical_covariates
)
stopifnot(all(categorical_columns %in% wide_columns))

feature_matrix <- function(wide) {
  value <- as.matrix(wide[, wide_columns, drop = FALSE])
  storage.mode(value) <- "double"
  value
}

fit_lightgbm <- function(x, y, config, seed) {
  dataset <- lgb.Dataset(
    data = x,
    label = max.col(y) - 1L,
    colnames = colnames(x),
    categorical_feature = categorical_columns
  )
  started <- proc.time()[["elapsed"]]
  model <- lgb.train(
    params = list(
      objective = "multiclass",
      num_class = 4L,
      metric = "multi_logloss",
      num_leaves = config$num_leaves,
      max_depth = config$max_depth,
      min_data_in_leaf = config$min_data_in_leaf,
      feature_fraction = config$feature_fraction,
      bagging_fraction = config$bagging_fraction,
      bagging_freq = 1L,
      lambda_l1 = 0,
      lambda_l2 = config$lambda_l2,
      learning_rate = config$learning_rate,
      max_cat_to_onehot = 8L,
      cat_smooth = 20,
      seed = seed,
      num_threads = 4L,
      deterministic = TRUE,
      force_col_wise = TRUE,
      verbosity = -1L
    ),
    data = dataset,
    nrounds = config$nrounds,
    verbose = -1L
  )
  list(
    model = model,
    elapsed_seconds = proc.time()[["elapsed"]] - started
  )
}

predict_lightgbm <- function(model, x) {
  prediction <- as.matrix(predict(model, x))
  stopifnot(identical(dim(prediction), c(nrow(x), 4L)))
  prediction <- pmax(prediction, 1e-15)
  prediction / rowSums(prediction)
}

crossfit_blend <- function(truth, baseline, component, row_fold) {
  weight_grid <- seq(0, 0.40, by = 0.01)
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
  list(
    prediction = prediction,
    weights = weights,
    logloss = log_loss_matrix(truth, prediction)
  )
}

bootstrap_gain <- function(truth, case, baseline, candidate,
                           family_size = 3L,
                           n_boot = 100000L) {
  row_loss <- function(prediction) {
    -rowSums(
      truth * log(pmax(prediction / rowSums(prediction), 1e-15))
    )
  }
  case_gain <- unname(tapply(
    row_loss(baseline) - row_loss(candidate),
    case,
    mean
  ))
  set.seed(4821L)
  n_case <- length(case_gain)
  boot <- numeric(n_boot)
  for (start in seq.int(1L, n_boot, by = 1000L)) {
    stop_at <- min(n_boot, start + 999L)
    n_this <- stop_at - start + 1L
    sampled <- matrix(
      sample.int(n_case, n_case * n_this, replace = TRUE),
      nrow = n_case
    )
    boot[start:stop_at] <- colMeans(matrix(
      case_gain[sampled],
      nrow = n_case
    ))
  }
  family_alpha <- 0.05 / family_size
  list(
    summary = data.frame(
      point_gain = mean(case_gain),
      bootstrap_sd = sd(boot),
      lower_95 = unname(quantile(boot, 0.025)),
      upper_95 = unname(quantile(boot, 0.975)),
      lower_99 = unname(quantile(boot, 0.005)),
      upper_99 = unname(quantile(boot, 0.995)),
      lower_bonferroni = unname(
        quantile(boot, family_alpha / 2)
      ),
      upper_bonferroni = unname(
        quantile(boot, 1 - family_alpha / 2)
      ),
      win_rate = mean(boot > 0),
      n_boot = n_boot,
      family_size = family_size
    ),
    bootstrap = boot,
    case_gain = case_gain
  )
}

if (stage == "screen") {
  split_data <- readRDS("data_processed/train_val_split.rds")
  train_order <- order(split_data$train_wide_tr$No)
  validation_order <- order(split_data$train_wide_val$No)
  x_train <- feature_matrix(
    split_data$train_wide_tr[train_order, , drop = FALSE]
  )
  x_valid <- feature_matrix(
    split_data$train_wide_val[
      validation_order, , drop = FALSE
    ]
  )
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
  stopifnot(identical(
    validation_no, saved_screen$validation_no
  ))
  v11 <- saved_screen$v11_screen
  shallow <- saved_screen$predictions[["h08_d0.100"]]
  current <- 0.85 * v11 + 0.15 * shallow
  weight_grid <- seq(0, 0.40, by = 0.01)

  result_rows <- list()
  predictions <- list()
  for (config_name in names(lgb_configs)) {
    config <- lgb_configs[[config_name]]
    fitted <- fit_lightgbm(
      x_train, y_train, config,
      seed = 4821L + match(config_name, names(lgb_configs))
    )
    prediction <- predict_lightgbm(fitted$model, x_valid)
    losses <- vapply(weight_grid, function(weight) {
      log_loss_matrix(
        truth,
        (1 - weight) * current + weight * prediction
      )
    }, numeric(1))
    best <- which.min(losses)
    result_rows[[config_name]] <- data.frame(
      config = config_name,
      num_leaves = config$num_leaves,
      max_depth = config$max_depth,
      min_data_in_leaf = config$min_data_in_leaf,
      learning_rate = config$learning_rate,
      nrounds = config$nrounds,
      elapsed_seconds = fitted$elapsed_seconds,
      component_logloss = log_loss_matrix(truth, prediction),
      current_logloss = log_loss_matrix(truth, current),
      best_weight = weight_grid[[best]],
      blend_logloss = losses[[best]],
      gain_vs_current =
        log_loss_matrix(truth, current) - losses[[best]]
    )
    predictions[[config_name]] <- prediction
    cat(sprintf(
      "%s: component %.6f; weight %.2f; blend %.6f\n",
      config_name, log_loss_matrix(truth, prediction),
      weight_grid[[best]], losses[[best]]
    ))
  }
  result <- do.call(rbind, result_rows)
  write.csv(
    result,
    file.path(output_dir, "lightgbm_screen.csv"),
    row.names = FALSE
  )
  saveRDS(
    list(
      result = result,
      predictions = predictions,
      validation_no = validation_no,
      truth = truth,
      current = current
    ),
    file.path(output_dir, "lightgbm_screen.rds")
  )
  print(result, digits = 9)
}

if (stage == "cv") {
  screen <- read.csv(
    file.path(output_dir, "lightgbm_screen.csv")
  )
  requested <- Sys.getenv("CODEX_LGB_CONFIG", "")
  if (nzchar(requested)) {
    config_name <- requested
  } else {
    viable <- screen[
      screen$gain_vs_current > 0,
      ,
      drop = FALSE
    ]
    if (nrow(viable) == 0L) {
      cat("No LightGBM screen candidate improved current; CV skipped.\n")
      quit(save = "no", status = 0)
    }
    config_name <- viable$config[[which.min(viable$blend_logloss)]]
  }
  stopifnot(config_name %in% names(lgb_configs))
  config <- lgb_configs[[config_name]]

  train <- read.csv("csv files/train.csv")
  truth <- as.matrix(train[, paste0("Ch", 1:4)])
  fold_map <- canonical_fold_map()
  row_fold <- unname(fold_map[as.character(train$Case)])
  current <- readRDS(
    "data_processed/codex_behavioral_round/mlp_precision.rds"
  )$crossfit_prediction
  stopifnot(abs(
    log_loss_matrix(truth, current) - 1.14378944178118
  ) < 1e-8)

  oof <- matrix(NA_real_, nrow(train), 4L)
  fit_rows <- list()
  for (fold in 1:5) {
    validation_cases <- as.integer(
      names(fold_map)[fold_map == fold]
    )
    training_rows <- !(train$Case %in% validation_cases)
    validation_rows <- train$Case %in% validation_cases
    tr <- train[training_rows, , drop = FALSE]
    va <- train[validation_rows, , drop = FALSE]
    fitted <- fit_lightgbm(
      feature_matrix(tr),
      as.matrix(tr[, paste0("Ch", 1:4)]),
      config,
      seed = 5821L + fold
    )
    prediction <- predict_lightgbm(
      fitted$model, feature_matrix(va)
    )
    index <- match(va$No, train$No)
    oof[index, ] <- prediction
    fit_rows[[fold]] <- data.frame(
      config = config_name,
      fold = fold,
      elapsed_seconds = fitted$elapsed_seconds,
      fold_logloss = log_loss_matrix(
        truth[index, , drop = FALSE], prediction
      )
    )
    cat(sprintf(
      "LightGBM fold %d: %.9f\n",
      fold, fit_rows[[fold]]$fold_logloss
    ))
    flush.console()
  }
  stopifnot(
    !anyNA(oof),
    max(abs(rowSums(oof) - 1)) < 1e-6
  )
  blend <- crossfit_blend(
    truth, current, oof, row_fold
  )
  bootstrap <- bootstrap_gain(
    truth, train$Case, current, blend$prediction,
    family_size = length(lgb_configs),
    n_boot = 100000L
  )
  summary <- data.frame(
    config = config_name,
    component_logloss = log_loss_matrix(truth, oof),
    current_logloss = log_loss_matrix(truth, current),
    blend_logloss = blend$logloss,
    gain_vs_current =
      log_loss_matrix(truth, current) - blend$logloss
  )
  write.csv(
    summary,
    file.path(output_dir, "lightgbm_cv.csv"),
    row.names = FALSE
  )
  write.csv(
    do.call(rbind, fit_rows),
    file.path(output_dir, "lightgbm_cv_fits.csv"),
    row.names = FALSE
  )
  write.csv(
    data.frame(fold = 1:5, lightgbm_weight = blend$weights),
    file.path(output_dir, "lightgbm_cv_weights.csv"),
    row.names = FALSE
  )
  write.csv(
    bootstrap$summary,
    file.path(output_dir, "lightgbm_cv_bootstrap.csv"),
    row.names = FALSE
  )
  saveRDS(
    list(
      config = config_name,
      specification = config,
      oof = oof,
      blend = blend,
      summary = summary,
      bootstrap = bootstrap
    ),
    file.path(output_dir, "lightgbm_oof.rds")
  )
  print(summary, digits = 9)
  print(bootstrap$summary, digits = 9)
  print(data.frame(fold = 1:5, weight = blend$weights))
}
