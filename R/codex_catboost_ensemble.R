# CatBoost as a distinct tree-based ensemble component.
#
# The screen deliberately mirrors R/codex_lightgbm_ensemble.R:
#   1. use the canonical respondent-level single split;
#   2. use the established wide_feature_matrix() inputs;
#   3. compare each component against the submitted shallow-MLP blend;
#   4. stop before CV if every configuration selects zero blend weight.
#
# CatBoost receives the attribute, price, and categorical respondent columns
# as native categorical features. With one_hot_max_size = 2, non-binary
# categoricals use CatBoost's ordered target-statistic machinery rather than
# being treated as ordinary numeric split variables.

options(stringsAsFactors = FALSE)

suppressPackageStartupMessages(library(catboost))
source("R/codex_modeling_common.R")

stage <- Sys.getenv("CODEX_CAT_STAGE", "screen")
stopifnot(stage %in% c("screen", "cv"))

output_dir <- "data_processed/codex_catboost"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

catboost_configs <- list(
  ordered_d5 = list(
    depth = 5L,
    learning_rate = 0.04,
    iterations = 1000L,
    l2_leaf_reg = 10,
    random_strength = 1.0,
    bagging_temperature = 1.0,
    rsm = 0.90,
    max_ctr_complexity = 1L
  ),
  ordered_d6 = list(
    depth = 6L,
    learning_rate = 0.03,
    iterations = 1200L,
    l2_leaf_reg = 12,
    random_strength = 0.5,
    bagging_temperature = 0.5,
    rsm = 0.90,
    max_ctr_complexity = 1L
  ),
  ordered_d7 = list(
    depth = 7L,
    learning_rate = 0.025,
    iterations = 1400L,
    l2_leaf_reg = 15,
    random_strength = 1.0,
    bagging_temperature = 1.0,
    rsm = 0.85,
    max_ctr_complexity = 1L
  ),
  ordered_ctr2 = list(
    depth = 6L,
    learning_rate = 0.03,
    iterations = 1200L,
    l2_leaf_reg = 15,
    random_strength = 1.0,
    bagging_temperature = 0.5,
    rsm = 0.85,
    max_ctr_complexity = 2L
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
  value <- wide_feature_matrix(wide)
  stopifnot(identical(colnames(value), wide_columns))
  value
}

# catboost.load_pool() uses zero-based categorical feature indices.
categorical_indices <- match(categorical_columns, wide_columns) - 1L
stopifnot(
  !anyNA(categorical_indices),
  length(unique(categorical_indices)) == length(categorical_indices),
  all(categorical_indices >= 0L),
  all(categorical_indices < length(wide_columns))
)

make_pool <- function(x, truth = NULL) {
  label <- if (is.null(truth)) NULL else max.col(truth) - 1L
  catboost.load_pool(
    data = x,
    label = label,
    cat_features = categorical_indices,
    feature_names = as.list(colnames(x)),
    thread_count = 4L
  )
}

fit_catboost <- function(x_train, y_train, config, seed,
                         x_valid = NULL, y_valid = NULL,
                         fixed_iterations = NULL) {
  train_pool <- make_pool(x_train, y_train)
  has_validation <- !is.null(x_valid)
  if (has_validation) {
    stopifnot(!is.null(y_valid))
    validation_pool <- make_pool(x_valid, y_valid)
  } else {
    validation_pool <- NULL
  }

  iteration_count <- if (is.null(fixed_iterations)) {
    config$iterations
  } else {
    as.integer(fixed_iterations)
  }
  stopifnot(iteration_count >= 1L)

  params <- list(
    loss_function = "MultiClass",
    eval_metric = "MultiClass",
    boosting_type = "Ordered",
    depth = config$depth,
    learning_rate = config$learning_rate,
    iterations = iteration_count,
    l2_leaf_reg = config$l2_leaf_reg,
    random_strength = config$random_strength,
    bootstrap_type = "Bayesian",
    bagging_temperature = config$bagging_temperature,
    rsm = config$rsm,
    one_hot_max_size = 2L,
    max_ctr_complexity = config$max_ctr_complexity,
    random_seed = seed,
    thread_count = 4L,
    allow_writing_files = FALSE,
    logging_level = "Silent"
  )
  if (has_validation && is.null(fixed_iterations)) {
    params$od_type <- "Iter"
    params$od_wait <- 80L
    params$use_best_model <- TRUE
  } else {
    params$use_best_model <- FALSE
  }

  started <- proc.time()[["elapsed"]]
  model <- catboost.train(
    learn_pool = train_pool,
    test_pool = validation_pool,
    params = params
  )
  list(
    model = model,
    tree_count = model$tree_count,
    elapsed_seconds = proc.time()[["elapsed"]] - started
  )
}

predict_catboost <- function(model, x) {
  pool <- make_pool(x)
  raw <- catboost.predict(
    model,
    pool,
    prediction_type = "Probability"
  )
  if (is.matrix(raw)) {
    prediction <- as.matrix(raw)
  } else {
    stopifnot(length(raw) == nrow(x) * 4L)
    # CatBoost's R multiclass output is flattened class-major.
    prediction <- matrix(raw, nrow = nrow(x), ncol = 4L, byrow = FALSE)
  }
  stopifnot(identical(dim(prediction), c(nrow(x), 4L)))
  prediction <- pmax(prediction, 1e-15)
  prediction <- prediction / rowSums(prediction)
  stopifnot(
    all(is.finite(prediction)),
    max(abs(rowSums(prediction) - 1)) < 1e-10
  )
  prediction
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
  stopifnot(!anyNA(prediction))
  list(
    prediction = prediction,
    weights = weights,
    logloss = log_loss_matrix(truth, prediction)
  )
}

bootstrap_gain <- function(truth, case, baseline, candidate,
                           family_size = 4L,
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
  current_logloss <- log_loss_matrix(truth, current)
  stopifnot(abs(current_logloss - 1.16041189298361) < 1e-10)

  weight_grid <- seq(0, 0.40, by = 0.01)
  result_rows <- list()
  predictions <- list()
  models <- list()
  for (config_name in names(catboost_configs)) {
    config <- catboost_configs[[config_name]]
    fitted <- fit_catboost(
      x_train, y_train, config,
      seed = 6821L + match(config_name, names(catboost_configs)),
      x_valid = x_valid,
      y_valid = truth
    )
    prediction <- predict_catboost(fitted$model, x_valid)
    losses <- vapply(weight_grid, function(weight) {
      log_loss_matrix(
        truth,
        (1 - weight) * current + weight * prediction
      )
    }, numeric(1))
    best <- which.min(losses)
    result_rows[[config_name]] <- data.frame(
      config = config_name,
      depth = config$depth,
      learning_rate = config$learning_rate,
      max_iterations = config$iterations,
      best_iterations = fitted$tree_count,
      l2_leaf_reg = config$l2_leaf_reg,
      random_strength = config$random_strength,
      bagging_temperature = config$bagging_temperature,
      rsm = config$rsm,
      max_ctr_complexity = config$max_ctr_complexity,
      elapsed_seconds = fitted$elapsed_seconds,
      component_logloss = log_loss_matrix(truth, prediction),
      current_logloss = current_logloss,
      best_weight = weight_grid[[best]],
      blend_logloss = losses[[best]],
      gain_vs_current = current_logloss - losses[[best]]
    )
    predictions[[config_name]] <- prediction
    models[[config_name]] <- fitted$model
    cat(sprintf(
      paste0(
        "%s: trees %d; component %.6f; ",
        "weight %.2f; blend %.6f\n"
      ),
      config_name, fitted$tree_count,
      log_loss_matrix(truth, prediction),
      weight_grid[[best]], losses[[best]]
    ))
    flush.console()
  }
  result <- do.call(rbind, result_rows)
  write.csv(
    result,
    file.path(output_dir, "catboost_screen.csv"),
    row.names = FALSE
  )
  saveRDS(
    list(
      package_version = as.character(packageVersion("catboost")),
      result = result,
      predictions = predictions,
      models = models,
      validation_no = validation_no,
      truth = truth,
      current = current,
      categorical_columns = categorical_columns
    ),
    file.path(output_dir, "catboost_screen.rds")
  )
  print(result, digits = 9)
}

if (stage == "cv") {
  screen <- read.csv(
    file.path(output_dir, "catboost_screen.csv")
  )
  requested <- Sys.getenv("CODEX_CAT_CONFIG", "")
  if (nzchar(requested)) {
    config_name <- requested
  } else {
    viable <- screen[
      screen$best_weight > 0 & screen$gain_vs_current > 0,
      ,
      drop = FALSE
    ]
    if (nrow(viable) == 0L) {
      cat("No CatBoost screen candidate improved current; CV skipped.\n")
      quit(save = "no", status = 0)
    }
    config_name <- viable$config[[which.min(viable$blend_logloss)]]
  }
  stopifnot(config_name %in% names(catboost_configs))
  config <- catboost_configs[[config_name]]
  frozen_iterations <- screen$best_iterations[
    match(config_name, screen$config)
  ]
  stopifnot(
    length(frozen_iterations) == 1L,
    is.finite(frozen_iterations),
    frozen_iterations >= 1L
  )

  train <- read.csv("csv files/train.csv")
  truth <- as.matrix(train[, paste0("Ch", 1:4)])
  fold_map <- canonical_fold_map()
  row_fold <- unname(fold_map[as.character(train$Case)])
  current <- readRDS(
    "data_processed/codex_behavioral_round/mlp_precision.rds"
  )$crossfit_prediction
  stopifnot(
    !anyNA(row_fold),
    abs(
      log_loss_matrix(truth, current) - 1.14378944178118
    ) < 1e-8
  )

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
    fitted <- fit_catboost(
      feature_matrix(tr),
      as.matrix(tr[, paste0("Ch", 1:4)]),
      config,
      seed = 7821L + fold,
      fixed_iterations = frozen_iterations
    )
    prediction <- predict_catboost(
      fitted$model, feature_matrix(va)
    )
    index <- match(va$No, train$No)
    stopifnot(!anyNA(index))
    oof[index, ] <- prediction
    fit_rows[[fold]] <- data.frame(
      config = config_name,
      fold = fold,
      frozen_iterations = frozen_iterations,
      fitted_tree_count = fitted$tree_count,
      elapsed_seconds = fitted$elapsed_seconds,
      fold_logloss = log_loss_matrix(
        truth[index, , drop = FALSE], prediction
      )
    )
    cat(sprintf(
      "CatBoost fold %d: %.9f\n",
      fold, fit_rows[[fold]]$fold_logloss
    ))
    flush.console()
  }
  stopifnot(
    !anyNA(oof),
    max(abs(rowSums(oof) - 1)) < 1e-10
  )

  blend <- crossfit_blend(
    truth, current, oof, row_fold
  )
  bootstrap <- bootstrap_gain(
    truth, train$Case, current, blend$prediction,
    family_size = length(catboost_configs),
    n_boot = 100000L
  )
  summary <- data.frame(
    config = config_name,
    frozen_iterations = frozen_iterations,
    component_logloss = log_loss_matrix(truth, oof),
    current_logloss = log_loss_matrix(truth, current),
    blend_logloss = blend$logloss,
    gain_vs_current =
      log_loss_matrix(truth, current) - blend$logloss
  )
  write.csv(
    summary,
    file.path(output_dir, "catboost_cv.csv"),
    row.names = FALSE
  )
  write.csv(
    do.call(rbind, fit_rows),
    file.path(output_dir, "catboost_cv_fits.csv"),
    row.names = FALSE
  )
  write.csv(
    data.frame(fold = 1:5, catboost_weight = blend$weights),
    file.path(output_dir, "catboost_cv_weights.csv"),
    row.names = FALSE
  )
  write.csv(
    bootstrap$summary,
    file.path(output_dir, "catboost_cv_bootstrap.csv"),
    row.names = FALSE
  )
  saveRDS(
    list(
      package_version = as.character(packageVersion("catboost")),
      config = config_name,
      specification = config,
      frozen_iterations = frozen_iterations,
      oof = oof,
      blend = blend,
      summary = summary,
      bootstrap = bootstrap
    ),
    file.path(output_dir, "catboost_oof.rds")
  )
  print(summary, digits = 9)
  print(bootstrap$summary, digits = 9)
  print(data.frame(fold = 1:5, weight = blend$weights))
}
