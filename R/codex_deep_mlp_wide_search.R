## Pre-registered 24-configuration deep-MLP search.
## See codex_deep_mlp_wide_preregister.md for the registry and decision rule.

old_deep_stage <- Sys.getenv(
  "CODEX_DEEP_STAGE", unset = NA_character_
)
Sys.setenv(CODEX_DEEP_STAGE = "define")
source("R/codex_torch_deep_mlp.R")
if (is.na(old_deep_stage)) {
  Sys.unsetenv("CODEX_DEEP_STAGE")
} else {
  Sys.setenv(CODEX_DEEP_STAGE = old_deep_stage)
}

wide_stage <- Sys.getenv("CODEX_DEEP_WIDE_STAGE", "screen")
stopifnot(wide_stage %in% c("screen", "cv"))

wide_output_dir <- "data_processed/codex_overnight_queue"
wide_checkpoint_dir <- file.path(
  wide_output_dir, "deep_wide_checkpoints"
)
dir.create(
  wide_checkpoint_dir, recursive = TRUE, showWarnings = FALSE
)

hidden_layouts <- list(
  h032_016 = c(32L, 16L),
  h064_032 = c(64L, 32L),
  h128_064 = c(128L, 64L),
  h256_128 = c(256L, 128L),
  h128_064_032 = c(128L, 64L, 32L),
  h256_128_064 = c(256L, 128L, 64L)
)
training_recipes <- list(
  A = list(
    dropout = 0.15,
    weight_decay = 1e-4,
    learning_rate = 5e-4,
    epochs = 16L
  ),
  B = list(
    dropout = 0.30,
    weight_decay = 1e-3,
    learning_rate = 1e-3,
    epochs = 12L
  ),
  C = list(
    dropout = 0.45,
    weight_decay = 2e-3,
    learning_rate = 1e-3,
    epochs = 12L
  ),
  D = list(
    dropout = 0.30,
    weight_decay = 5e-4,
    learning_rate = 2e-3,
    epochs = 8L
  )
)
wide_configs <- list()
registry_rows <- list()
for (layout_name in names(hidden_layouts)) {
  for (recipe_name in names(training_recipes)) {
    config_name <- paste(layout_name, recipe_name, sep = "__")
    recipe <- training_recipes[[recipe_name]]
    wide_configs[[config_name]] <- c(
      list(
        hidden = hidden_layouts[[layout_name]],
        batch_size = 256L
      ),
      recipe
    )
    registry_rows[[config_name]] <- data.frame(
      config = config_name,
      layout = layout_name,
      hidden = paste(
        hidden_layouts[[layout_name]], collapse = "-"
      ),
      recipe = recipe_name,
      dropout = recipe$dropout,
      weight_decay = recipe$weight_decay,
      learning_rate = recipe$learning_rate,
      epochs = recipe$epochs,
      batch_size = 256L,
      stringsAsFactors = FALSE
    )
  }
}
wide_registry <- do.call(rbind, registry_rows)
wide_family_size <- nrow(wide_registry)
stopifnot(
  wide_family_size == 24L,
  "h128_064__B" %in% names(wide_configs)
)
write.csv(
  wide_registry,
  file.path(wide_output_dir, "deep_wide_registry.csv"),
  row.names = FALSE
)

bootstrap_wide_primary <- function(truth, baseline, candidate,
                                   case, n_boot = 100000L,
                                   seed = 4821L) {
  row_loss <- function(prediction) {
    prediction <- prediction / rowSums(prediction)
    -rowSums(truth * log(pmax(prediction, 1e-15)))
  }
  case_gain <- unname(tapply(
    row_loss(baseline) - row_loss(candidate),
    case, mean
  ))
  set.seed(seed)
  n_case <- length(case_gain)
  bootstrap <- numeric(n_boot)
  for (start in seq.int(1L, n_boot, by = 1000L)) {
    stop_at <- min(n_boot, start + 999L)
    n_this <- stop_at - start + 1L
    sampled <- matrix(
      sample.int(
        n_case, n_case * n_this, replace = TRUE
      ),
      nrow = n_case
    )
    bootstrap[start:stop_at] <- colMeans(matrix(
      case_gain[sampled], nrow = n_case
    ))
  }
  family_alpha <- 0.05 / wide_family_size
  data.frame(
    point_gain = mean(case_gain),
    bootstrap_mean = mean(bootstrap),
    bootstrap_sd = sd(bootstrap),
    lower_95 = unname(quantile(bootstrap, 0.025)),
    upper_95 = unname(quantile(bootstrap, 0.975)),
    lower_99 = unname(quantile(bootstrap, 0.005)),
    upper_99 = unname(quantile(bootstrap, 0.995)),
    lower_bonferroni_24 = unname(quantile(
      bootstrap, family_alpha / 2
    )),
    upper_bonferroni_24 = unname(quantile(
      bootstrap, 1 - family_alpha / 2
    )),
    win_rate = mean(bootstrap > 0),
    n_boot = n_boot,
    family_size = wide_family_size
  )
}

if (wide_stage == "screen") {
  split_data <- readRDS(
    "data_processed/train_val_split.rds"
  )
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
  x_train <- train_matrix$x[
    train_order, , drop = FALSE
  ]
  x_valid <- validation_matrix$x[
    validation_order, , drop = FALSE
  ]
  y_train <- as.matrix(
    split_data$train_wide_tr[
      train_order, paste0("Ch", 1:4), drop = FALSE
    ]
  )
  target_no <- split_data$train_wide_val$No[
    validation_order
  ]
  mlp_screen <- readRDS(
    "data_processed/codex_behavioral_round/mlp_screen.rds"
  )
  stopifnot(identical(
    as.integer(mlp_screen$validation_no),
    as.integer(target_no)
  ))
  truth <- mlp_screen$truth
  v11 <- mlp_screen$v11_screen
  shallow <- mlp_screen$predictions[["h08_d0.100"]]
  current <- 0.85 * v11 + 0.15 * shallow
  current_loss <- log_loss_matrix(truth, current)
  screen_seeds <- c(13101L, 13102L)

  rows <- list()
  fit_rows <- list()
  prediction_list <- list()
  trace_list <- list()
  for (config_name in names(wide_configs)) {
    config <- wide_configs[[config_name]]
    fitted <- fit_torch_average(
      x_train, y_train, x_valid, config,
      seeds = screen_seeds,
      checkpoint_prefix = file.path(
        wide_checkpoint_dir,
        paste0("screen_", config_name)
      )
    )
    deep <- fitted$prediction
    components <- list(
      v11 = v11, shallow = shallow, deep = deep
    )
    weights <- fit_arithmetic_weights(
      truth, components, seq_len(nrow(truth))
    )
    names(weights) <- names(components)
    joint <- blend_components(components, weights)
    rows[[config_name]] <- cbind(
      wide_registry[
        wide_registry$config == config_name,
        ,
        drop = FALSE
      ],
      data.frame(
        n_seeds = length(screen_seeds),
        n_features = ncol(x_train),
        component_loss = log_loss_matrix(truth, deep),
        current_loss = current_loss,
        joint_loss = log_loss_matrix(truth, joint),
        gain = current_loss - log_loss_matrix(truth, joint),
        weight_v11 = weights[["v11"]],
        weight_shallow = weights[["shallow"]],
        weight_deep = weights[["deep"]]
      )
    )
    fit_rows[[config_name]] <- cbind(
      data.frame(config = config_name),
      fitted$fits
    )
    prediction_list[[config_name]] <- deep
    trace_list[[config_name]] <- fitted$traces
    cat(sprintf(
      "%s: component %.9f; joint %.9f; gain %+.9f\n",
      config_name,
      log_loss_matrix(truth, deep),
      log_loss_matrix(truth, joint),
      current_loss - log_loss_matrix(truth, joint)
    ))
    flush.console()
    invisible(gc())
  }
  result <- do.call(rbind, rows)
  best_index <- which.min(result$joint_loss)
  frozen_config <- result$config[[best_index]]
  passed_screen <- result$gain[[best_index]] > 0
  selection <- data.frame(
    frozen_config = frozen_config,
    passed_screen = passed_screen,
    family_size = wide_family_size,
    selection_metric = "three_way_joint_logloss",
    screen_gain = result$gain[[best_index]]
  )
  write.csv(
    result,
    file.path(wide_output_dir, "deep_wide_screen.csv"),
    row.names = FALSE
  )
  write.csv(
    do.call(rbind, fit_rows),
    file.path(wide_output_dir, "deep_wide_screen_fits.csv"),
    row.names = FALSE
  )
  write.csv(
    selection,
    file.path(wide_output_dir, "deep_wide_selection.csv"),
    row.names = FALSE
  )
  saveRDS(
    list(
      registry = wide_registry,
      result = result,
      selection = selection,
      predictions = prediction_list,
      traces = trace_list,
      truth = truth,
      v11 = v11,
      shallow = shallow,
      current = current,
      no = target_no,
      feature_columns = train_matrix$keep_columns
    ),
    file.path(wide_output_dir, "deep_wide_screen.rds")
  )
  print(result[order(result$joint_loss), ], digits = 9)
  print(selection, digits = 9)
}

if (wide_stage == "cv") {
  selection <- read.csv(file.path(
    wide_output_dir, "deep_wide_selection.csv"
  ))
  if (!isTRUE(selection$passed_screen[[1]])) {
    cat("No deep-wide configuration passed the screen.\n")
    quit(save = "no", status = 0)
  }
  frozen_config <- selection$frozen_config[[1]]
  stopifnot(frozen_config %in% names(wide_configs))
  config <- wide_configs[[frozen_config]]

  train <- read.csv("csv files/train.csv")
  train <- train[order(train$No), , drop = FALSE]
  truth <- as.matrix(
    train[, paste0("Ch", 1:4), drop = FALSE]
  )
  reference <- feature_reference(train)
  base <- readRDS("data_processed/oof_ensemble_v10.rds")
  fold_map <- base$fold_of_case
  row_fold <- unname(
    fold_map[as.character(train$Case)]
  )
  v11 <- 0.8 * base$oof_mlogit + 0.2 * base$oof_xgb
  shallow <- readRDS(
    "data_processed/codex_behavioral_round/mlp_oof.rds"
  )$oof[["h08_d0.100"]]
  current <- 0.85 * v11 + 0.15 * shallow
  stopifnot(abs(
    log_loss_matrix(truth, current) -
      1.143686618134879
  ) < 1e-10)

  deep_oof <- matrix(NA_real_, nrow(train), 4L)
  fit_rows <- list()
  cv_seeds <- c(13201L, 13202L, 13203L)
  for (fold in 1:5) {
    validation_cases <- as.integer(
      names(fold_map)[fold_map == fold]
    )
    source_rows <- !(train$Case %in% validation_cases)
    target_rows <- train$Case %in% validation_cases
    source <- train[source_rows, , drop = FALSE]
    target <- train[target_rows, , drop = FALSE]
    scaler <- continuous_scaler(source)
    source_matrix <- build_mlp_matrix(
      source, reference, scaler
    )
    target_matrix <- build_mlp_matrix(
      target, reference, scaler,
      keep_columns = source_matrix$keep_columns
    )
    fitted <- fit_torch_average(
      source_matrix$x,
      as.matrix(
        source[, paste0("Ch", 1:4), drop = FALSE]
      ),
      target_matrix$x,
      config,
      seeds = cv_seeds + fold * 100L,
      checkpoint_prefix = file.path(
        wide_checkpoint_dir,
        sprintf(
          "cv_%s_fold_%d", frozen_config, fold
        )
      )
    )
    deep_oof[target_rows, ] <- fitted$prediction
    fit_rows[[as.character(fold)]] <- cbind(
      data.frame(config = frozen_config, fold = fold),
      fitted$fits
    )
    invisible(gc())
  }
  stopifnot(
    !anyNA(deep_oof),
    max(abs(rowSums(deep_oof) - 1)) < 1e-6
  )

  primary <- crossfit_three_way(
    truth, v11, shallow, deep_oof, row_fold
  )
  incremental <- crossfit_two_way(
    truth, current, deep_oof, row_fold,
    weight_grid = seq(0, 0.4, by = 0.01)
  )
  fold_rows <- do.call(rbind, lapply(1:5, function(fold) {
    rows <- row_fold == fold
    data.frame(
      fold = fold,
      baseline_loss = log_loss_matrix(
        truth[rows, , drop = FALSE],
        current[rows, , drop = FALSE]
      ),
      primary_loss = log_loss_matrix(
        truth[rows, , drop = FALSE],
        primary$prediction[rows, , drop = FALSE]
      )
    )
  }))
  fold_rows$gain <- with(
    fold_rows, baseline_loss - primary_loss
  )
  bootstrap <- bootstrap_wide_primary(
    truth, current, primary$prediction, train$Case
  )
  summary <- data.frame(
    frozen_config = frozen_config,
    family_size = wide_family_size,
    component_loss = log_loss_matrix(truth, deep_oof),
    baseline_loss = log_loss_matrix(truth, current),
    primary_joint_loss = primary$logloss,
    primary_gain =
      log_loss_matrix(truth, current) - primary$logloss,
    folds_improved = sum(fold_rows$gain > 0),
    worst_fold_gain = min(fold_rows$gain),
    incremental_loss = incremental$logloss,
    incremental_gain =
      log_loss_matrix(truth, current) - incremental$logloss
  )
  primary_weights <- do.call(rbind, lapply(1:5, function(fold) {
    data.frame(
      fold = fold,
      component = colnames(primary$weights),
      weight = as.numeric(primary$weights[fold, ])
    )
  }))
  write.csv(
    summary,
    file.path(wide_output_dir, "deep_wide_cv.csv"),
    row.names = FALSE
  )
  write.csv(
    fold_rows,
    file.path(wide_output_dir, "deep_wide_cv_folds.csv"),
    row.names = FALSE
  )
  write.csv(
    primary_weights,
    file.path(wide_output_dir, "deep_wide_cv_weights.csv"),
    row.names = FALSE
  )
  write.csv(
    do.call(rbind, fit_rows),
    file.path(wide_output_dir, "deep_wide_cv_fits.csv"),
    row.names = FALSE
  )
  write.csv(
    bootstrap,
    file.path(wide_output_dir, "deep_wide_bootstrap.csv"),
    row.names = FALSE
  )
  saveRDS(
    list(
      frozen_config = frozen_config,
      specification = config,
      family_size = wide_family_size,
      deep_oof = deep_oof,
      baseline = current,
      primary = primary,
      incremental = incremental,
      summary = summary,
      fold_results = fold_rows,
      bootstrap = bootstrap,
      truth = truth,
      case = train$Case,
      no = train$No,
      row_fold = row_fold
    ),
    file.path(wide_output_dir, "deep_wide_cv.rds")
  )
  print(summary, digits = 10)
  print(fold_rows, digits = 10)
  print(bootstrap, digits = 10)
  print(primary_weights, digits = 8)
}
