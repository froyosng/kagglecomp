# Screen: does adding a variance-across-alternatives pooling statistic to the
# set-context network beat the existing mean+max-only architecture, on the
# project's canonical single 80/20 split? Cheap first look before committing
# to the full nested-CV protocol (torch retraining is expensive).
#
# Pre-registration: codex_variance_pool_preregister.md
#
# Run:
#   source("R/codex_variance_pool_screen.R")

suppressPackageStartupMessages(library(torch))

## ---- load every definition from the validated v2 runner without executing
## its terminal run_experiment() call (same pattern codex_set_context_
## candidate_submission.R already uses) ----
runner_path <- "R/codex_set_context_network_v2.R"
runner_lines <- readLines(runner_path, warn = FALSE)
nonblank <- which(nzchar(trimws(runner_lines)))
terminal_line <- tail(nonblank, 1L)
stopifnot(trimws(runner_lines[[terminal_line]]) == "run_experiment()")
v2 <- new.env(parent = globalenv())
eval(parse(text = runner_lines[-terminal_line], keep.source = TRUE), envir = v2)

output_dir <- file.path("data_processed", "codex_variance_pool_screen")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

screen_seed <- 22501L
screen_config <- v2$network_config # identical hyperparameters, unchanged

## ---- new architecture: adds a variance-across-alternatives pooling term,
## broadcast the same way mean_repeated/max_repeated already are. No other
## change: same encoder_hidden, embedding_dim, head_hidden, dropout, lr. ----
set_context_net_variance <- nn_module(
  "set_context_net_variance",
  initialize = function(profile_dim, common_dim, config) {
    self$profile_dim <- as.integer(profile_dim)
    self$common_dim <- as.integer(common_dim)
    self$embedding_dim <- as.integer(config$embedding_dim)
    self$encoder <- nn_sequential(
      nn_linear(self$profile_dim, as.integer(config$encoder_hidden)),
      nn_relu(),
      nn_dropout(p = config$encoder_dropout),
      nn_linear(as.integer(config$encoder_hidden), self$embedding_dim),
      nn_relu()
    )
    head_input_dim <- 5L * self$embedding_dim + self$common_dim
    self$head_input_dim <- as.integer(head_input_dim)
    self$head <- nn_sequential(
      nn_linear(self$head_input_dim, as.integer(config$head_hidden[[1L]])),
      nn_relu(),
      nn_dropout(p = config$head_dropout),
      nn_linear(as.integer(config$head_hidden[[1L]]), as.integer(config$head_hidden[[2L]])),
      nn_relu(),
      nn_linear(as.integer(config$head_hidden[[2L]]), 1L)
    )
  },
  forward = function(profile, common) {
    n_tasks <- profile$size(1L)
    embedding <- self$encoder(
      profile$reshape(c(-1L, self$profile_dim))
    )$reshape(c(n_tasks, 4L, self$embedding_dim))
    inside_embedding <- embedding[, 1:3, ]
    mean_inside <- inside_embedding$mean(dim = 2L)
    max_inside <- torch_maximum(
      torch_maximum(inside_embedding[, 1L, ], inside_embedding[, 2L, ]),
      inside_embedding[, 3L, ]
    )
    # Population variance across the 3 inside alternatives, per embedding
    # coordinate -- the pooling statistic missing from the existing
    # mean+max architecture (verified by direct code read: neither mean nor
    # elementwise max preserves any spread/similarity information across
    # alternatives). Small epsilon avoids a zero-variance NaN in backprop.
    mean_repeated_inside <- torch_stack(list(mean_inside, mean_inside, mean_inside), dim = 2L)
    squared_dev <- (inside_embedding - mean_repeated_inside)$pow(2)
    var_inside <- squared_dev$mean(dim = 2L) + 1e-8

    mean_repeated <- torch_stack(list(mean_inside, mean_inside, mean_inside, mean_inside), dim = 2L)
    max_repeated <- torch_stack(list(max_inside, max_inside, max_inside, max_inside), dim = 2L)
    var_repeated <- torch_stack(list(var_inside, var_inside, var_inside, var_inside), dim = 2L)
    common_repeated <- torch_stack(list(common, common, common, common), dim = 2L)

    combined <- torch_cat(
      list(embedding, mean_repeated, max_repeated, var_repeated, embedding - mean_repeated, common_repeated),
      dim = 3L
    )
    self$head(
      combined$reshape(c(-1L, self$head_input_dim))
    )$reshape(c(n_tasks, 4L))
  }
)

## ---- fit/predict helpers, identical to v2's except for the model class ----

fit_variance_net_once <- function(train_features, validation_features, config, seed) {
  set.seed(as.integer(seed))
  torch_manual_seed(as.integer(seed))
  n_tasks <- nrow(train_features$common)
  profile_tensor <- torch_tensor(train_features$profile, dtype = torch_float())$reshape(
    c(n_tasks, 4L, ncol(train_features$profile))
  )
  common_tensor <- torch_tensor(train_features$common, dtype = torch_float())
  target_tensor <- torch_tensor(as.integer(max.col(train_features$truth)), dtype = torch_long())
  model <- set_context_net_variance(
    profile_dim = ncol(train_features$profile), common_dim = ncol(train_features$common), config = config
  )
  optimizer <- optim_adam(model$parameters, lr = config$learning_rate, weight_decay = config$weight_decay)
  started <- proc.time()[["elapsed"]]
  for (epoch in seq_len(config$epochs)) {
    model$train()
    task_order <- sample.int(n_tasks)
    total_loss <- 0
    total_tasks <- 0L
    for (start in seq.int(1L, n_tasks, by = config$batch_tasks)) {
      stop_at <- min(n_tasks, start + config$batch_tasks - 1L)
      task_index <- task_order[start:stop_at]
      optimizer$zero_grad()
      utility <- model(profile_tensor[task_index, , ], common_tensor[task_index, ])
      loss <- nnf_cross_entropy(utility, target_tensor[task_index])
      loss$backward()
      optimizer$step()
      count <- length(task_index)
      total_loss <- total_loss + loss$item() * count
      total_tasks <- total_tasks + count
    }
  }
  elapsed <- proc.time()[["elapsed"]] - started
  prediction <- v2$predict_set_context(model, validation_features$profile, validation_features$common)
  list(
    prediction = prediction, seed = as.integer(seed), elapsed_seconds = elapsed,
    final_training_logloss = total_loss / total_tasks
  )
}

## ---- run the screen: fit BOTH architectures on the identical split ----

split <- readRDS("data_processed/train_val_split.rds")
fitting <- split$train_wide_tr
validation <- split$train_wide_val
stopifnot(length(intersect(unique(fitting$Case), unique(validation$Case))) == 0L)

fitting_long <- v2$add_choice_context(v2$build_full_long(fitting))
validation_long <- v2$add_choice_context(v2$build_full_long(validation))
levels_by_attribute <- v2$attribute_levels(fitting_long)
profile_scaler <- v2$fit_scaler(fitting_long, v2$profile_continuous)
common_scaler <- v2$fit_scaler(fitting_long, v2$common_continuous)
training_features <- v2$build_set_context_features(fitting_long, levels_by_attribute, profile_scaler, common_scaler)
validation_features <- v2$build_set_context_features(validation_long, levels_by_attribute, profile_scaler, common_scaler)

truth_val <- validation_features$truth

cat("Fitting EXISTING (mean+max) architecture...\n")
existing_fit <- v2$fit_set_context_once(training_features, validation_features, screen_config, screen_seed)
cat(sprintf(
  "  existing: train logloss %.6f, val logloss %.6f, %.1fs\n",
  existing_fit$final_training_logloss,
  v2$log_loss_matrix_local(truth_val, existing_fit$prediction),
  existing_fit$elapsed_seconds
))

cat("Fitting NEW (mean+max+variance) architecture...\n")
variance_fit <- fit_variance_net_once(training_features, validation_features, screen_config, screen_seed)
cat(sprintf(
  "  variance: train logloss %.6f, val logloss %.6f, %.1fs\n",
  variance_fit$final_training_logloss,
  v2$log_loss_matrix_local(truth_val, variance_fit$prediction),
  variance_fit$elapsed_seconds
))

## ---- blend each candidate component against the frozen flat baseline on
## the same validation split (screen-stage comparison only) ----
saved <- readRDS("data_processed/oof_ensemble_v10.rds")
train_full <- read.csv("csv files/train.csv")
train_full <- train_full[order(train_full$No), , drop = FALSE]
rownames(train_full) <- NULL
truth_full <- as.matrix(train_full[, paste0("Ch", 1:4), drop = FALSE])
baseline_full <- v2$canonical_baseline(train_full, truth_full, saved)
val_rows <- match(validation$No, train_full$No)
stopifnot(!anyNA(val_rows))
baseline_val <- baseline_full$prediction[val_rows, , drop = FALSE]
baseline_val_loss <- v2$log_loss_matrix_local(truth_val, baseline_val)
cat(sprintf("\nFlat baseline (mlogit+xgb+shallowMLP) val logloss: %.6f\n", baseline_val_loss))

best_blend <- function(component) {
  losses <- vapply(v2$blend_weight_grid, function(weight) {
    v2$log_loss_matrix_local(truth_val, (1 - weight) * baseline_val + weight * component)
  }, numeric(1))
  best_index <- which.min(losses)
  list(weight = v2$blend_weight_grid[[best_index]], logloss = losses[[best_index]])
}

existing_blend <- best_blend(existing_fit$prediction)
variance_blend <- best_blend(variance_fit$prediction)

cat(sprintf(
  "Existing architecture: best blend weight %.2f, blended val logloss %.6f\n",
  existing_blend$weight, existing_blend$logloss
))
cat(sprintf(
  "Variance architecture: best blend weight %.2f, blended val logloss %.6f\n",
  variance_blend$weight, variance_blend$logloss
))
cat(sprintf(
  "\nScreen gain (existing - variance, blended val logloss): %.6f\n",
  existing_blend$logloss - variance_blend$logloss
))

result <- list(
  existing_fit = existing_fit, variance_fit = variance_fit,
  baseline_val_loss = baseline_val_loss,
  existing_blend = existing_blend, variance_blend = variance_blend
)
saveRDS(result, file.path(output_dir, "screen_result.rds"))
write.csv(
  data.frame(
    architecture = c("existing_mean_max", "variance_pool"),
    standalone_val_logloss = c(
      v2$log_loss_matrix_local(truth_val, existing_fit$prediction),
      v2$log_loss_matrix_local(truth_val, variance_fit$prediction)
    ),
    best_blend_weight = c(existing_blend$weight, variance_blend$weight),
    best_blend_val_logloss = c(existing_blend$logloss, variance_blend$logloss)
  ),
  file.path(output_dir, "screen_summary.csv"),
  row.names = FALSE
)
cat("\nScreen complete. Results in", output_dir, "\n")
