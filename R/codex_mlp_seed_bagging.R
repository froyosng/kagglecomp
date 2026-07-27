# Targeted seed-bagging follow-up for the frozen MLP candidate.
#
# Everything except the number of averaged random initializations is held
# fixed: 8 hidden units, decay 0.1, 200 iterations, the canonical respondent
# folds, and the fold-cross-fitted blend-weight selection. Each fit is
# checkpointed separately so the 100-fit run can resume safely.

options(stringsAsFactors = FALSE)

old_stage <- Sys.getenv("CODEX_STAGE", unset = NA_character_)
Sys.setenv(CODEX_STAGE = "define")
source("R/codex_mlp_ensemble.R")
if (is.na(old_stage)) {
  Sys.unsetenv("CODEX_STAGE")
} else {
  Sys.setenv(CODEX_STAGE = old_stage)
}

output_dir <- "data_processed/codex_mlp_seed_bagging"
checkpoint_dir <- file.path(output_dir, "checkpoints")
dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)

n_seeds <- 20L
seed_counts <- seq_len(n_seeds)
weight_grid <- seq(0, 0.30, by = 0.01)

train <- read.csv("csv files/train.csv")
truth <- as.matrix(train[, paste0("Ch", 1:4)])
reference <- feature_reference(train)
saved_base <- readRDS("data_processed/oof_ensemble_v10.rds")
fold_map <- saved_base$fold_of_case
row_fold <- unname(fold_map[as.character(train$Case)])
baseline <- saved_base$oof_mlogit
xgb <- saved_base$oof_xgb
v11 <- 0.8 * baseline + 0.2 * xgb

stopifnot(
  !anyNA(row_fold),
  all(row_fold %in% 1:5),
  abs(log_loss_matrix(truth, baseline) - 1.1470212110518) < 1e-8,
  abs(log_loss_matrix(truth, v11) - 1.14509421298673) < 1e-10
)

oof_by_count <- lapply(
  seed_counts,
  function(x) matrix(NA_real_, nrow(train), 4L)
)
fit_rows <- list()

for (fold in 1:5) {
  validation_cases <- as.integer(
    names(fold_map)[fold_map == fold]
  )
  train_rows <- !(train$Case %in% validation_cases)
  validation_rows <- train$Case %in% validation_cases
  tr <- train[train_rows, , drop = FALSE]
  va <- train[validation_rows, , drop = FALSE]
  validation_index <- match(va$No, train$No)
  stopifnot(!anyNA(validation_index))

  scaler <- continuous_scaler(tr)
  train_matrix <- build_mlp_matrix(tr, reference, scaler)
  validation_matrix <- build_mlp_matrix(
    va,
    reference,
    scaler,
    keep_columns = train_matrix$keep_columns
  )
  train_truth <- as.matrix(tr[, paste0("Ch", 1:4)])
  fold_prediction <- array(
    NA_real_,
    dim = c(nrow(va), 4L, n_seeds)
  )

  for (seed_index in seed_counts) {
    seed <- 4821L + fold * 100L + seed_index - 1L
    checkpoint_path <- file.path(
      checkpoint_dir,
      sprintf("fold_%d_seed_%d.rds", fold, seed)
    )

    if (file.exists(checkpoint_path)) {
      checkpoint <- readRDS(checkpoint_path)
      stopifnot(
        identical(checkpoint$fold, fold),
        identical(checkpoint$seed, seed),
        identical(
          checkpoint$validation_index,
          validation_index
        ),
        identical(
          checkpoint$feature_columns,
          train_matrix$keep_columns
        ),
        identical(dim(checkpoint$prediction), c(nrow(va), 4L)),
        !anyNA(checkpoint$prediction),
        max(abs(rowSums(checkpoint$prediction) - 1)) < 1e-10
      )
      prediction <- checkpoint$prediction
      fit_summary <- checkpoint$fit_summary
      source_status <- "checkpoint"
    } else {
      fitted <- fit_mlp_average(
        train_matrix$x,
        train_truth,
        validation_matrix$x,
        size = 8L,
        decay = 0.1,
        seeds = seed,
        max_iterations = 200L
      )
      prediction <- fitted$pred
      fit_summary <- fitted$fits
      checkpoint <- list(
        fold = fold,
        seed = seed,
        validation_index = validation_index,
        prediction = prediction,
        fit_summary = fit_summary,
        feature_columns = train_matrix$keep_columns,
        specification = list(
          size = 8L,
          decay = 0.1,
          max_iterations = 200L
        )
      )
      saveRDS(checkpoint, checkpoint_path)
      source_status <- "fitted"
    }

    stopifnot(
      !anyNA(prediction),
      all(prediction > 0),
      max(abs(rowSums(prediction) - 1)) < 1e-10
    )
    fold_prediction[, , seed_index] <- prediction
    fit_rows[[length(fit_rows) + 1L]] <- data.frame(
      fold = fold,
      seed_index = seed_index,
      seed = seed,
      convergence = fit_summary$convergence,
      training_objective = fit_summary$training_objective,
      elapsed_seconds = fit_summary$elapsed_seconds,
      source_status = source_status
    )
    cat(sprintf(
      "fold %d seed %02d/%02d (%d) complete [%s]\n",
      fold, seed_index, n_seeds, seed, source_status
    ))
    flush.console()
  }

  running_sum <- matrix(0, nrow(va), 4L)
  for (seed_index in seed_counts) {
    running_sum <- running_sum +
      fold_prediction[, , seed_index]
    cumulative_average <- running_sum / seed_index
    oof_by_count[[seed_index]][validation_index, ] <-
      cumulative_average
  }
  saveRDS(
    list(
      fold = fold,
      seeds = 4821L + fold * 100L + 0:(n_seeds - 1L),
      validation_index = validation_index,
      cumulative_prediction = lapply(
        seed_counts,
        function(seed_index) {
          oof_by_count[[seed_index]][
            validation_index, , drop = FALSE
          ]
        }
      )
    ),
    file.path(output_dir, sprintf("fold_%d_cumulative.rds", fold))
  )
  cat(sprintf("fold %d cumulative predictions saved\n", fold))
  flush.console()
}

stopifnot(all(vapply(
  oof_by_count,
  function(prediction) {
    !anyNA(prediction) &&
      all(prediction > 0) &&
      max(abs(rowSums(prediction) - 1)) < 1e-10
  },
  logical(1)
)))

# Reproduce the existing five-seed candidate before accepting the extension.
old_mlp_saved <- readRDS(
  "data_processed/codex_behavioral_round/mlp_oof.rds"
)
old_mlp <- old_mlp_saved$oof[["h08_d0.100"]]
first_five_max_abs_difference <- max(
  abs(oof_by_count[[5L]] - old_mlp)
)
stopifnot(
  first_five_max_abs_difference < 1e-12,
  abs(log_loss_matrix(truth, oof_by_count[[5L]]) -
    1.19054334979533) < 1e-10
)

crossfit_blend <- function(component_prediction) {
  prediction <- matrix(NA_real_, nrow(train), 4L)
  fold_weights <- numeric(5L)
  tuning_loss <- numeric(5L)
  for (fold in 1:5) {
    tuning_rows <- row_fold != fold
    validation_rows <- row_fold == fold
    losses <- vapply(weight_grid, function(weight) {
      log_loss_matrix(
        truth[tuning_rows, , drop = FALSE],
        (1 - weight) * v11[tuning_rows, , drop = FALSE] +
          weight * component_prediction[
            tuning_rows, , drop = FALSE
          ]
      )
    }, numeric(1))
    best <- which.min(losses)
    fold_weights[[fold]] <- weight_grid[[best]]
    tuning_loss[[fold]] <- losses[[best]]
    prediction[validation_rows, ] <-
      (1 - fold_weights[[fold]]) *
        v11[validation_rows, , drop = FALSE] +
      fold_weights[[fold]] *
        component_prediction[validation_rows, , drop = FALSE]
  }
  stopifnot(!anyNA(prediction))
  list(
    prediction = prediction,
    weights = fold_weights,
    tuning_loss = tuning_loss,
    logloss = log_loss_matrix(truth, prediction)
  )
}

blend_by_count <- lapply(oof_by_count, crossfit_blend)
stopifnot(
  abs(blend_by_count[[5L]]$logloss -
    1.14378944178118) < 1e-10
)

learning_curve <- do.call(rbind, lapply(seed_counts, function(seed_count) {
  component <- oof_by_count[[seed_count]]
  global_losses <- vapply(weight_grid, function(weight) {
    log_loss_matrix(
      truth,
      (1 - weight) * v11 + weight * component
    )
  }, numeric(1))
  best_global <- which.min(global_losses)
  data.frame(
    n_seeds = seed_count,
    component_logloss = log_loss_matrix(truth, component),
    global_best_weight = weight_grid[[best_global]],
    global_blend_logloss = global_losses[[best_global]],
    global_optimistic_gain =
      log_loss_matrix(truth, v11) - global_losses[[best_global]],
    crossfit_blend_logloss =
      blend_by_count[[seed_count]]$logloss,
    crossfit_blend_gain =
      log_loss_matrix(truth, v11) -
      blend_by_count[[seed_count]]$logloss
  )
}))

fold_weights <- do.call(rbind, lapply(seed_counts, function(seed_count) {
  data.frame(
    n_seeds = seed_count,
    fold = 1:5,
    mlp_weight = blend_by_count[[seed_count]]$weights,
    tuning_logloss = blend_by_count[[seed_count]]$tuning_loss
  )
}))
fit_summary <- do.call(rbind, fit_rows)

write.csv(
  learning_curve,
  file.path(output_dir, "mlp_seed_learning_curve.csv"),
  row.names = FALSE
)
write.csv(
  fold_weights,
  file.path(output_dir, "mlp_seed_fold_weights.csv"),
  row.names = FALSE
)
write.csv(
  fit_summary,
  file.path(output_dir, "mlp_seed_fit_summary.csv"),
  row.names = FALSE
)
saveRDS(
  list(
    specification = list(
      size = 8L,
      decay = 0.1,
      max_iterations = 200L,
      n_seeds = n_seeds,
      seed_rule = "4821 + fold * 100 + seed_index - 1"
    ),
    oof_by_count = oof_by_count,
    blend_by_count = blend_by_count,
    learning_curve = learning_curve,
    fold_weights = fold_weights,
    first_five_max_abs_difference =
      first_five_max_abs_difference
  ),
  file.path(output_dir, "mlp_seed_bagging_oof.rds")
)

cat("\nSelected learning-curve checkpoints:\n")
print(
  learning_curve[
    learning_curve$n_seeds %in% c(1L, 5L, 10L, 15L, 20L),
    ,
    drop = FALSE
  ],
  digits = 9
)
cat(sprintf(
  "\nFirst-five maximum absolute difference vs saved OOF: %.3g\n",
  first_five_max_abs_difference
))
