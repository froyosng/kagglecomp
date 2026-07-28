# Repeated respondent-grouped CV for the two frozen near-miss candidates.
# See codex_repeated_cv_preregister.md.

source("R/codex_repeat_cv_common.R")

train <- read.csv("csv files/train.csv")
train <- train[order(train$No), , drop = FALSE]
truth <- as.matrix(
  train[, paste0("Ch", 1:4), drop = FALSE]
)
cases <- unique(train$Case)
stopifnot(
  identical(train$No, seq_len(nrow(train))),
  length(cases) == 1135L,
  all(table(train$Case) == 19L)
)
full_long <- reshape_choice_long(train)
reference <- feature_reference(train)
attr_max <- as.list(vapply(
  full_long[, attrs, drop = FALSE], max, numeric(1)
))

save_fold_state <- function(state, path) {
  saveRDS(state, path)
  invisible(gc())
}

run_repeat_fold <- function(repeat_seed, repeat_index, fold,
                            fold_map) {
  validation_cases <- as.integer(
    names(fold_map)[fold_map == fold]
  )
  train_rows <- !(train$Case %in% validation_cases)
  validation_rows <- train$Case %in% validation_cases
  source_wide <- train[train_rows, , drop = FALSE]
  target_wide <- train[validation_rows, , drop = FALSE]
  source_long <- full_long[
    !(full_long$Case %in% validation_cases),
    ,
    drop = FALSE
  ]
  target_long <- full_long[
    full_long$Case %in% validation_cases,
    ,
    drop = FALSE
  ]
  target_no <- target_wide$No
  checkpoint <- file.path(
    repeat_checkpoint_dir,
    sprintf("seed_%d_fold_%d.rds", repeat_seed, fold)
  )
  state <- if (file.exists(checkpoint)) {
    readRDS(checkpoint)
  } else {
    list(
      repeat_seed = repeat_seed,
      repeat_index = repeat_index,
      fold = fold,
      validation_no = target_no,
      predictions = list(),
      fit_details = list()
    )
  }
  stopifnot(
    state$repeat_seed == repeat_seed,
    state$fold == fold,
    identical(
      as.integer(state$validation_no),
      as.integer(target_no)
    )
  )

  if (is.null(state$predictions$mlogit)) {
    started <- proc.time()[["elapsed"]]
    fitted <- fit_predict_m8trpg(
      source_long, target_long
    )
    state$predictions$mlogit <- fitted$pred[
      match(target_no, fitted$no), , drop = FALSE
    ]
    state$fit_details$mlogit_elapsed <-
      proc.time()[["elapsed"]] - started
    save_fold_state(state, checkpoint)
    cat("  mlogit complete\n")
    flush.console()
  }

  if (is.null(state$predictions$history_mlogit)) {
    started <- proc.time()[["elapsed"]]
    design_prior <- history_design_prior(source_long)
    source_history <- add_prior_smoothed_history(
      source_long, design_prior
    )
    target_history <- add_prior_smoothed_history(
      target_long, design_prior
    )
    fitted <- fit_predict_history_prior(
      source_history, target_history, "both_k3"
    )
    state$predictions$history_mlogit <- fitted$pred[
      match(target_no, fitted$no), , drop = FALSE
    ]
    state$fit_details$history_coefficient <-
      fitted$extra_coefficient
    state$fit_details$history_elapsed <-
      proc.time()[["elapsed"]] - started
    save_fold_state(state, checkpoint)
    cat("  history mlogit complete\n")
    flush.console()
  }

  if (is.null(state$predictions$triple_mlogit)) {
    started <- proc.time()[["elapsed"]]
    fitted <- fit_predict_candidate(
      source_long, target_long,
      "triple_price_income_miles"
    )
    state$predictions$triple_mlogit <- fitted$pred[
      match(target_no, fitted$no), , drop = FALSE
    ]
    state$fit_details$triple_coefficient <-
      fitted$extra_coef
    state$fit_details$triple_elapsed <-
      proc.time()[["elapsed"]] - started
    save_fold_state(state, checkpoint)
    cat("  triple mlogit complete\n")
    flush.console()
  }

  if (is.null(state$predictions$original_xgb)) {
    started <- proc.time()[["elapsed"]]
    model <- repeat_fit_original_xgb(source_wide)
    state$predictions$original_xgb <-
      repeat_predict_original_xgb(model, target_wide)
    state$fit_details$original_xgb_elapsed <-
      proc.time()[["elapsed"]] - started
    save_fold_state(state, checkpoint)
    cat("  original xgboost complete\n")
    flush.console()
  }

  if (is.null(state$predictions$retuned_xgb)) {
    started <- proc.time()[["elapsed"]]
    model <- repeat_fit_retuned_xgb(source_wide)
    state$predictions$retuned_xgb <-
      repeat_validate_prediction(
        predict_multiclass(model, target_wide),
        nrow(target_wide)
      )
    state$fit_details$retuned_xgb_elapsed <-
      proc.time()[["elapsed"]] - started
    save_fold_state(state, checkpoint)
    cat("  retuned xgboost complete\n")
    flush.console()
  }

  if (is.null(state$predictions$rank_margin)) {
    started <- proc.time()[["elapsed"]]
    model <- repeat_fit_rank(source_long)
    state$predictions$rank_margin <-
      repeat_rank_margin_matrix(model, target_long)
    stopifnot(
      identical(
        sort(unique(target_long$No)),
        as.integer(target_no)
      )
    )
    state$fit_details$rank_elapsed <-
      proc.time()[["elapsed"]] - started
    save_fold_state(state, checkpoint)
    cat("  rank:ndcg complete\n")
    flush.console()
  }

  if (is.null(state$predictions$cox)) {
    started <- proc.time()[["elapsed"]]
    fitted <- fit_cox_cv(
      source_long, attr_max,
      seed = 4821L + repeat_index * 100L + fold
    )
    state$predictions$cox <- repeat_validate_prediction(
      predict_cox_choice(
        fitted, target_long, fitted$cvfit$lambda.min
      ),
      nrow(target_wide)
    )
    state$fit_details$cox_lambda <-
      fitted$cvfit$lambda.min
    state$fit_details$cox_selected <-
      selected_interactions(
        fitted, fitted$cvfit$lambda.min
      )
    state$fit_details$cox_elapsed <-
      proc.time()[["elapsed"]] - started
    save_fold_state(state, checkpoint)
    cat("  glmnet-Cox complete\n")
    flush.console()
  }

  scaler <- continuous_scaler(source_wide)
  source_matrix <- build_mlp_matrix(
    source_wide, reference, scaler
  )
  target_matrix <- build_mlp_matrix(
    target_wide, reference, scaler,
    keep_columns = source_matrix$keep_columns
  )
  source_truth <- as.matrix(
    source_wide[, paste0("Ch", 1:4), drop = FALSE]
  )

  if (is.null(state$predictions$shallow_mlp)) {
    started <- proc.time()[["elapsed"]]
    fitted <- fit_mlp_average(
      source_matrix$x,
      source_truth,
      target_matrix$x,
      size = 8L,
      decay = 0.1,
      seeds =
        4821L + 0:4 +
        repeat_index * 10000L + fold * 100L,
      max_iterations = 200L
    )
    state$predictions$shallow_mlp <-
      fitted$pred
    state$fit_details$shallow_fits <- fitted$fits
    state$fit_details$shallow_elapsed <-
      proc.time()[["elapsed"]] - started
    save_fold_state(state, checkpoint)
    cat("  shallow MLP complete\n")
    flush.console()
  }

  if (is.null(state$predictions$deep_mlp)) {
    started <- proc.time()[["elapsed"]]
    fitted <- fit_torch_average(
      source_matrix$x,
      source_truth,
      target_matrix$x,
      config = repeat_deep_spec,
      seeds =
        c(9201L, 9202L, 9203L) +
        repeat_index * 10000L + fold * 100L
    )
    state$predictions$deep_mlp <-
      fitted$prediction
    state$fit_details$deep_fits <- fitted$fits
    state$fit_details$deep_elapsed <-
      proc.time()[["elapsed"]] - started
    save_fold_state(state, checkpoint)
    cat("  deep MLP complete\n")
    flush.console()
  }

  for (name in c(
    "mlogit", "history_mlogit", "triple_mlogit",
    "original_xgb", "retuned_xgb", "cox",
    "shallow_mlp", "deep_mlp"
  )) {
    state$predictions[[name]] <-
      repeat_validate_prediction(
        state$predictions[[name]], nrow(target_wide)
      )
  }
  stopifnot(
    identical(
      dim(state$predictions$rank_margin),
      c(nrow(target_wide), 4L)
    ),
    !anyNA(state$predictions$rank_margin)
  )
  save_fold_state(state, checkpoint)
  state
}

assemble_additional_repeat <- function(
    repeat_seed, repeat_index) {
  output_path <- file.path(
    repeat_output_dir,
    sprintf("repeat_seed_%d.rds", repeat_seed)
  )
  if (file.exists(output_path)) {
    saved <- readRDS(output_path)
    if (identical(saved$repeat_seed, repeat_seed)) {
      cat("Repeat", repeat_seed, "loaded from checkpoint.\n")
      return(saved)
    }
  }

  fold_map <- repeat_fold_map(cases, repeat_seed)
  row_fold <- unname(
    fold_map[as.character(train$Case)]
  )
  oof <- list(
    mlogit = matrix(NA_real_, nrow(train), 4L),
    history_mlogit = matrix(NA_real_, nrow(train), 4L),
    original_xgb = matrix(NA_real_, nrow(train), 4L),
    rank_margin = matrix(NA_real_, nrow(train), 4L),
    retuned_xgb = matrix(NA_real_, nrow(train), 4L),
    cox = matrix(NA_real_, nrow(train), 4L),
    shallow_mlp = matrix(NA_real_, nrow(train), 4L),
    triple_mlogit = matrix(NA_real_, nrow(train), 4L),
    deep_mlp = matrix(NA_real_, nrow(train), 4L)
  )
  fold_details <- list()

  for (fold in 1:5) {
    cat(sprintf(
      "\n=== repeat seed %d, fold %d/5 ===\n",
      repeat_seed, fold
    ))
    flush.console()
    state <- run_repeat_fold(
      repeat_seed, repeat_index, fold, fold_map
    )
    rows <- match(state$validation_no, train$No)
    stopifnot(!anyNA(rows), all(row_fold[rows] == fold))
    for (name in names(oof)) {
      oof[[name]][rows, ] <- state$predictions[[name]]
    }
    fold_details[[fold]] <- state$fit_details
    invisible(gc())
  }
  stopifnot(all(vapply(oof, function(x) {
    !anyNA(x)
  }, logical(1))))

  rank <- repeat_calibrate_rank(
    oof$rank_margin, truth, row_fold
  )
  components <- list(
    mlogit = oof$mlogit,
    original_xgb = oof$original_xgb,
    rank_ndcg = rank$prediction,
    retuned_xgb = oof$retuned_xgb,
    cox = oof$cox,
    shallow_mlp = oof$shallow_mlp,
    triple_mlogit = oof$triple_mlogit,
    deep_mlp = oof$deep_mlp
  )
  for (name in names(components)) {
    components[[name]] <- repeat_validate_prediction(
      components[[name]], nrow(train)
    )
  }
  current <- repeat_crossfit_current(
    truth,
    components$mlogit,
    components$original_xgb,
    components$shallow_mlp,
    row_fold
  )
  augmented8 <- repeat_crossfit_arithmetic(
    truth, components, row_fold
  )
  history <- repeat_fixed_history_predictions(
    components$mlogit,
    oof$history_mlogit,
    components$original_xgb,
    components$shallow_mlp
  )

  result <- list(
    repeat_seed = repeat_seed,
    repeat_index = repeat_index,
    fold_map = fold_map,
    row_fold = row_fold,
    components = components,
    rank_scales = rank$scales,
    current = current,
    augmented8 = augmented8,
    history = history,
    history_mlogit = oof$history_mlogit,
    fold_details = fold_details
  )
  saveRDS(result, output_path)
  cat(sprintf(
    paste0(
      "repeat %d complete: history gain %+.9f; ",
      "eight gain %+.9f\n"
    ),
    repeat_seed,
    log_loss_matrix(truth, history$baseline) -
      log_loss_matrix(truth, history$candidate),
    current$logloss - augmented8$logloss
  ))
  flush.console()
  result
}

load_canonical_repeat <- function() {
  base <- readRDS("data_processed/oof_ensemble_v10.rds")
  history <- readRDS(
    "data_processed/codex_overnight_queue/history_prior_cv.rds"
  )
  full_stack <- readRDS(
    "data_processed/codex_deep_stack/full_stacking.rds"
  )
  components <- list(
    mlogit = base$oof_mlogit,
    original_xgb = base$oof_xgb,
    rank_ndcg = readRDS(
      "data_processed/codex/rank_oof.rds"
    )[[1]]$pred,
    retuned_xgb = readRDS(
      "data_processed/codex/xgb_retune_oof.rds"
    )[[1]]$pred,
    cox = readRDS(
      "data_processed/codex/cox_oof.rds"
    )$oof_min,
    shallow_mlp = readRDS(
      "data_processed/codex_behavioral_round/mlp_oof.rds"
    )$oof[["h08_d0.100"]],
    triple_mlogit = readRDS(
      "data_processed/codex_triples/triple_oof.rds"
    )$oof[["triple_price_income_miles"]],
    deep_mlp = readRDS(
      "data_processed/codex_deep_stack/torch_deep_oof.rds"
    )$deep_oof
  )
  stopifnot(identical(names(components), repeat_component_names))
  row_fold <- unname(
    base$fold_of_case[as.character(train$Case)]
  )
  current <- list(
    prediction = full_stack$current,
    logloss = log_loss_matrix(truth, full_stack$current),
    weights = readRDS(
      "data_processed/codex_behavioral_round/mlp_precision.rds"
    )$fold_results$mlp_weight
  )
  augmented8_saved <-
    full_stack$results$augmented8_arithmetic
  augmented8_weights <- matrix(
    NA_real_, 5L, length(repeat_component_names),
    dimnames = list(
      paste0("fold", 1:5), repeat_component_names
    )
  )
  for (fold in 1:5) {
    rows <- augmented8_saved$weights$outer_fold == fold
    fold_weights <- augmented8_saved$weights[rows, ]
    augmented8_weights[
      fold, fold_weights$component
    ] <- fold_weights$weight
  }
  stopifnot(
    !anyNA(augmented8_weights),
    max(abs(rowSums(augmented8_weights) - 1)) < 1e-10
  )
  augmented8 <- list(
    prediction = augmented8_saved$prediction,
    logloss = augmented8_saved$logloss,
    weights = augmented8_weights
  )
  history_prediction <-
    history$predictions[["both_k3"]]
  stopifnot(
    abs(current$logloss - 1.14378944178118) < 1e-10,
    abs(augmented8$logloss -
      1.142111909841650) < 1e-10,
    abs(log_loss_matrix(truth, history$baseline) -
      1.143686618134879) < 1e-10,
    abs(log_loss_matrix(truth, history_prediction) -
      1.14295145009933) < 1e-10
  )
  list(
    repeat_seed = 4821L,
    repeat_index = 0L,
    fold_map = base$fold_of_case,
    row_fold = row_fold,
    components = components,
    current = current,
    augmented8 = augmented8,
    history = list(
      baseline = history$baseline,
      candidate = history_prediction
    )
  )
}

repeats <- list(`4821` = load_canonical_repeat())
for (index in seq_along(repeat_additional_seeds)) {
  seed <- repeat_additional_seeds[[index]]
  repeats[[as.character(seed)]] <-
    assemble_additional_repeat(seed, index)
}
stopifnot(
  identical(
    as.integer(names(repeats)),
    repeat_all_seeds
  )
)

history_case_gain <- matrix(
  NA_real_, nrow = length(cases),
  ncol = length(repeats),
  dimnames = list(as.character(cases), names(repeats))
)
eight_case_gain <- history_case_gain
repeat_rows <- list()
fold_rows <- list()
weight_rows <- list()

for (repeat_name in names(repeats)) {
  result <- repeats[[repeat_name]]
  history_case_gain[, repeat_name] <- repeat_case_gain(
    truth,
    result$history$baseline,
    result$history$candidate,
    train$Case
  )
  eight_case_gain[, repeat_name] <- repeat_case_gain(
    truth,
    result$current$prediction,
    result$augmented8$prediction,
    train$Case
  )
  repeat_rows[[paste0("history_", repeat_name)]] <- data.frame(
    candidate = "history_both_k3",
    repeat_seed = as.integer(repeat_name),
    baseline_loss = log_loss_matrix(
      truth, result$history$baseline
    ),
    candidate_loss = log_loss_matrix(
      truth, result$history$candidate
    )
  )
  repeat_rows[[paste0("eight_", repeat_name)]] <- data.frame(
    candidate = "augmented8_arithmetic",
    repeat_seed = as.integer(repeat_name),
    baseline_loss = result$current$logloss,
    candidate_loss = result$augmented8$logloss
  )
  for (fold in 1:5) {
    rows <- result$row_fold == fold
    fold_rows[[length(fold_rows) + 1L]] <- data.frame(
      candidate = "history_both_k3",
      repeat_seed = as.integer(repeat_name),
      fold = fold,
      baseline_loss = log_loss_matrix(
        truth[rows, , drop = FALSE],
        result$history$baseline[rows, , drop = FALSE]
      ),
      candidate_loss = log_loss_matrix(
        truth[rows, , drop = FALSE],
        result$history$candidate[rows, , drop = FALSE]
      )
    )
    fold_rows[[length(fold_rows) + 1L]] <- data.frame(
      candidate = "augmented8_arithmetic",
      repeat_seed = as.integer(repeat_name),
      fold = fold,
      baseline_loss = log_loss_matrix(
        truth[rows, , drop = FALSE],
        result$current$prediction[rows, , drop = FALSE]
      ),
      candidate_loss = log_loss_matrix(
        truth[rows, , drop = FALSE],
        result$augmented8$prediction[
          rows, , drop = FALSE
        ]
      )
    )
  }
  weight_rows[[repeat_name]] <- data.frame(
    repeat_seed = as.integer(repeat_name),
    fold = rep(1:5, each = length(repeat_component_names)),
    component = rep(repeat_component_names, times = 5),
    weight = as.numeric(t(result$augmented8$weights))
  )
}

repeat_summary <- do.call(rbind, repeat_rows)
repeat_summary$gain <- with(
  repeat_summary, baseline_loss - candidate_loss
)
fold_summary <- do.call(rbind, fold_rows)
fold_summary$gain <- with(
  fold_summary, baseline_loss - candidate_loss
)
weight_summary <- do.call(rbind, weight_rows)

history_bootstrap <- repeat_bootstrap_average_gain(
  history_case_gain, family_size = 13L
)
eight_bootstrap <- repeat_bootstrap_average_gain(
  eight_case_gain, family_size = 6L
)
bootstrap_summary <- rbind(
  cbind(
    candidate = "history_both_k3",
    history_bootstrap$summary
  ),
  cbind(
    candidate = "augmented8_arithmetic",
    eight_bootstrap$summary
  )
)
decision <- data.frame(
  candidate = c(
    "history_both_k3", "augmented8_arithmetic"
  ),
  positive_repeats = c(
    sum(colMeans(history_case_gain) > 0),
    sum(colMeans(eight_case_gain) > 0)
  ),
  family_lower = c(
    history_bootstrap$summary$lower_family,
    eight_bootstrap$summary$lower_family
  )
)
decision$promote <-
  decision$positive_repeats >= 5L &
  decision$family_lower > 0

write.csv(
  repeat_summary,
  file.path(repeat_output_dir, "repeat_cv_summary.csv"),
  row.names = FALSE
)
write.csv(
  fold_summary,
  file.path(repeat_output_dir, "repeat_cv_folds.csv"),
  row.names = FALSE
)
write.csv(
  weight_summary,
  file.path(repeat_output_dir, "repeat_cv_weights.csv"),
  row.names = FALSE
)
write.csv(
  bootstrap_summary,
  file.path(repeat_output_dir, "repeat_cv_bootstrap.csv"),
  row.names = FALSE
)
write.csv(
  decision,
  file.path(repeat_output_dir, "repeat_cv_decision.csv"),
  row.names = FALSE
)
saveRDS(
  list(
    seeds = repeat_all_seeds,
    repeat_summary = repeat_summary,
    fold_summary = fold_summary,
    weight_summary = weight_summary,
    history_case_gain = history_case_gain,
    eight_case_gain = eight_case_gain,
    history_bootstrap = history_bootstrap,
    eight_bootstrap = eight_bootstrap,
    bootstrap_summary = bootstrap_summary,
    decision = decision
  ),
  file.path(repeat_output_dir, "repeat_cv_results.rds")
)

print(repeat_summary, digits = 10, row.names = FALSE)
print(bootstrap_summary, digits = 10, row.names = FALSE)
print(decision, digits = 10, row.names = FALSE)
