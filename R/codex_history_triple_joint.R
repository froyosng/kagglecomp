# One frozen joint candidate: both_k3 history terms plus
# Price x z(income) x z(mileage), replacing the triple component in the
# canonical eight-component arithmetic pool.

source("R/codex_repeat_cv_common.R")

joint_stage <- Sys.getenv("CODEX_JOINT_STAGE", "screen")
stopifnot(joint_stage %in% c("screen", "cv"))

joint_output_dir <- "data_processed/codex_repeat_cv"
dir.create(
  joint_output_dir, recursive = TRUE, showWarnings = FALSE
)

bootstrap_pair <- function(
    truth, baseline, candidate, case,
    family_size = 7L, n_boot = 100000L,
    seed = 4821L) {
  case_gain <- repeat_case_gain(
    truth, baseline, candidate, case
  )
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
  family_alpha <- 0.05 / family_size
  list(
    summary = data.frame(
      point_gain = mean(case_gain),
      bootstrap_mean = mean(bootstrap),
      bootstrap_sd = sd(bootstrap),
      lower_95 = unname(quantile(bootstrap, 0.025)),
      upper_95 = unname(quantile(bootstrap, 0.975)),
      lower_99 = unname(quantile(bootstrap, 0.005)),
      upper_99 = unname(quantile(bootstrap, 0.995)),
      lower_family_7 = unname(quantile(
        bootstrap, family_alpha / 2
      )),
      upper_family_7 = unname(quantile(
        bootstrap, 1 - family_alpha / 2
      )),
      win_rate = mean(bootstrap > 0),
      n_boot = n_boot,
      family_size = family_size
    ),
    case_gain = case_gain,
    bootstrap = bootstrap
  )
}

if (joint_stage == "screen") {
  split <- readRDS("data_processed/train_val_split.rds")
  source_raw <- prepare_history_long(
    split$train_long_tr
  )
  target_raw <- prepare_history_long(
    split$train_long_val
  )
  prior <- history_design_prior(source_raw)
  source <- add_prior_smoothed_history(
    source_raw, prior
  )
  target <- add_prior_smoothed_history(
    target_raw, prior
  )
  target_wide <- split$train_wide_val
  target_wide <- target_wide[
    order(target_wide$No), , drop = FALSE
  ]
  truth <- as.matrix(
    target_wide[, paste0("Ch", 1:4), drop = FALSE]
  )
  fitted <- fit_predict_joint_history_triple(
    source, target
  )
  joint_prediction <- fitted$pred[
    match(target_wide$No, fitted$no),
    ,
    drop = FALSE
  ]
  triple_screen <- readRDS(
    "data_processed/codex_triples/triple_screen.rds"
  )
  stopifnot(identical(
    as.integer(triple_screen$validation_no),
    as.integer(target_wide$No)
  ))
  triple_prediction <-
    triple_screen$predictions[[
      "triple_price_income_miles"
    ]]
  triple_loss <- log_loss_matrix(
    truth, triple_prediction
  )
  joint_loss <- log_loss_matrix(
    truth, joint_prediction
  )
  result <- data.frame(
    candidate = "history_both_k3_plus_triple",
    triple_loss = triple_loss,
    joint_loss = joint_loss,
    gain_vs_triple = triple_loss - joint_loss,
    passed_screen = joint_loss < triple_loss,
    coefficient_price_income_miles =
      fitted$extra_coefficient[["P_income_miles"]],
    coefficient_history_price =
      fitted$extra_coefficient[[
        "hist_prior_price_gap_k3"
      ]],
    coefficient_history_attribute =
      fitted$extra_coefficient[[
        "hist_prior_attr_familiarity_k3"
      ]]
  )
  write.csv(
    result,
    file.path(joint_output_dir, "joint_screen.csv"),
    row.names = FALSE
  )
  saveRDS(
    list(
      result = result,
      prediction = joint_prediction,
      truth = truth,
      no = target_wide$No,
      triple_prediction = triple_prediction
    ),
    file.path(joint_output_dir, "joint_screen.rds")
  )
  print(result, digits = 12, row.names = FALSE)
}

if (joint_stage == "cv") {
  screen <- read.csv(
    file.path(joint_output_dir, "joint_screen.csv")
  )
  if (!isTRUE(screen$passed_screen[[1]])) {
    cat("Joint candidate failed its frozen screen gate; CV skipped.\n")
    quit(save = "no", status = 0)
  }

  train <- read.csv("csv files/train.csv")
  train <- train[order(train$No), , drop = FALSE]
  truth <- as.matrix(
    train[, paste0("Ch", 1:4), drop = FALSE]
  )
  full_long <- reshape_choice_long(train)
  base <- readRDS("data_processed/oof_ensemble_v10.rds")
  row_fold <- unname(
    base$fold_of_case[as.character(train$Case)]
  )
  joint_oof <- matrix(
    NA_real_, nrow(train), 4L
  )
  coefficient_rows <- list()

  for (fold in 1:5) {
    checkpoint <- file.path(
      joint_output_dir,
      sprintf("joint_cv_fold_%d.rds", fold)
    )
    validation_cases <- as.integer(
      names(base$fold_of_case)[
        base$fold_of_case == fold
      ]
    )
    target_rows <- train$Case %in% validation_cases
    target_no <- train$No[target_rows]
    if (file.exists(checkpoint)) {
      fitted <- readRDS(checkpoint)
      stopifnot(identical(
        as.integer(fitted$no),
        as.integer(target_no)
      ))
    } else {
      source_raw <- full_long[
        !(full_long$Case %in% validation_cases),
        ,
        drop = FALSE
      ]
      target_raw <- full_long[
        full_long$Case %in% validation_cases,
        ,
        drop = FALSE
      ]
      prior <- history_design_prior(source_raw)
      source <- add_prior_smoothed_history(
        source_raw, prior
      )
      target <- add_prior_smoothed_history(
        target_raw, prior
      )
      model_fit <- fit_predict_joint_history_triple(
        source, target
      )
      fitted <- list(
        no = target_no,
        prediction = model_fit$pred[
          match(target_no, model_fit$no),
          ,
          drop = FALSE
        ],
        coefficient = model_fit$extra_coefficient
      )
      saveRDS(fitted, checkpoint)
    }
    joint_oof[target_rows, ] <- fitted$prediction
    coefficient_rows[[fold]] <- data.frame(
      fold = fold,
      term = names(fitted$coefficient),
      coefficient = unname(fitted$coefficient)
    )
    cat(sprintf(
      "joint fold %d complete: %.12f\n",
      fold,
      log_loss_matrix(
        truth[target_rows, , drop = FALSE],
        fitted$prediction
      )
    ))
    flush.console()
  }
  joint_oof <- repeat_validate_prediction(
    joint_oof, nrow(train)
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
    triple_mlogit = joint_oof,
    deep_mlp = readRDS(
      "data_processed/codex_deep_stack/torch_deep_oof.rds"
    )$deep_oof
  )
  joint_blend <- repeat_crossfit_arithmetic(
    truth, components, row_fold
  )
  full_stack <- readRDS(
    "data_processed/codex_deep_stack/full_stacking.rds"
  )
  current <- full_stack$current
  plain_eight <-
    full_stack$results$augmented8_arithmetic$prediction
  stopifnot(
    abs(log_loss_matrix(truth, current) -
      1.14378944178118) < 1e-10,
    abs(log_loss_matrix(truth, plain_eight) -
      1.142111909841650) < 1e-10
  )

  comparisons <- list(
    vs_current = bootstrap_pair(
      truth, current, joint_blend$prediction,
      train$Case
    ),
    vs_plain_eight = bootstrap_pair(
      truth, plain_eight, joint_blend$prediction,
      train$Case
    )
  )
  bootstrap_summary <- do.call(rbind, lapply(
    names(comparisons),
    function(name) cbind(
      comparison = name,
      comparisons[[name]]$summary
    )
  ))
  fold_rows <- do.call(rbind, lapply(1:5, function(fold) {
    rows <- row_fold == fold
    data.frame(
      fold = fold,
      current_loss = log_loss_matrix(
        truth[rows, , drop = FALSE],
        current[rows, , drop = FALSE]
      ),
      plain_eight_loss = log_loss_matrix(
        truth[rows, , drop = FALSE],
        plain_eight[rows, , drop = FALSE]
      ),
      joint_loss = log_loss_matrix(
        truth[rows, , drop = FALSE],
        joint_blend$prediction[
          rows, , drop = FALSE
        ]
      )
    )
  }))
  fold_rows$gain_vs_current <- with(
    fold_rows, current_loss - joint_loss
  )
  fold_rows$gain_vs_plain_eight <- with(
    fold_rows, plain_eight_loss - joint_loss
  )
  summary <- data.frame(
    candidate = "history_both_k3_plus_triple",
    component_loss = log_loss_matrix(truth, joint_oof),
    current_loss = log_loss_matrix(truth, current),
    plain_eight_loss = log_loss_matrix(
      truth, plain_eight
    ),
    joint_blend_loss = joint_blend$logloss,
    gain_vs_current =
      log_loss_matrix(truth, current) -
      joint_blend$logloss,
    gain_vs_plain_eight =
      log_loss_matrix(truth, plain_eight) -
      joint_blend$logloss
  )
  weights <- data.frame(
    fold = rep(1:5, each = length(repeat_component_names)),
    component = rep(repeat_component_names, times = 5),
    weight = as.numeric(t(joint_blend$weights))
  )
  write.csv(
    summary,
    file.path(joint_output_dir, "joint_cv.csv"),
    row.names = FALSE
  )
  write.csv(
    fold_rows,
    file.path(joint_output_dir, "joint_cv_folds.csv"),
    row.names = FALSE
  )
  write.csv(
    do.call(rbind, coefficient_rows),
    file.path(
      joint_output_dir, "joint_cv_coefficients.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    weights,
    file.path(joint_output_dir, "joint_cv_weights.csv"),
    row.names = FALSE
  )
  write.csv(
    bootstrap_summary,
    file.path(
      joint_output_dir, "joint_cv_bootstrap.csv"
    ),
    row.names = FALSE
  )
  saveRDS(
    list(
      summary = summary,
      fold_results = fold_rows,
      coefficients = do.call(rbind, coefficient_rows),
      weights = weights,
      joint_oof = joint_oof,
      joint_prediction = joint_blend$prediction,
      current = current,
      plain_eight = plain_eight,
      bootstrap = comparisons
    ),
    file.path(joint_output_dir, "joint_cv.rds")
  )
  print(summary, digits = 12, row.names = FALSE)
  print(fold_rows, digits = 12, row.names = FALSE)
  print(bootstrap_summary, digits = 12, row.names = FALSE)
}
