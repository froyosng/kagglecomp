## Fold-safe empirical-Bayes initialization for observable design history.
##
## The earlier design-history round used a respondent's tasks 1..t-1 but
## zeroed Task 1. This targeted follow-up initializes the running price
## reference and attribute-level exposure rates from the fitting fold's design
## distribution, then lets respondent-specific exposure progressively dominate.
##
## Pre-registered family (before screening): five candidates.
##   1. price only, prior strength 9 shown alternatives
##   2. attribute familiarity only, prior strength 9
##   3-5. both features, prior strengths 3 / 9 / 27
##
## Stages:
##   CODEX_HISTORY_PRIOR_STAGE=screen
##   CODEX_HISTORY_PRIOR_STAGE=cv

suppressPackageStartupMessages({
  library(mlogit)
  library(dfidx)
})
source("R/codex_shift_common.R")

history_prior_stage <- Sys.getenv(
  "CODEX_HISTORY_PRIOR_STAGE", "screen"
)
stopifnot(history_prior_stage %in% c("screen", "cv"))

history_prior_output_dir <-
  "data_processed/codex_overnight_queue"
dir.create(
  history_prior_output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

history_prior_specs <- list(
  price_k9 = list(
    strength = 9,
    terms = "hist_prior_price_gap_k9"
  ),
  attribute_k9 = list(
    strength = 9,
    terms = "hist_prior_attr_familiarity_k9"
  ),
  both_k3 = list(
    strength = 3,
    terms = c(
      "hist_prior_price_gap_k3",
      "hist_prior_attr_familiarity_k3"
    )
  ),
  both_k9 = list(
    strength = 9,
    terms = c(
      "hist_prior_price_gap_k9",
      "hist_prior_attr_familiarity_k9"
    )
  ),
  both_k27 = list(
    strength = 27,
    terms = c(
      "hist_prior_price_gap_k27",
      "hist_prior_attr_familiarity_k27"
    )
  )
)
history_prior_family_size <- length(history_prior_specs)
stopifnot(history_prior_family_size == 5L)

prepare_history_long <- function(df) {
  df <- as.data.frame(df)
  if (!("chid" %in% names(df))) {
    df$chid <- paste(df$Case, df$Task, sep = "_")
  }
  if (!("d2" %in% names(df))) {
    df$d2 <- as.integer(df$alt == 2L)
  }
  if (!("d3" %in% names(df))) {
    df$d3 <- as.integer(df$alt == 3L)
  }
  df
}

history_design_prior <- function(fitting_long) {
  fitting_long <- prepare_history_long(fitting_long)
  inside <- fitting_long$alt != 4L
  list(
    mean_price = mean(as.numeric(fitting_long$Price[inside])),
    attribute_frequency = lapply(attrs, function(attribute) {
      values <- as.integer(fitting_long[[attribute]][inside])
      level <- sort(unique(values))
      frequency <- vapply(
        level, function(x) mean(values == x), numeric(1)
      )
      names(frequency) <- as.character(level)
      frequency
    })
  )
}

add_prior_smoothed_history <- function(df, design_prior,
                                       strengths = c(3, 9, 27)) {
  df <- prepare_history_long(df)
  df$.history_order <- seq_len(nrow(df))
  df <- df[order(df$Case, df$Task, df$alt), , drop = FALSE]
  for (strength in strengths) {
    df[[paste0("hist_prior_price_gap_k", strength)]] <- 0
    df[[
      paste0("hist_prior_attr_familiarity_k", strength)
    ]] <- 0
  }

  for (respondent in sort(unique(df$Case))) {
    respondent_rows <- which(df$Case == respondent)
    tasks <- sort(unique(df$Task[respondent_rows]))
    stopifnot(length(tasks) == 19L)
    prior_prices <- numeric()
    prior_attributes <- matrix(
      numeric(), nrow = 0L, ncol = length(attrs),
      dimnames = list(NULL, attrs)
    )

    for (task in tasks) {
      task_rows <- respondent_rows[
        df$Task[respondent_rows] == task
      ]
      task_rows <- task_rows[order(df$alt[task_rows])]
      stopifnot(
        length(task_rows) == 4L,
        identical(as.integer(df$alt[task_rows]), 1:4)
      )
      inside_rows <- task_rows[1:3]
      current_prices <- as.numeric(df$Price[inside_rows])
      current_attributes <- as.matrix(
        df[inside_rows, attrs, drop = FALSE]
      )
      storage.mode(current_attributes) <- "double"
      stopifnot(all(rowSums(current_attributes != 0) == 9L))

      for (strength in strengths) {
        price_reference <- (
          strength * design_prior$mean_price +
            sum(prior_prices)
        ) / (strength + length(prior_prices))
        df[[
          paste0("hist_prior_price_gap_k", strength)
        ]][inside_rows] <- current_prices - price_reference

        for (alternative in 1:3) {
          row <- inside_rows[[alternative]]
          current <- current_attributes[alternative, ]
          active <- which(current != 0)
          familiarity <- vapply(active, function(attribute_index) {
            level <- as.character(current[[attribute_index]])
            global_frequency <-
              design_prior$attribute_frequency[[
                attribute_index
              ]][level]
            if (length(global_frequency) == 0L ||
                is.na(global_frequency)) {
              global_frequency <- 0
            }
            past_match <- if (nrow(prior_attributes) == 0L) {
              0
            } else {
              sum(
                prior_attributes[, attribute_index] ==
                  current[[attribute_index]]
              )
            }
            (
              strength * global_frequency + past_match
            ) / (strength + nrow(prior_attributes))
          }, numeric(1))
          df[[
            paste0(
              "hist_prior_attr_familiarity_k", strength
            )
          ]][[row]] <- mean(familiarity)
        }
      }
      prior_prices <- c(prior_prices, current_prices)
      prior_attributes <- rbind(
        prior_attributes, current_attributes
      )
    }
  }

  history_columns <- grep(
    "^hist_prior_", names(df), value = TRUE
  )
  stopifnot(all(
    as.matrix(
      df[df$alt == 4L, history_columns, drop = FALSE]
    ) == 0
  ))
  df <- df[order(df$.history_order), , drop = FALSE]
  df$.history_order <- NULL
  rownames(df) <- NULL
  df
}

history_prior_formula <- function(candidate) {
  stopifnot(candidate %in% names(history_prior_specs))
  base <- m8trpg_formula("none")
  base_rhs <- sub("\\| 0$", "", as.character(base)[3])
  extra <- paste(
    history_prior_specs[[candidate]]$terms,
    collapse = " + "
  )
  formula <- as.formula(paste(
    "chosen ~", base_rhs, "+", extra, "| 0"
  ))
  environment(formula) <- environment()
  formula
}

fit_predict_history_prior <- function(source_long, target_long,
                                      candidate) {
  scaler <- choice_scaler(source_long)
  source_features <- make_m8trpg_features(
    source_long, scaler$ctr, scaler$scl, "none"
  )
  target_features <- make_m8trpg_features(
    target_long, scaler$ctr, scaler$scl, "none"
  )
  formula <- history_prior_formula(candidate)
  model <- mlogit(
    formula,
    data = source_features,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen"
  )
  indexed <- dfidx(
    target_features,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen"
  )
  class(indexed) <- c("dfidx_mlogit", class(indexed))
  frame <- model.frame(indexed, formula, balanced = TRUE)
  design <- model.matrix(frame, rhs = 1:3)
  coefficient <- coef(model)
  stopifnot(all(names(coefficient) %in% colnames(design)))
  margin <- as.numeric(
    design[, names(coefficient), drop = FALSE] %*%
      coefficient
  )
  task_map <- unique(target_features[, c("chid", "No")])
  task_map <- task_map[order(task_map$No), , drop = FALSE]
  margin_chid <- as.character(dfidx::idx(frame, 1))
  margin_alt <- as.integer(as.character(dfidx::idx(frame, 2)))
  margin_order <- order(
    match(margin_chid, task_map$chid), margin_alt
  )
  stopifnot(
    identical(
      margin_chid[margin_order],
      rep(task_map$chid, each = 4L)
    ),
    identical(
      margin_alt[margin_order],
      rep(1:4, times = nrow(task_map))
    )
  )
  list(
    pred = softmax_margins(margin[margin_order]),
    no = as.integer(task_map$No),
    extra_coefficient =
      coefficient[history_prior_specs[[candidate]]$terms]
  )
}

history_prior_bootstrap <- function(truth, baseline, candidate,
                                    case, replicates = 100000L,
                                    seed = 4821L) {
  row_loss <- function(prediction) {
    prediction <- prediction / rowSums(prediction)
    prediction <- pmin(pmax(prediction, 1e-15), 1)
    -rowSums(truth * log(prediction))
  }
  case_gain <- tapply(
    row_loss(baseline) - row_loss(candidate),
    case, mean
  )
  set.seed(seed)
  bootstrap <- replicate(
    as.integer(replicates),
    mean(sample(case_gain, length(case_gain), replace = TRUE))
  )
  family_alpha <- 0.05 / history_prior_family_size
  total_family_alpha <- 0.05 / (
    history_prior_family_size + 8L
  )
  data.frame(
    point_gain = mean(case_gain),
    bootstrap_mean = mean(bootstrap),
    bootstrap_sd = sd(bootstrap),
    lower_95 = unname(quantile(bootstrap, 0.025)),
    upper_95 = unname(quantile(bootstrap, 0.975)),
    lower_bonferroni_5 = unname(quantile(
      bootstrap, family_alpha / 2
    )),
    upper_bonferroni_5 = unname(quantile(
      bootstrap, 1 - family_alpha / 2
    )),
    lower_bonferroni_13 = unname(quantile(
      bootstrap, total_family_alpha / 2
    )),
    upper_bonferroni_13 = unname(quantile(
      bootstrap, 1 - total_family_alpha / 2
    )),
    win_rate = mean(bootstrap > 0),
    replicates = as.integer(replicates)
  )
}

history_prior_xgb_screen <- function(split_data) {
  source_wide <- split_data$train_wide_tr
  target_wide <- split_data$train_wide_val
  source_wide$Choice <- max.col(
    source_wide[, paste0("Ch", 1:4), drop = FALSE]
  )
  model <- xgboost::xgb.train(
    params = list(
      objective = "multi:softprob",
      num_class = 4L,
      eval_metric = "mlogloss",
      eta = 0.1,
      max_depth = 4L,
      subsample = 0.8,
      colsample_bytree = 0.8,
      seed = 4821L,
      nthread = 1L
    ),
    data = xgboost::xgb.DMatrix(
      wide_feature_matrix(source_wide),
      label = source_wide$Choice - 1L,
      nthread = 1L
    ),
    nrounds = 73L,
    verbose = 0
  )
  target_order <- order(target_wide$No)
  prediction <- as.matrix(predict(
    model,
    xgboost::xgb.DMatrix(
      wide_feature_matrix(target_wide[target_order, , drop = FALSE]),
      nthread = 1L
    )
  ))
  stopifnot(
    ncol(prediction) == 4L,
    max(abs(rowSums(prediction) - 1)) < 1e-6
  )
  prediction / rowSums(prediction)
}

if (history_prior_stage == "screen") {
  split_data <- readRDS(
    "data_processed/train_val_split.rds"
  )
  source_raw <- prepare_history_long(
    split_data$train_long_tr
  )
  target_raw <- prepare_history_long(
    split_data$train_long_val
  )
  prior <- history_design_prior(source_raw)
  source <- add_prior_smoothed_history(source_raw, prior)
  target <- add_prior_smoothed_history(target_raw, prior)
  target_wide <- split_data$train_wide_val
  target_wide <- target_wide[
    order(target_wide$No), , drop = FALSE
  ]
  truth <- as.matrix(
    target_wide[, paste0("Ch", 1:4), drop = FALSE]
  )

  baseline_fit <- fit_predict_m8trpg(source, target)
  baseline <- baseline_fit$pred[
    match(target_wide$No, baseline_fit$no),
    ,
    drop = FALSE
  ]
  stopifnot(
    abs(
      log_loss_matrix(truth, baseline) -
        1.15968144721113
    ) < 1e-8
  )
  mlp_screen <- readRDS(
    "data_processed/codex_behavioral_round/mlp_screen.rds"
  )
  stopifnot(identical(
    as.integer(mlp_screen$validation_no),
    as.integer(target_wide$No)
  ))
  shallow <- mlp_screen$predictions[["h08_d0.100"]]
  xgb <- history_prior_xgb_screen(split_data)
  v11 <- 0.8 * baseline + 0.2 * xgb
  current <- 0.85 * v11 + 0.15 * shallow

  rows <- list()
  coefficients <- list()
  predictions <- list()
  for (candidate in names(history_prior_specs)) {
    fitted <- fit_predict_history_prior(
      source, target, candidate
    )
    history_mlogit <- fitted$pred[
      match(target_wide$No, fitted$no),
      ,
      drop = FALSE
    ]
    history_v11 <- 0.8 * history_mlogit + 0.2 * xgb
    history_current <- 0.85 * history_v11 + 0.15 * shallow
    rows[[candidate]] <- data.frame(
      candidate = candidate,
      prior_strength =
        history_prior_specs[[candidate]]$strength,
      n_terms = length(
        history_prior_specs[[candidate]]$terms
      ),
      mlogit_loss = log_loss_matrix(truth, history_mlogit),
      current_loss = log_loss_matrix(truth, current),
      candidate_loss =
        log_loss_matrix(truth, history_current),
      gain =
        log_loss_matrix(truth, current) -
        log_loss_matrix(truth, history_current)
    )
    coefficients[[candidate]] <- data.frame(
      candidate = candidate,
      term = names(fitted$extra_coefficient),
      coefficient = unname(fitted$extra_coefficient)
    )
    predictions[[candidate]] <- history_current
  }
  result <- do.call(rbind, rows)
  write.csv(
    result,
    file.path(
      history_prior_output_dir,
      "history_prior_screen.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    do.call(rbind, coefficients),
    file.path(
      history_prior_output_dir,
      "history_prior_screen_coefficients.csv"
    ),
    row.names = FALSE
  )
  saveRDS(
    list(
      result = result,
      predictions = predictions,
      baseline = current,
      truth = truth,
      case = target_wide$Case,
      no = target_wide$No,
      family_size = history_prior_family_size
    ),
    file.path(
      history_prior_output_dir,
      "history_prior_screen.rds"
    )
  )
  print(result, digits = 10)
}

if (history_prior_stage == "cv") {
  screen <- read.csv(file.path(
    history_prior_output_dir,
    "history_prior_screen.csv"
  ))
  candidates <- screen$candidate[screen$gain > 0]
  if (length(candidates) == 0L) {
    cat("No pre-registered history-prior candidate passed screen.\n")
    quit(save = "no", status = 0)
  }

  train <- read.csv("csv files/train.csv")
  train <- train[order(train$No), , drop = FALSE]
  truth <- as.matrix(
    train[, paste0("Ch", 1:4), drop = FALSE]
  )
  base <- readRDS("data_processed/oof_ensemble_v10.rds")
  fold_map <- base$fold_of_case
  row_fold <- unname(
    fold_map[as.character(train$Case)]
  )
  shallow <- readRDS(
    "data_processed/codex_behavioral_round/mlp_oof.rds"
  )$oof[["h08_d0.100"]]
  v11 <- 0.8 * base$oof_mlogit + 0.2 * base$oof_xgb
  current <- 0.85 * v11 + 0.15 * shallow
  stopifnot(
    abs(log_loss_matrix(truth, current) -
      1.143686618134879) < 1e-10
  )

  candidate_oof <- lapply(
    candidates,
    function(x) matrix(NA_real_, nrow(train), 4L)
  )
  names(candidate_oof) <- candidates
  fold_rows <- list()
  coefficient_rows <- list()
  full_long <- reshape_choice_long(train)

  for (fold in 1:5) {
    validation_cases <- as.integer(
      names(fold_map)[fold_map == fold]
    )
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
    source <- add_prior_smoothed_history(source_raw, prior)
    target <- add_prior_smoothed_history(target_raw, prior)
    target_rows <- train$Case %in% validation_cases
    target_no <- train$No[target_rows]

    for (candidate in candidates) {
      fitted <- fit_predict_history_prior(
        source, target, candidate
      )
      history_mlogit <- fitted$pred[
        match(target_no, fitted$no),
        ,
        drop = FALSE
      ]
      candidate_fold <- 0.85 * (
        0.8 * history_mlogit +
          0.2 * base$oof_xgb[target_rows, , drop = FALSE]
      ) + 0.15 * shallow[target_rows, , drop = FALSE]
      candidate_oof[[candidate]][target_rows, ] <-
        candidate_fold
      fold_truth <- truth[target_rows, , drop = FALSE]
      fold_current <- current[target_rows, , drop = FALSE]
      fold_rows[[length(fold_rows) + 1L]] <- data.frame(
        candidate = candidate,
        fold = fold,
        baseline_loss =
          log_loss_matrix(fold_truth, fold_current),
        candidate_loss =
          log_loss_matrix(fold_truth, candidate_fold),
        gain =
          log_loss_matrix(fold_truth, fold_current) -
          log_loss_matrix(fold_truth, candidate_fold)
      )
      coefficient_rows[[length(coefficient_rows) + 1L]] <-
        data.frame(
          candidate = candidate,
          fold = fold,
          term = names(fitted$extra_coefficient),
          coefficient = unname(fitted$extra_coefficient)
        )
    }
  }

  summary_rows <- list()
  bootstrap_rows <- list()
  for (candidate in candidates) {
    prediction <- candidate_oof[[candidate]]
    stopifnot(!anyNA(prediction))
    bootstrap <- history_prior_bootstrap(
      truth, current, prediction, train$Case
    )
    summary_rows[[candidate]] <- data.frame(
      candidate = candidate,
      family_size = history_prior_family_size,
      baseline_loss = log_loss_matrix(truth, current),
      candidate_loss = log_loss_matrix(truth, prediction),
      gain =
        log_loss_matrix(truth, current) -
        log_loss_matrix(truth, prediction),
      folds_improved = sum(
        do.call(rbind, fold_rows)$candidate == candidate &
          do.call(rbind, fold_rows)$gain > 0
      )
    )
    bootstrap_rows[[candidate]] <- cbind(
      data.frame(candidate = candidate),
      bootstrap
    )
  }
  summary <- do.call(rbind, summary_rows)
  bootstrap <- do.call(rbind, bootstrap_rows)
  write.csv(
    do.call(rbind, fold_rows),
    file.path(
      history_prior_output_dir,
      "history_prior_cv_folds.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    do.call(rbind, coefficient_rows),
    file.path(
      history_prior_output_dir,
      "history_prior_cv_coefficients.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    summary,
    file.path(
      history_prior_output_dir,
      "history_prior_cv.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    bootstrap,
    file.path(
      history_prior_output_dir,
      "history_prior_bootstrap.csv"
    ),
    row.names = FALSE
  )
  saveRDS(
    list(
      candidates = candidates,
      predictions = candidate_oof,
      summary = summary,
      folds = do.call(rbind, fold_rows),
      coefficients = do.call(rbind, coefficient_rows),
      bootstrap = bootstrap,
      baseline = current,
      truth = truth,
      case = train$Case,
      no = train$No,
      row_fold = row_fold,
      family_size = history_prior_family_size
    ),
    file.path(
      history_prior_output_dir,
      "history_prior_cv.rds"
    )
  )
  print(summary, digits = 10)
  print(bootstrap, digits = 10)
}
