## Scoped Random Regret Minimization (RRM) models implemented directly because
## the Apollo package is not installed. The likelihood follows Chorus (2010):
## R_i = sum_{j != i} sum_m log(1 + exp(beta_m (x_jm - x_im))),
## P_i = exp(-R_i) / sum_j exp(-R_j).

suppressPackageStartupMessages({
  library(mlogit)
  library(dfidx)
  library(xgboost)
})
source("R/codex_shift_common.R")

stage <- Sys.getenv("CODEX_STAGE", "check")
stopifnot(stage %in% c("check", "screen", "reweight", "cv"))

output_dir <- "data_processed/codex_behavioral_round"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

position_terms <- c("d2", "d3")
observed_context_terms <- c(
  position_terms,
  "P_income", "P_age", "P_miles", "P_night",
  "In_income", "In_age", "In_miles", "In_night",
  "In_gender", "In_urb", "In_educ",
  paste0("P_seg", 2:6), paste0("In_seg", 2:6),
  "P_task", "In_task",
  paste0("P_region", 2:5), paste0("In_region", 2:5),
  paste0("P_ppark", 2:5), paste0("In_ppark", 2:5)
)
price_context_terms <- c(
  "is_cheapest", "is_dearest",
  "price_gap_min", "price_gap_max"
)

rrm_specs <- list(
  rrm_core_linear_price = list(
    price_mode = "linear",
    context_terms = position_terms
  ),
  rrm_observed_linear_price = list(
    price_mode = "linear",
    context_terms = observed_context_terms
  ),
  rrm_observed_factor_price = list(
    price_mode = "factor",
    context_terms = observed_context_terms
  ),
  rrm_full_linear_price = list(
    price_mode = "linear",
    context_terms = c(
      observed_context_terms, price_context_terms
    )
  ),
  rrm_full_factor_price = list(
    price_mode = "factor",
    context_terms = c(
      observed_context_terms, price_context_terms
    )
  )
)

prepare_rrm_long <- function(df) {
  df <- as.data.frame(df)
  if (!("chid" %in% names(df))) {
    df$chid <- paste(df$Case, df$Task, sep = "_")
  }
  if (!("d2" %in% names(df))) df$d2 <- as.integer(df$alt == 2L)
  if (!("d3" %in% names(df))) df$d3 <- as.integer(df$alt == 3L)
  df[order(df$No, df$alt), , drop = FALSE]
}

build_rrm_matrices <- function(long_df, scaler, attr_max, spec) {
  long_df <- prepare_rrm_long(long_df)
  feat <- make_m8trpg_features(
    long_df, scaler$ctr, scaler$scl, "none"
  )
  stopifnot(
    nrow(feat) %% 4L == 0L,
    identical(
      as.integer(feat$alt),
      rep(1:4, times = nrow(feat) / 4L)
    )
  )
  n_tasks <- nrow(feat) / 4L

  regret_features <- list()
  for (attribute in attrs) {
    for (level in seq_len(attr_max[[attribute]])) {
      regret_features[[
        paste0(attribute, "_", level)
      ]] <- as.numeric(feat[[attribute]] == level)
    }
  }
  if (spec$price_mode == "linear") {
    regret_features$Price_scaled <- feat$Price_num / 12
  } else {
    for (level in 2:12) {
      regret_features[[paste0("Price_", level)]] <-
        as.numeric(feat$Price_num == level)
    }
  }
  x_long <- do.call(cbind, regret_features)
  storage.mode(x_long) <- "double"

  z_long <- as.matrix(
    feat[, spec$context_terms, drop = FALSE]
  )
  storage.mode(z_long) <- "double"

  x_alt <- lapply(1:4, function(alternative) {
    x_long[feat$alt == alternative, , drop = FALSE]
  })
  z_alt <- lapply(1:4, function(alternative) {
    z_long[feat$alt == alternative, , drop = FALSE]
  })
  truth <- matrix(
    as.integer(feat$chosen), ncol = 4, byrow = TRUE
  )
  stopifnot(all(rowSums(truth) == 1L))

  list(
    x = x_alt,
    z = z_alt,
    truth = truth,
    no = as.integer(feat$No[feat$alt == 1L]),
    case = as.integer(feat$Case[feat$alt == 1L]),
    x_names = colnames(x_long),
    z_names = colnames(z_long),
    n_tasks = n_tasks,
    n_beta = ncol(x_long),
    n_gamma = ncol(z_long)
  )
}

stable_softplus <- function(x) {
  pmax(x, 0) + log1p(exp(-abs(x)))
}

stable_sigmoid <- function(x) {
  positive <- x >= 0
  out <- numeric(length(x))
  out[positive] <- 1 / (1 + exp(-x[positive]))
  exp_x <- exp(x[!positive])
  out[!positive] <- exp_x / (1 + exp_x)
  dim(out) <- dim(x)
  out
}

rrm_scores_and_derivatives <- function(par, data,
                                       need_gradient = TRUE) {
  beta <- par[seq_len(data$n_beta)]
  gamma <- par[
    data$n_beta + seq_len(data$n_gamma)
  ]
  regret <- matrix(0, data$n_tasks, 4)
  if (need_gradient) {
    regret_derivative <- array(
      0,
      dim = c(data$n_tasks, 4, data$n_beta)
    )
  }

  for (i in 1:4) {
    for (j in setdiff(1:4, i)) {
      difference <- data$x[[j]] - data$x[[i]]
      scaled_difference <- sweep(
        difference, 2, beta, "*"
      )
      regret[, i] <- regret[, i] +
        rowSums(stable_softplus(scaled_difference))
      if (need_gradient) {
        regret_derivative[, i, ] <-
          regret_derivative[, i, ] +
          stable_sigmoid(scaled_difference) * difference
      }
    }
  }

  score <- -regret
  for (i in 1:4) {
    score[, i] <- score[, i] +
      as.numeric(data$z[[i]] %*% gamma)
  }
  score <- score - apply(score, 1, max)
  probability <- exp(score)
  probability <- probability / rowSums(probability)

  out <- list(probability = probability)
  if (need_gradient) {
    residual <- data$truth - probability
    beta_gradient <- numeric(data$n_beta)
    gamma_gradient <- numeric(data$n_gamma)
    for (i in 1:4) {
      beta_gradient <- beta_gradient +
        colSums(
          regret_derivative[, i, , drop = FALSE][, 1, ] *
            residual[, i]
        )
      gamma_gradient <- gamma_gradient -
        colSums(data$z[[i]] * residual[, i])
    }
    out$gradient <- c(
      beta_gradient, gamma_gradient
    ) / data$n_tasks
  }
  out
}

make_rrm_objective <- function(data) {
  cache <- new.env(parent = emptyenv())
  evaluate <- function(par) {
    if (
      exists("par", envir = cache, inherits = FALSE) &&
        identical(par, cache$par)
    ) {
      return(cache$result)
    }
    calculated <- rrm_scores_and_derivatives(
      par, data, need_gradient = TRUE
    )
    probability <- pmin(
      pmax(calculated$probability, 1e-15),
      1 - 1e-15
    )
    value <- -mean(rowSums(
      data$truth * log(probability)
    ))
    result <- list(
      value = value,
      gradient = calculated$gradient,
      probability = calculated$probability
    )
    cache$par <- par
    cache$result <- result
    result
  }
  list(
    fn = function(par) evaluate(par)$value,
    gr = function(par) evaluate(par)$gradient,
    evaluate = evaluate
  )
}

fit_rrm <- function(data, seed, n_starts = 2L,
                    max_iterations = 250L) {
  objective <- make_rrm_objective(data)
  parameter_count <- data$n_beta + data$n_gamma
  set.seed(seed)
  starts <- vector("list", n_starts)
  starts[[1]] <- rep(0, parameter_count)
  if (n_starts > 1L) {
    for (start in 2:n_starts) {
      starts[[start]] <- rnorm(
        parameter_count, mean = 0, sd = 0.02
      )
    }
  }

  fits <- lapply(seq_along(starts), function(start) {
    started <- proc.time()[["elapsed"]]
    fitted <- optim(
      par = starts[[start]],
      fn = objective$fn,
      gr = objective$gr,
      method = "BFGS",
      control = list(
        maxit = max_iterations,
        reltol = 1e-9
      )
    )
    fitted$elapsed_seconds <-
      proc.time()[["elapsed"]] - started
    fitted$start <- start
    cat(sprintf(
      "RRM start %d: loss %.6f, convergence %d, %d fn evals, %.1fs\n",
      start, fitted$value, fitted$convergence,
      fitted$counts[["function"]],
      fitted$elapsed_seconds
    ))
    flush.console()
    fitted
  })
  best <- fits[[which.min(vapply(
    fits, function(x) x$value, numeric(1)
  ))]]
  list(best = best, all_fits = fits)
}

predict_rrm <- function(fit, data) {
  rrm_scores_and_derivatives(
    fit$best$par, data, need_gradient = FALSE
  )$probability
}

gradient_check <- function(data, seed = 4821,
                           checks = 12L, epsilon = 1e-6) {
  set.seed(seed)
  par <- rnorm(
    data$n_beta + data$n_gamma, sd = 0.03
  )
  objective <- make_rrm_objective(data)
  analytic <- objective$gr(par)
  indices <- sort(sample(
    seq_along(par), min(checks, length(par))
  ))
  numeric_gradient <- vapply(indices, function(index) {
    upper <- par
    lower <- par
    upper[[index]] <- upper[[index]] + epsilon
    lower[[index]] <- lower[[index]] - epsilon
    (
      objective$fn(upper) - objective$fn(lower)
    ) / (2 * epsilon)
  }, numeric(1))
  result <- data.frame(
    parameter_index = indices,
    analytic = analytic[indices],
    numeric = numeric_gradient,
    absolute_difference =
      abs(analytic[indices] - numeric_gradient)
  )
  stopifnot(max(result$absolute_difference) < 1e-6)
  result
}

xgb_screen_prediction <- function(split_data) {
  tr <- split_data$train_wide_tr
  va <- split_data$train_wide_val
  tr$Choice <- max.col(tr[, paste0("Ch", 1:4)])
  x_tr <- wide_feature_matrix(tr)
  x_va <- wide_feature_matrix(va)
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
    data = xgb.DMatrix(x_tr, label = tr$Choice - 1L),
    nrounds = 73,
    verbose = 0
  )
  pred <- predict(
    model, xgb.DMatrix(x_va),
    reshape = TRUE
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

if (stage == "check") {
  split_data <- readRDS("data_processed/train_val_split.rds")
  tr <- prepare_rrm_long(split_data$train_long_tr)
  scaler <- choice_scaler(tr)
  attr_max <- as.list(vapply(
    rbind(
      split_data$train_long_tr,
      split_data$train_long_val
    )[, attrs],
    max,
    numeric(1)
  ))
  small_cases <- head(sort(unique(tr$Case)), 8L)
  small <- tr[tr$Case %in% small_cases, , drop = FALSE]
  data <- build_rrm_matrices(
    small, scaler, attr_max,
    rrm_specs$rrm_observed_linear_price
  )
  result <- gradient_check(data)
  write_result_csv(
    result,
    file.path(output_dir, "rrm_gradient_check.csv")
  )
  print(result, digits = 10)
}

if (stage == "screen") {
  split_data <- readRDS("data_processed/train_val_split.rds")
  tr <- prepare_rrm_long(split_data$train_long_tr)
  va <- prepare_rrm_long(split_data$train_long_val)
  va_wide <- split_data$train_wide_val
  va_wide <- va_wide[order(va_wide$No), , drop = FALSE]
  truth <- as.matrix(va_wide[, paste0("Ch", 1:4)])
  scaler <- choice_scaler(tr)
  attr_max <- as.list(vapply(
    rbind(tr, va)[, attrs], max, numeric(1)
  ))

  baseline_fit <- fit_predict_m8trpg(tr, va)
  baseline <- baseline_fit$pred[
    match(va_wide$No, baseline_fit$no), , drop = FALSE
  ]
  baseline_loss <- log_loss_matrix(truth, baseline)
  stopifnot(abs(baseline_loss - 1.15968144721113) < 1e-8)
  xgb <- xgb_screen_prediction(split_data)
  v11_screen <- 0.8 * baseline + 0.2 * xgb
  v11_screen_loss <- log_loss_matrix(truth, v11_screen)

  # Allow the screen to reveal whether RRM deserves more than a token weight.
  # The full factor-price specification has a screen optimum above 0.30.
  weight_grid <- seq(0, 1, by = 0.01)
  result_rows <- list()
  fit_rows <- list()
  predictions <- list()
  models <- list()

  for (candidate in names(rrm_specs)) {
    spec <- rrm_specs[[candidate]]
    tr_data <- build_rrm_matrices(
      tr, scaler, attr_max, spec
    )
    va_data <- build_rrm_matrices(
      va, scaler, attr_max, spec
    )
    fit <- fit_rrm(
      tr_data, seed = 4821, n_starts = 2L
    )
    pred_raw <- predict_rrm(fit, va_data)
    pred <- pred_raw[
      match(va_wide$No, va_data$no), ,
      drop = FALSE
    ]
    stopifnot(!anyNA(pred))

    blend_losses <- vapply(weight_grid, function(weight) {
      log_loss_matrix(
        truth,
        (1 - weight) * v11_screen + weight * pred
      )
    }, numeric(1))
    best_index <- which.min(blend_losses)
    result_rows[[candidate]] <- data.frame(
      candidate = candidate,
      price_mode = spec$price_mode,
      n_regret_parameters = tr_data$n_beta,
      n_context_parameters = tr_data$n_gamma,
      training_logloss = fit$best$value,
      component_logloss = log_loss_matrix(truth, pred),
      m8trpg_logloss = baseline_loss,
      v11_screen_logloss = v11_screen_loss,
      best_rrm_weight = weight_grid[[best_index]],
      best_blend_logloss = blend_losses[[best_index]],
      blend_gain =
        v11_screen_loss - blend_losses[[best_index]]
    )
    fit_rows[[candidate]] <- do.call(rbind, lapply(
      fit$all_fits,
      function(x) {
        data.frame(
          candidate = candidate,
          start = x$start,
          training_logloss = x$value,
          convergence = x$convergence,
          function_evaluations = x$counts[["function"]],
          gradient_evaluations = x$counts[["gradient"]],
          elapsed_seconds = x$elapsed_seconds
        )
      }
    ))
    predictions[[candidate]] <- pred
    models[[candidate]] <- list(
      par = fit$best$par,
      x_names = tr_data$x_names,
      z_names = tr_data$z_names,
      spec = spec
    )
    cat(sprintf(
      "%s: component %.6f; best blend w %.2f, loss %.6f, gain %+.6f\n",
      candidate, log_loss_matrix(truth, pred),
      weight_grid[[best_index]], blend_losses[[best_index]],
      v11_screen_loss - blend_losses[[best_index]]
    ))
    flush.console()
  }

  result <- do.call(rbind, result_rows)
  write_result_csv(
    result, file.path(output_dir, "rrm_screen.csv")
  )
  write_result_csv(
    do.call(rbind, fit_rows),
    file.path(output_dir, "rrm_screen_fits.csv")
  )
  saveRDS(
    list(
      result = result,
      predictions = predictions,
      models = models,
      truth = truth,
      validation_no = va_wide$No,
      m8trpg = baseline,
      xgb = xgb,
      v11_screen = v11_screen
    ),
    file.path(output_dir, "rrm_screen.rds")
  )
  print(result, digits = 7)
}

if (stage == "reweight") {
  saved_screen <- readRDS(
    file.path(output_dir, "rrm_screen.rds")
  )
  weight_grid <- seq(0, 1, by = 0.01)
  baseline_loss <- log_loss_matrix(
    saved_screen$truth, saved_screen$v11_screen
  )
  rows <- lapply(
    names(saved_screen$predictions),
    function(candidate) {
      pred <- saved_screen$predictions[[candidate]]
      losses <- vapply(weight_grid, function(weight) {
        log_loss_matrix(
          saved_screen$truth,
          (1 - weight) * saved_screen$v11_screen +
            weight * pred
        )
      }, numeric(1))
      best <- which.min(losses)
      data.frame(
        candidate = candidate,
        best_rrm_weight = weight_grid[[best]],
        best_blend_logloss = losses[[best]],
        blend_gain = baseline_loss - losses[[best]]
      )
    }
  )
  result <- do.call(rbind, rows)
  write_result_csv(
    result,
    file.path(output_dir, "rrm_screen_reweighted.csv")
  )
  print(result, digits = 9)
}

if (stage == "cv") {
  screen <- read.csv(
    file.path(output_dir, "rrm_screen.csv"),
    stringsAsFactors = FALSE
  )
  requested <- Sys.getenv("CODEX_CANDIDATES", "")
  if (nzchar(requested)) {
    candidates <- strsplit(requested, ",", fixed = TRUE)[[1]]
  } else {
    candidates <- screen$candidate[
      screen$best_rrm_weight > 0 &
        screen$blend_gain > 0
    ]
  }
  if (length(candidates) == 0L) {
    cat("No RRM candidate improved the screen blend; CV skipped.\n")
    quit(save = "no", status = 0)
  }
  stopifnot(all(candidates %in% names(rrm_specs)))

  train <- read.csv("csv files/train.csv")
  truth <- as.matrix(train[, paste0("Ch", 1:4)])
  train_long <- prepare_rrm_long(reshape_choice_long(train))
  attr_max <- as.list(vapply(
    train_long[, attrs], max, numeric(1)
  ))
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
  fold_rows <- list()

  for (candidate in candidates) {
    spec <- rrm_specs[[candidate]]
    for (fold in 1:5) {
      val_cases <- as.integer(names(fold_map)[fold_map == fold])
      tr <- train_long[
        !(train_long$Case %in% val_cases), , drop = FALSE
      ]
      va <- train_long[
        train_long$Case %in% val_cases, , drop = FALSE
      ]
      scaler <- choice_scaler(tr)
      tr_data <- build_rrm_matrices(
        tr, scaler, attr_max, spec
      )
      va_data <- build_rrm_matrices(
        va, scaler, attr_max, spec
      )
      fit <- fit_rrm(
        tr_data, seed = 4821 + fold,
        n_starts = 1L
      )
      pred <- predict_rrm(fit, va_data)
      idx <- match(va_data$no, train$No)
      stopifnot(!anyNA(idx))
      candidate_oof[[candidate]][idx, ] <- pred
      fold_rows[[length(fold_rows) + 1L]] <- data.frame(
        candidate = candidate,
        fold = fold,
        training_logloss = fit$best$value,
        validation_logloss =
          log_loss_matrix(truth[idx, ], pred),
        convergence = fit$best$convergence,
        function_evaluations =
          fit$best$counts[["function"]],
        elapsed_seconds = fit$best$elapsed_seconds
      )
      saveRDS(
        list(
          candidate = candidate,
          fold = fold,
          prediction = pred,
          validation_index = idx,
          par = fit$best$par,
          x_names = tr_data$x_names,
          z_names = tr_data$z_names,
          fit_summary = fold_rows[[length(fold_rows)]]
        ),
        file.path(
          output_dir,
          sprintf("rrm_%s_fold_%d.rds", candidate, fold)
        )
      )
      cat(sprintf(
        "%s fold %d complete: validation %.6f\n",
        candidate, fold,
        fold_rows[[length(fold_rows)]]$validation_logloss
      ))
      flush.console()
    }
    stopifnot(!anyNA(candidate_oof[[candidate]]))
  }

  weight_grid <- seq(0, 1, by = 0.01)
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
        rrm_weight = best_weight,
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
    do.call(rbind, fold_rows),
    file.path(output_dir, "rrm_cv_folds.csv")
  )
  write_result_csv(
    do.call(rbind, weight_rows),
    file.path(output_dir, "rrm_cv_weights.csv")
  )
  write_result_csv(
    summary,
    file.path(output_dir, "rrm_cv.csv")
  )
  write_result_csv(
    bootstrap,
    file.path(output_dir, "rrm_bootstrap.csv")
  )
  saveRDS(
    list(
      candidates = candidates,
      oof = candidate_oof,
      summary = summary,
      bootstrap = bootstrap
    ),
    file.path(output_dir, "rrm_oof.rds")
  )
  print(summary, digits = 7)
  print(bootstrap, digits = 7)
}
