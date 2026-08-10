suppressPackageStartupMessages({
  library(xgboost)
  library(mlogit)
  library(dfidx)
})
source("R/codex_shift_common.R")

shared_utility_output_dir <- "data_processed/codex_shared_utility"

ensure_choice_long <- function(df) {
  df <- as.data.frame(df)
  df$alt <- as.integer(df$alt)
  if (!("chid" %in% names(df))) {
    df$chid <- paste(df$Case, df$Task, sep = "_")
  }
  if (!("d2" %in% names(df))) {
    df$d2 <- as.integer(df$alt == 2L)
  }
  if (!("d3" %in% names(df))) {
    df$d3 <- as.integer(df$alt == 3L)
  }
  sort_long_tasks(df)
}

group_softmax_vector <- function(margin) {
  stopifnot(length(margin) %% 4L == 0L)
  matrix_prob <- softmax_margins(margin)
  as.vector(t(matrix_prob))
}

group_softmax_loss <- function(margin, label) {
  stopifnot(
    length(margin) == length(label),
    length(margin) %% 4L == 0L
  )
  truth <- matrix(label, ncol = 4L, byrow = TRUE)
  stopifnot(all(rowSums(truth) == 1L))
  log_loss_matrix(truth, softmax_margins(margin))
}

group_softmax_derivatives <- function(margin, label) {
  stopifnot(
    length(margin) == length(label),
    length(margin) %% 4L == 0L,
    all(rowSums(matrix(label, ncol = 4L, byrow = TRUE)) == 1L)
  )
  probability <- group_softmax_vector(margin)
  # The exact Hessian has negative cross-alternative entries. XGBoost accepts
  # only one Hessian value per long-format row, so use the same diagonally
  # dominant upper-bound construction as its multinomial objective:
  # diag_j = |H_jj| + sum_{k != j} |H_jk| = 2 p_j (1 - p_j).
  list(
    grad = probability - label,
    hess = pmax(2 * probability * (1 - probability), 1e-8)
  )
}

group_softmax_objective <- function(prediction, dtrain) {
  label <- getinfo(dtrain, "label")
  group_softmax_derivatives(prediction, label)
}

group_softmax_metric <- function(prediction, dtrain) {
  label <- getinfo(dtrain, "label")
  list(
    metric = "group_logloss",
    value = group_softmax_loss(prediction, label)
  )
}

check_group_softmax_gradient <- function(seed = 4821L,
                                         epsilon = 1e-6) {
  set.seed(seed)
  margin <- rnorm(28L)
  chosen <- sample.int(4L, length(margin) / 4L, replace = TRUE)
  label <- as.vector(t(diag(4L)[chosen, , drop = FALSE]))
  analytic <- group_softmax_derivatives(margin, label)$grad /
    (length(margin) / 4L)
  finite_difference <- vapply(seq_along(margin), function(index) {
    plus <- margin
    minus <- margin
    plus[index] <- plus[index] + epsilon
    minus[index] <- minus[index] - epsilon
    (
      group_softmax_loss(plus, label) -
        group_softmax_loss(minus, label)
    ) / (2 * epsilon)
  }, numeric(1))
  error <- analytic - finite_difference
  data.frame(
    seed = seed,
    epsilon = epsilon,
    max_absolute_error = max(abs(error)),
    mean_absolute_error = mean(abs(error)),
    passed = max(abs(error)) < 1e-7
  )
}

shared_feature_matrix <- function(long_df) {
  long_df <- ensure_choice_long(long_df)
  rank_feature_matrix(long_df)
}

make_group_dmatrix <- function(long_df, base_margin = NULL,
                               include_label = TRUE) {
  long_df <- ensure_choice_long(long_df)
  stopifnot(
    all(table(long_df$No) == 4L),
    identical(
      as.integer(long_df$alt),
      rep(1:4, times = nrow(long_df) / 4L)
    )
  )
  label <- if (include_label) {
    as.numeric(long_df$chosen)
  } else {
    NULL
  }
  qid <- as.integer(factor(
    long_df$No,
    levels = unique(long_df$No)
  ))
  dmatrix <- xgb.DMatrix(
    shared_feature_matrix(long_df),
    label = label,
    qid = qid,
    nthread = 1L
  )
  if (!is.null(base_margin)) {
    stopifnot(length(base_margin) == nrow(long_df))
    setinfo(dmatrix, "base_margin", as.numeric(base_margin))
  }
  dmatrix
}

shared_xgb_params <- function(max_depth, eta, min_child_weight,
                              lambda, alpha = 0,
                              subsample = 0.8,
                              colsample_bytree = 0.8,
                              seed = 4821L) {
  list(
    max_depth = as.integer(max_depth),
    eta = eta,
    min_child_weight = min_child_weight,
    lambda = lambda,
    alpha = alpha,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    tree_method = "hist",
    base_score = 0,
    disable_default_eval_metric = TRUE,
    seed = as.integer(seed),
    nthread = 1L
  )
}

fit_group_softmax_xgb <- function(train_long, params, nrounds,
                                  base_margin = NULL) {
  dtrain <- make_group_dmatrix(
    train_long,
    base_margin = base_margin,
    include_label = TRUE
  )
  xgb.train(
    params = params,
    data = dtrain,
    nrounds = as.integer(nrounds),
    objective = group_softmax_objective,
    custom_metric = group_softmax_metric,
    verbose = 0
  )
}

predict_group_softmax_xgb <- function(model, valid_long,
                                      base_margin = NULL,
                                      nrounds = NULL) {
  dvalid <- make_group_dmatrix(
    valid_long,
    base_margin = base_margin,
    include_label = FALSE
  )
  iterationrange <- if (is.null(nrounds)) {
    NULL
  } else {
    c(1L, as.integer(nrounds))
  }
  margin <- as.numeric(predict(
    model,
    dvalid,
    outputmargin = TRUE,
    iterationrange = iterationrange
  ))
  list(
    margin = margin,
    pred = softmax_margins(margin)
  )
}

fit_m8trpg_model <- function(train_long) {
  train_long <- ensure_choice_long(train_long)
  scaler <- choice_scaler(train_long)
  train_features <- make_m8trpg_features(
    train_long, scaler$ctr, scaler$scl, "none"
  )
  formula <- m8trpg_formula("none")
  environment(formula) <- environment()
  model <- mlogit(
    formula,
    data = train_features,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen"
  )
  list(
    model = model,
    scaler = scaler,
    formula = formula,
    train_features = train_features
  )
}

predict_m8trpg_margin <- function(fitted, long_df) {
  long_df <- ensure_choice_long(long_df)
  features <- make_m8trpg_features(
    long_df,
    fitted$scaler$ctr,
    fitted$scaler$scl,
    "none"
  )
  indexed <- dfidx(
    features,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen"
  )
  class(indexed) <- c("dfidx_mlogit", class(indexed))
  model_frame <- model.frame(
    indexed, fitted$formula, balanced = TRUE
  )
  design <- model.matrix(model_frame, rhs = 1:3)
  coefficient <- coef(fitted$model)
  stopifnot(all(names(coefficient) %in% colnames(design)))
  raw_margin <- as.numeric(
    design[, names(coefficient), drop = FALSE] %*% coefficient
  )

  task_map <- unique(features[, c("chid", "No")])
  task_map <- task_map[order(task_map$No), , drop = FALSE]
  margin_chid <- as.character(dfidx::idx(model_frame, 1))
  margin_alt <- as.integer(as.character(dfidx::idx(model_frame, 2)))
  margin_order <- order(
    match(margin_chid, task_map$chid),
    margin_alt
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
  margin <- raw_margin[margin_order]
  pred <- softmax_margins(margin)
  list(
    margin = margin,
    pred = pred,
    no = task_map$No,
    features = features
  )
}

aligned_prediction <- function(prediction_result, no_order) {
  prediction_result$pred[
    match(no_order, prediction_result$no), ,
    drop = FALSE
  ]
}

row_log_loss <- function(truth, prediction) {
  prediction <- prediction / rowSums(prediction)
  prediction <- pmin(pmax(prediction, 1e-15), 1 - 1e-15)
  -rowSums(as.matrix(truth) * log(prediction))
}

crossfit_two_way_blend <- function(baseline, candidate, truth,
                                   row_fold,
                                   weights = seq(0, 0.4, by = 0.01)) {
  prediction <- matrix(NA_real_, nrow(truth), ncol(truth))
  selected_weight <- numeric(5L)
  for (fold in 1:5) {
    fit_rows <- row_fold != fold
    valid_rows <- row_fold == fold
    loss <- vapply(weights, function(weight) {
      log_loss_matrix(
        truth[fit_rows, , drop = FALSE],
        (1 - weight) * baseline[fit_rows, , drop = FALSE] +
          weight * candidate[fit_rows, , drop = FALSE]
      )
    }, numeric(1))
    best <- which.min(loss)
    selected_weight[fold] <- weights[best]
    prediction[valid_rows, ] <-
      (1 - weights[best]) *
        baseline[valid_rows, , drop = FALSE] +
      weights[best] *
        candidate[valid_rows, , drop = FALSE]
  }
  stopifnot(!anyNA(prediction))
  list(
    pred = prediction,
    fold_weights = selected_weight,
    logloss = log_loss_matrix(truth, prediction)
  )
}

respondent_bootstrap_comparison <- function(truth, baseline, candidate,
                                            case,
                                            seed = 4821L,
                                            replicates = 100000L) {
  gain <- tapply(
    row_log_loss(truth, baseline) -
      row_log_loss(truth, candidate),
    case,
    mean
  )
  set.seed(seed)
  boot <- replicate(
    replicates,
    mean(sample(gain, length(gain), replace = TRUE))
  )
  data.frame(
    point_gain = mean(gain),
    bootstrap_mean = mean(boot),
    bootstrap_sd = sd(boot),
    lower_95 = unname(quantile(boot, 0.025)),
    upper_95 = unname(quantile(boot, 0.975)),
    lower_99 = unname(quantile(boot, 0.005)),
    upper_99 = unname(quantile(boot, 0.995)),
    win_rate = mean(boot > 0),
    n_boot = replicates
  )
}

scale_softmax_evaluate <- function(gamma, utility, truth, design,
                                   ridge = 0) {
  log_scale <- as.numeric(design %*% gamma)
  scale <- exp(log_scale)
  stopifnot(all(is.finite(scale)))
  scaled_utility <- utility * scale
  shifted <- scaled_utility - apply(scaled_utility, 1, max)
  exponential <- exp(shifted)
  probability <- exponential / rowSums(exponential)
  score <- scale * rowSums((probability - truth) * utility)
  penalty_mask <- c(0, rep(1, length(gamma) - 1L))
  list(
    value = log_loss_matrix(truth, probability) +
      0.5 * ridge * sum((gamma * penalty_mask)^2),
    gradient = colMeans(design * score) +
      ridge * gamma * penalty_mask,
    probability = probability,
    scale = scale
  )
}

check_scale_gradient <- function(seed = 4821L, epsilon = 1e-6) {
  set.seed(seed)
  utility <- matrix(rnorm(40L), ncol = 4L)
  chosen <- sample.int(4L, nrow(utility), replace = TRUE)
  truth <- diag(4L)[chosen, , drop = FALSE]
  design <- cbind(1, matrix(rnorm(nrow(utility) * 3L), ncol = 3L))
  gamma <- rnorm(ncol(design), sd = 0.1)
  ridge <- 0.003
  analytic <- scale_softmax_evaluate(
    gamma, utility, truth, design, ridge
  )$gradient
  finite_difference <- vapply(seq_along(gamma), function(index) {
    plus <- gamma
    minus <- gamma
    plus[index] <- plus[index] + epsilon
    minus[index] <- minus[index] - epsilon
    (
      scale_softmax_evaluate(
        plus, utility, truth, design, ridge
      )$value -
        scale_softmax_evaluate(
          minus, utility, truth, design, ridge
        )$value
    ) / (2 * epsilon)
  }, numeric(1))
  error <- analytic - finite_difference
  data.frame(
    seed = seed,
    epsilon = epsilon,
    max_absolute_error = max(abs(error)),
    mean_absolute_error = mean(abs(error)),
    passed = max(abs(error)) < 1e-7
  )
}

fit_scale_model <- function(utility, truth, design, ridge = 0) {
  objective <- function(gamma) {
    scale_softmax_evaluate(
      gamma, utility, truth, design, ridge
    )$value
  }
  gradient <- function(gamma) {
    scale_softmax_evaluate(
      gamma, utility, truth, design, ridge
    )$gradient
  }
  fitted <- optim(
    rep(0, ncol(design)),
    objective,
    gr = gradient,
    method = "BFGS",
    control = list(maxit = 1000, reltol = 1e-12)
  )
  stopifnot(fitted$convergence == 0L)
  fitted
}
