## Penalized partial pooling for segment-specific price sensitivity to mileage
## and income. The full m8trpg design remains unpenalized; five segment
## deviations per covariate are shrunk by nested respondent-grouped CV.

suppressPackageStartupMessages({
  library(glmnet)
  library(survival)
  library(mlogit)
  library(dfidx)
})
source("R/codex_shift_common.R")

stage <- Sys.getenv("CODEX_STAGE", "screen")
stopifnot(stage %in% c("screen", "cv"))

output_dir <- "data_processed/codex_final_round"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

alpha_grid <- c(0, 0.25, 0.5, 0.75, 1)
candidate_specs <- list(
  segment_miles_shrunk = "milesa",
  segment_income_shrunk = "incomea",
  segment_miles_income_shrunk = c("milesa", "incomea")
)

make_choice_design <- function(long_df, ctr, scl, candidate_vars,
                               candidate_scale = NULL) {
  long_df <- as.data.frame(long_df)
  if (!("chid" %in% names(long_df))) {
    long_df$chid <- paste(long_df$Case, long_df$Task, sep = "_")
  }
  if (!("d2" %in% names(long_df))) {
    long_df$d2 <- as.integer(long_df$alt == 2L)
  }
  if (!("d3" %in% names(long_df))) {
    long_df$d3 <- as.integer(long_df$alt == 3L)
  }

  feat <- make_m8trpg_features(long_df, ctr, scl, "none")
  fml <- m8trpg_formula("none")
  environment(fml) <- environment()
  mdat <- dfidx(
    feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen"
  )
  class(mdat) <- c("dfidx_mlogit", class(mdat))
  mf <- model.frame(mdat, fml, balanced = TRUE)
  core <- model.matrix(mf, rhs = 1:3)

  matrix_chid <- as.character(dfidx::idx(mf, 1))
  matrix_alt <- as.integer(as.character(dfidx::idx(mf, 2)))
  feature_key <- paste(feat$chid, feat$alt, sep = "::")
  matrix_key <- paste(matrix_chid, matrix_alt, sep = "::")
  feature_order <- match(matrix_key, feature_key)
  stopifnot(!anyNA(feature_order), !anyDuplicated(feature_order))
  aligned <- feat[feature_order, , drop = FALSE]

  candidate <- list()
  for (v in candidate_vars) {
    short <- sub("a$", "", v)
    z <- (as.numeric(aligned[[v]]) - ctr[[v]]) / scl[[v]]
    for (segment in 2:6) {
      candidate[[paste0("P_", short, "_seg", segment)]] <-
        aligned$Price_num * z *
        as.integer(aligned$segmentind == segment)
    }
  }
  candidate <- do.call(cbind, candidate)
  storage.mode(candidate) <- "double"

  if (is.null(candidate_scale)) {
    candidate_scale <- apply(candidate, 2, sd)
    candidate_scale[
      !is.finite(candidate_scale) | candidate_scale == 0
    ] <- 1
  }
  candidate <- sweep(candidate, 2, candidate_scale, "/")
  x <- cbind(core, candidate)
  storage.mode(x) <- "double"

  list(
    x = x,
    y = as.integer(aligned$chosen),
    chid = as.character(aligned$chid),
    case = as.integer(aligned$Case),
    alt = as.integer(aligned$alt),
    no = as.integer(aligned$No),
    candidate_names = colnames(candidate),
    candidate_scale = candidate_scale,
    n_core = ncol(core),
    penalty = c(rep(0, ncol(core)), rep(1, ncol(candidate)))
  )
}

inner_fold_id <- function(case, seed) {
  respondents <- sort(unique(case))
  set.seed(seed)
  fold <- sample(rep(1:5, length.out = length(respondents)))
  names(fold) <- respondents
  unname(fold[as.character(case)])
}

fit_nested_penalty <- function(train_long, valid_long, candidate_vars,
                               seed) {
  scaler <- choice_scaler(train_long)
  tr <- make_choice_design(
    train_long, scaler$ctr, scaler$scl, candidate_vars
  )
  va <- make_choice_design(
    valid_long, scaler$ctr, scaler$scl, candidate_vars,
    candidate_scale = tr$candidate_scale
  )
  stopifnot(
    identical(colnames(tr$x), colnames(va$x)),
    identical(tr$candidate_names, va$candidate_names)
  )

  y <- stratifySurv(
    Surv(rep(1, length(tr$y)), tr$y),
    as.factor(tr$chid)
  )
  foldid <- inner_fold_id(tr$case, seed)
  fits <- vector("list", length(alpha_grid))
  tuning <- vector("list", length(alpha_grid))

  for (i in seq_along(alpha_grid)) {
    alpha <- alpha_grid[[i]]
    cvfit <- cv.glmnet(
      x = tr$x, y = y, family = "cox", alpha = alpha,
      penalty.factor = tr$penalty, standardize = FALSE,
      foldid = foldid, type.measure = "deviance",
      grouped = FALSE, nlambda = 60, cox.ties = "breslow"
    )
    min_index <- which.min(cvfit$cvm)
    coef_min <- as.matrix(coef(
      cvfit$glmnet.fit, s = cvfit$lambda.min
    ))
    selected <- sum(
      abs(coef_min[
        (tr$n_core + 1L):nrow(coef_min), 1
      ]) > 1e-10
    )
    fits[[i]] <- cvfit
    tuning[[i]] <- data.frame(
      alpha = alpha,
      lambda = cvfit$lambda.min,
      inner_deviance = cvfit$cvm[[min_index]],
      inner_deviance_se = cvfit$cvsd[[min_index]],
      selected_candidate_terms = selected
    )
  }

  tuning <- do.call(rbind, tuning)
  best_index <- which.min(tuning$inner_deviance)
  best_fit <- fits[[best_index]]
  eta <- as.numeric(predict(
    best_fit$glmnet.fit,
    newx = va$x,
    s = best_fit$lambda.min,
    type = "link"
  ))

  task_map <- unique(data.frame(
    chid = va$chid,
    no = va$no,
    stringsAsFactors = FALSE
  ))
  task_map <- task_map[order(task_map$no), , drop = FALSE]
  eta_order <- order(match(va$chid, task_map$chid), va$alt)
  stopifnot(
    identical(va$chid[eta_order], rep(task_map$chid, each = 4)),
    identical(va$alt[eta_order], rep(1:4, times = nrow(task_map)))
  )
  pred <- softmax_margins(eta[eta_order])
  rownames(pred) <- task_map$chid

  best_coef <- as.matrix(coef(
    best_fit$glmnet.fit, s = best_fit$lambda.min
  ))
  candidate_coef <- best_coef[
    va$candidate_names, 1, drop = TRUE
  ] / tr$candidate_scale

  list(
    pred = pred,
    no = task_map$no,
    tuning = tuning,
    selected_alpha = tuning$alpha[[best_index]],
    selected_lambda = tuning$lambda[[best_index]],
    candidate_coef = candidate_coef
  )
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

if (stage == "screen") {
  split <- readRDS("data_processed/train_val_split.rds")
  tr <- split$train_long_tr
  va <- split$train_long_val
  tr$chid <- paste(tr$Case, tr$Task, sep = "_")
  va$chid <- paste(va$Case, va$Task, sep = "_")
  tr$d2 <- as.integer(tr$alt == 2L)
  tr$d3 <- as.integer(tr$alt == 3L)
  va$d2 <- as.integer(va$alt == 2L)
  va$d3 <- as.integer(va$alt == 3L)
  va_wide <- split$train_wide_val
  va_wide <- va_wide[order(va_wide$No), , drop = FALSE]
  truth <- as.matrix(va_wide[, paste0("Ch", 1:4)])

  baseline_fit <- fit_predict_m8trpg(tr, va)
  baseline <- baseline_fit$pred[
    match(va_wide$No, baseline_fit$no), , drop = FALSE
  ]
  baseline_loss <- log_loss_matrix(truth, baseline)
  stopifnot(abs(baseline_loss - 1.15968144721113) < 1e-8)

  results <- list()
  tuning_rows <- list()
  coefficient_rows <- list()
  prediction_list <- list()
  for (candidate in names(candidate_specs)) {
    fitted <- fit_nested_penalty(
      tr, va, candidate_specs[[candidate]], seed = 4821
    )
    idx <- match(va_wide$No, fitted$no)
    pred <- fitted$pred[idx, , drop = FALSE]
    loss <- log_loss_matrix(truth, pred)
    results[[candidate]] <- data.frame(
      candidate = candidate,
      n_candidate_terms = length(fitted$candidate_coef),
      selected_alpha = fitted$selected_alpha,
      selected_lambda = fitted$selected_lambda,
      baseline_logloss = baseline_loss,
      candidate_logloss = loss,
      gain = baseline_loss - loss
    )
    tuning_rows[[candidate]] <- cbind(
      data.frame(candidate = candidate), fitted$tuning
    )
    coefficient_rows[[candidate]] <- data.frame(
      candidate = candidate,
      term = names(fitted$candidate_coef),
      coefficient = unname(fitted$candidate_coef)
    )
    prediction_list[[candidate]] <- pred
    cat(sprintf(
      "%s: alpha %.2f; val %.6f; gain %+.6f\n",
      candidate, fitted$selected_alpha, loss, baseline_loss - loss
    ))
  }

  result <- do.call(rbind, results)
  write_result_csv(
    result,
    file.path(output_dir, "partial_pooling_screen.csv")
  )
  write_result_csv(
    do.call(rbind, tuning_rows),
    file.path(output_dir, "partial_pooling_screen_tuning.csv")
  )
  write_result_csv(
    do.call(rbind, coefficient_rows),
    file.path(output_dir, "partial_pooling_screen_coefficients.csv")
  )
  saveRDS(
    list(
      result = result,
      predictions = prediction_list,
      baseline = baseline,
      validation_no = va_wide$No
    ),
    file.path(output_dir, "partial_pooling_screen.rds")
  )
  print(result, digits = 7)
}

if (stage == "cv") {
  screen <- read.csv(
    file.path(output_dir, "partial_pooling_screen.csv")
  )
  requested <- Sys.getenv("CODEX_CANDIDATES", "")
  if (nzchar(requested)) {
    candidates <- strsplit(requested, ",", fixed = TRUE)[[1]]
  } else {
    candidates <- screen$candidate[screen$gain > 0]
  }
  if (length(candidates) == 0L) {
    cat("No partial-pooling candidate passed the screen; CV skipped.\n")
    quit(save = "no", status = 0)
  }
  stopifnot(all(candidates %in% names(candidate_specs)))

  train <- read.csv("csv files/train.csv")
  truth <- as.matrix(train[, paste0("Ch", 1:4)])
  train_long <- reshape_choice_long(train)
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
  tuning_rows <- list()
  coefficient_rows <- list()

  for (candidate in candidates) {
    for (fold in 1:5) {
      val_cases <- as.integer(names(fold_map)[fold_map == fold])
      tr <- train_long[
        !(train_long$Case %in% val_cases), , drop = FALSE
      ]
      va <- train_long[
        train_long$Case %in% val_cases, , drop = FALSE
      ]
      fitted <- fit_nested_penalty(
        tr, va, candidate_specs[[candidate]], seed = 4821 + fold
      )
      idx <- match(fitted$no, train$No)
      stopifnot(!anyNA(idx))
      candidate_oof[[candidate]][idx, ] <- fitted$pred
      base_fold <- log_loss_matrix(truth[idx, ], baseline[idx, ])
      candidate_fold <- log_loss_matrix(
        truth[idx, ], fitted$pred
      )
      fold_rows[[length(fold_rows) + 1L]] <- data.frame(
        candidate = candidate,
        fold = fold,
        selected_alpha = fitted$selected_alpha,
        selected_lambda = fitted$selected_lambda,
        baseline_logloss = base_fold,
        candidate_logloss = candidate_fold,
        gain = base_fold - candidate_fold
      )
      tuning_rows[[length(tuning_rows) + 1L]] <- cbind(
        data.frame(candidate = candidate, fold = fold),
        fitted$tuning
      )
      coefficient_rows[[length(coefficient_rows) + 1L]] <-
        data.frame(
          candidate = candidate,
          fold = fold,
          term = names(fitted$candidate_coef),
          coefficient = unname(fitted$candidate_coef)
        )
      cat(sprintf(
        "%s fold %d: alpha %.2f; gain %+.6f\n",
        candidate, fold, fitted$selected_alpha,
        base_fold - candidate_fold
      ))
    }
    stopifnot(!anyNA(candidate_oof[[candidate]]))
  }

  summary_rows <- list()
  bootstrap_rows <- list()
  for (candidate in candidates) {
    pred <- candidate_oof[[candidate]]
    fixed_blend <- 0.8 * pred + 0.2 * xgb
    summary_rows[[candidate]] <- data.frame(
      candidate = candidate,
      baseline_mlogit_logloss = log_loss_matrix(truth, baseline),
      candidate_mlogit_logloss = log_loss_matrix(truth, pred),
      mlogit_gain =
        log_loss_matrix(truth, baseline) -
        log_loss_matrix(truth, pred),
      v11_logloss = log_loss_matrix(truth, v11),
      fixed_080_blend_logloss =
        log_loss_matrix(truth, fixed_blend),
      fixed_080_blend_gain =
        log_loss_matrix(truth, v11) -
        log_loss_matrix(truth, fixed_blend)
    )
    for (comparison in c("mlogit", "fixed_080_blend")) {
      if (comparison == "mlogit") {
        base_pred <- baseline
        candidate_pred <- pred
      } else {
        base_pred <- v11
        candidate_pred <- fixed_blend
      }
      bootstrap_rows[[length(bootstrap_rows) + 1L]] <- cbind(
        data.frame(
          candidate = candidate,
          comparison = comparison
        ),
        respondent_bootstrap_gain(
          truth, base_pred, candidate_pred, train$Case
        )
      )
    }
  }

  summary <- do.call(rbind, summary_rows)
  bootstrap <- do.call(rbind, bootstrap_rows)
  write_result_csv(
    do.call(rbind, fold_rows),
    file.path(output_dir, "partial_pooling_cv_folds.csv")
  )
  write_result_csv(
    do.call(rbind, tuning_rows),
    file.path(output_dir, "partial_pooling_cv_tuning.csv")
  )
  write_result_csv(
    do.call(rbind, coefficient_rows),
    file.path(output_dir, "partial_pooling_cv_coefficients.csv")
  )
  write_result_csv(
    summary,
    file.path(output_dir, "partial_pooling_cv.csv")
  )
  write_result_csv(
    bootstrap,
    file.path(output_dir, "partial_pooling_bootstrap.csv")
  )
  saveRDS(
    list(
      candidates = candidates,
      oof = candidate_oof,
      summary = summary,
      bootstrap = bootstrap
    ),
    file.path(output_dir, "partial_pooling_oof.rds")
  )
  print(summary, digits = 7)
  print(bootstrap, digits = 7)
}
