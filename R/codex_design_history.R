## Leakage-safe design-exposure history features. These use only alternatives
## shown in earlier tasks for the same respondent; no prior choices/outcomes
## enter feature construction.

suppressPackageStartupMessages({
  library(mlogit)
  library(dfidx)
})
source("R/codex_shift_common.R")

stage <- Sys.getenv("CODEX_STAGE", "screen")
stopifnot(stage %in% c("screen", "cv"))

output_dir <- "data_processed/codex_behavioral_round"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

history_candidate_specs <- list(
  price_running = "hist_price_gap_running",
  price_last_task = "hist_price_gap_last",
  price_running_and_last = c(
    "hist_price_gap_running", "hist_price_gap_last"
  ),
  price_reference_decomposed = c(
    "hist_price_gap_running", "hist_task_price_shift"
  ),
  attribute_unseen = "hist_attr_unseen_share",
  attribute_familiarity = "hist_attr_familiarity",
  attribute_similarity = "hist_attr_max_similarity",
  attribute_novelty = "hist_attr_novelty",
  attribute_recent_similarity = "hist_attr_recent_similarity",
  attribute_novelty_unseen = c(
    "hist_attr_novelty",
    "hist_attr_unseen_share"
  ),
  attribute_history = c(
    "hist_attr_unseen_share",
    "hist_attr_familiarity",
    "hist_attr_max_similarity"
  ),
  attribute_history_extended = c(
    "hist_attr_unseen_share",
    "hist_attr_familiarity",
    "hist_attr_max_similarity",
    "hist_attr_recent_similarity"
  ),
  all_design_history = c(
    "hist_price_gap_running",
    "hist_price_gap_last",
    "hist_task_price_shift",
    "hist_attr_unseen_share",
    "hist_attr_familiarity",
    "hist_attr_max_similarity",
    "hist_attr_recent_similarity"
  )
)

prepare_choice_ids <- function(df) {
  df <- as.data.frame(df)
  if (!("chid" %in% names(df))) {
    df$chid <- paste(df$Case, df$Task, sep = "_")
  }
  if (!("d2" %in% names(df))) df$d2 <- as.integer(df$alt == 2L)
  if (!("d3" %in% names(df))) df$d3 <- as.integer(df$alt == 3L)
  df
}

add_design_history_features <- function(df) {
  df <- prepare_choice_ids(df)
  original_order <- seq_len(nrow(df))
  df$.original_order <- original_order
  df <- df[order(df$Case, df$Task, df$alt), , drop = FALSE]

  feature_names <- unique(unlist(history_candidate_specs))
  for (feature in feature_names) df[[feature]] <- 0

  for (respondent in sort(unique(df$Case))) {
    respondent_rows <- which(df$Case == respondent)
    tasks <- sort(unique(df$Task[respondent_rows]))
    stopifnot(length(tasks) == 19L)

    prior_prices <- numeric()
    prior_attributes <- matrix(
      numeric(), nrow = 0L, ncol = length(attrs),
      dimnames = list(NULL, attrs)
    )
    previous_task_mean_price <- NA_real_
    previous_task_attributes <- NULL

    for (task_index in seq_along(tasks)) {
      task <- tasks[[task_index]]
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

      if (task_index > 1L) {
        prior_mean_price <- mean(prior_prices)
        current_task_mean_price <- mean(current_prices)

        df$hist_price_gap_running[inside_rows] <-
          current_prices - prior_mean_price
        df$hist_price_gap_last[inside_rows] <-
          current_prices - previous_task_mean_price
        df$hist_task_price_shift[inside_rows] <-
          current_task_mean_price - prior_mean_price

        for (alternative_index in 1:3) {
          row <- inside_rows[[alternative_index]]
          current <- current_attributes[alternative_index, ]
          active <- which(current != 0)
          stopifnot(length(active) == 9L)

          exposure_frequency <- vapply(
            active,
            function(attribute_index) {
              mean(
                prior_attributes[, attribute_index] ==
                  current[[attribute_index]]
              )
            },
            numeric(1)
          )
          active_matches <- sweep(
            prior_attributes[, active, drop = FALSE],
            2,
            current[active],
            "=="
          )
          prior_similarity <- rowMeans(active_matches)
          recent_matches <- sweep(
            previous_task_attributes[, active, drop = FALSE],
            2,
            current[active],
            "=="
          )
          recent_similarity <- rowMeans(recent_matches)

          df$hist_attr_unseen_share[[row]] <-
            mean(exposure_frequency == 0)
          df$hist_attr_familiarity[[row]] <-
            mean(exposure_frequency)
          df$hist_attr_max_similarity[[row]] <-
            max(prior_similarity)
          df$hist_attr_novelty[[row]] <-
            1 - max(prior_similarity)
          df$hist_attr_recent_similarity[[row]] <-
            max(recent_similarity)
        }
      }

      prior_prices <- c(prior_prices, current_prices)
      prior_attributes <- rbind(
        prior_attributes, current_attributes
      )
      previous_task_mean_price <- mean(current_prices)
      previous_task_attributes <- current_attributes
    }
  }

  # Alternative 4 must remain exactly zero for every history feature so that
  # each term is an identified inside-vs-opt-out / alternative-varying effect.
  stopifnot(all(
    as.matrix(df[df$alt == 4L, feature_names, drop = FALSE]) == 0
  ))
  df <- df[order(df$.original_order), , drop = FALSE]
  df$.original_order <- NULL
  rownames(df) <- NULL
  df
}

history_formula <- function(candidate) {
  stopifnot(candidate %in% names(history_candidate_specs))
  base <- m8trpg_formula("none")
  base_rhs <- sub("\\| 0$", "", as.character(base)[3])
  extra <- paste(
    history_candidate_specs[[candidate]], collapse = " + "
  )
  fml <- as.formula(paste(
    "chosen ~", base_rhs, "+", extra, "| 0"
  ))
  environment(fml) <- environment()
  fml
}

fit_predict_history <- function(train_long, valid_long, candidate) {
  scaler <- choice_scaler(train_long)
  tr_feat <- make_m8trpg_features(
    train_long, scaler$ctr, scaler$scl, "none"
  )
  va_feat <- make_m8trpg_features(
    valid_long, scaler$ctr, scaler$scl, "none"
  )
  fml <- history_formula(candidate)
  model <- mlogit(
    fml,
    data = tr_feat,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen"
  )

  mdat_va <- dfidx(
    va_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen"
  )
  class(mdat_va) <- c("dfidx_mlogit", class(mdat_va))
  mf_va <- model.frame(mdat_va, fml, balanced = TRUE)
  x_va <- model.matrix(mf_va, rhs = 1:3)
  beta <- coef(model)
  stopifnot(all(names(beta) %in% colnames(x_va)))
  eta <- as.numeric(
    x_va[, names(beta), drop = FALSE] %*% beta
  )

  va_map <- unique(va_feat[, c("chid", "No")])
  va_map <- va_map[order(va_map$No), , drop = FALSE]
  eta_chid <- as.character(dfidx::idx(mf_va, 1))
  eta_alt <- as.integer(as.character(dfidx::idx(mf_va, 2)))
  eta_order <- order(match(eta_chid, va_map$chid), eta_alt)
  stopifnot(
    identical(eta_chid[eta_order], rep(va_map$chid, each = 4)),
    identical(eta_alt[eta_order], rep(1:4, times = nrow(va_map)))
  )
  pred <- softmax_margins(eta[eta_order])
  rownames(pred) <- va_map$chid
  extra_coef <- beta[history_candidate_specs[[candidate]]]
  stopifnot(!anyNA(pred), !anyNA(extra_coef))
  list(
    pred = pred,
    no = va_map$No,
    extra_coef = extra_coef
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
  split_data <- readRDS("data_processed/train_val_split.rds")
  tr <- add_design_history_features(split_data$train_long_tr)
  va <- add_design_history_features(split_data$train_long_val)
  va_wide <- split_data$train_wide_val
  va_wide <- va_wide[order(va_wide$No), , drop = FALSE]
  truth <- as.matrix(va_wide[, paste0("Ch", 1:4)])

  baseline_fit <- fit_predict_m8trpg(tr, va)
  baseline <- baseline_fit$pred[
    match(va_wide$No, baseline_fit$no), , drop = FALSE
  ]
  baseline_loss <- log_loss_matrix(truth, baseline)
  stopifnot(abs(baseline_loss - 1.15968144721113) < 1e-8)

  feature_names <- unique(unlist(history_candidate_specs))
  summary_rows <- lapply(feature_names, function(feature) {
    inside_history <- tr$alt != 4L & tr$Task > 1L
    values <- tr[[feature]][inside_history]
    data.frame(
      feature = feature,
      minimum = min(values),
      mean = mean(values),
      sd = sd(values),
      maximum = max(values),
      zero_share = mean(values == 0)
    )
  })
  write_result_csv(
    do.call(rbind, summary_rows),
    file.path(output_dir, "design_history_feature_summary.csv")
  )

  result_rows <- list()
  coefficient_rows <- list()
  prediction_list <- list()
  for (candidate in names(history_candidate_specs)) {
    fitted <- fit_predict_history(tr, va, candidate)
    idx <- match(va_wide$No, fitted$no)
    pred <- fitted$pred[idx, , drop = FALSE]
    loss <- log_loss_matrix(truth, pred)
    result_rows[[candidate]] <- data.frame(
      candidate = candidate,
      n_terms = length(fitted$extra_coef),
      baseline_logloss = baseline_loss,
      candidate_logloss = loss,
      gain = baseline_loss - loss
    )
    coefficient_rows[[candidate]] <- data.frame(
      candidate = candidate,
      term = names(fitted$extra_coef),
      coefficient = unname(fitted$extra_coef)
    )
    prediction_list[[candidate]] <- pred
    cat(sprintf(
      "%s: val %.6f; gain %+.6f\n",
      candidate, loss, baseline_loss - loss
    ))
  }

  result <- do.call(rbind, result_rows)
  write_result_csv(
    result,
    file.path(output_dir, "design_history_screen.csv")
  )
  write_result_csv(
    do.call(rbind, coefficient_rows),
    file.path(output_dir, "design_history_screen_coefficients.csv")
  )
  saveRDS(
    list(
      result = result,
      predictions = prediction_list,
      baseline = baseline,
      validation_no = va_wide$No
    ),
    file.path(output_dir, "design_history_screen.rds")
  )
  print(result, digits = 7)
}

if (stage == "cv") {
  screen <- read.csv(
    file.path(output_dir, "design_history_screen.csv"),
    stringsAsFactors = FALSE
  )
  requested <- Sys.getenv("CODEX_CANDIDATES", "")
  if (nzchar(requested)) {
    candidates <- strsplit(requested, ",", fixed = TRUE)[[1]]
  } else {
    candidates <- screen$candidate[screen$gain > 0]
  }
  if (length(candidates) == 0L) {
    cat("No design-history candidate passed the screen; CV skipped.\n")
    quit(save = "no", status = 0)
  }
  stopifnot(all(candidates %in% names(history_candidate_specs)))

  train <- read.csv("csv files/train.csv")
  truth <- as.matrix(train[, paste0("Ch", 1:4)])
  train_long <- add_design_history_features(
    reshape_choice_long(train)
  )
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
      fitted <- fit_predict_history(tr, va, candidate)
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
        baseline_logloss = base_fold,
        candidate_logloss = candidate_fold,
        gain = base_fold - candidate_fold
      )
      coefficient_rows[[length(coefficient_rows) + 1L]] <-
        data.frame(
          candidate = candidate,
          fold = fold,
          term = names(fitted$extra_coef),
          coefficient = unname(fitted$extra_coef)
        )
      cat(sprintf(
        "%s fold %d: gain %+.6f\n",
        candidate, fold, base_fold - candidate_fold
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
    file.path(output_dir, "design_history_cv_folds.csv")
  )
  write_result_csv(
    do.call(rbind, coefficient_rows),
    file.path(output_dir, "design_history_cv_coefficients.csv")
  )
  write_result_csv(
    summary,
    file.path(output_dir, "design_history_cv.csv")
  )
  write_result_csv(
    bootstrap,
    file.path(output_dir, "design_history_bootstrap.csv")
  )
  saveRDS(
    list(
      candidates = candidates,
      oof = candidate_oof,
      summary = summary,
      bootstrap = bootstrap
    ),
    file.path(output_dir, "design_history_oof.rds")
  )
  print(summary, digits = 7)
  print(bootstrap, digits = 7)
}
