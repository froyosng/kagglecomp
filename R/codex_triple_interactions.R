source("R/codex_triple_common.R")

stage <- Sys.getenv("CODEX_STAGE", "screen")
output_dir <- "data_processed/codex_triples"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

train <- read.csv("csv files/train.csv")
truth <- as.matrix(train[, paste0("Ch", 1:4)])

segment_support <- function(wide, sample_name) {
  respondents <- wide[!duplicated(wide$Case), , drop = FALSE]
  counts <- as.data.frame(table(respondents$segmentind))
  names(counts) <- c("segment", "respondents")
  counts$sample <- sample_name
  counts[, c("sample", "segment", "respondents")]
}

if (stage == "screen") {
  split <- readRDS("data_processed/train_val_split.rds")
  tr_long <- prepare_saved_long(split$train_long_tr)
  va_long <- prepare_saved_long(split$train_long_val)
  va_wide <- split$train_wide_val
  va_wide <- va_wide[order(va_wide$No), , drop = FALSE]
  va_truth <- as.matrix(va_wide[, paste0("Ch", 1:4)])

  support <- rbind(
    segment_support(train, "full_train"),
    segment_support(split$train_wide_tr, "screen_train"),
    segment_support(split$train_wide_val, "screen_validation")
  )
  write_result_csv(
    support, file.path(output_dir, "segment_support.csv")
  )
  print(support)

  baseline_fit <- fit_predict_m8trpg(
    tr_long, va_long,
    case_weights = NULL,
    attr_rank_scope = "none"
  )
  baseline_pred <- baseline_fit$pred[
    match(va_wide$No, baseline_fit$no), , drop = FALSE
  ]
  baseline_loss <- log_loss_matrix(va_truth, baseline_pred)
  stopifnot(abs(baseline_loss - 1.15968144721113) < 1e-8)
  cat(sprintf("baseline m8trpg: %.6f\n", baseline_loss))

  requested <- Sys.getenv("CODEX_CANDIDATES", "")
  candidates <- if (nzchar(requested)) {
    strsplit(requested, ",", fixed = TRUE)[[1]]
  } else {
    names(triple_candidate_specs)
  }
  stopifnot(all(candidates %in% names(triple_candidate_specs)))

  results <- list()
  predictions <- list()
  coefficients <- list()
  for (candidate in candidates) {
    fitted <- tryCatch(
      fit_predict_candidate(tr_long, va_long, candidate),
      error = function(e) e
    )
    spec <- triple_candidate_specs[[candidate]]
    if (inherits(fitted, "error")) {
      results[[candidate]] <- data.frame(
        candidate = candidate,
        family = spec$family,
        status = "fit_failed",
        n_terms = length(candidate_extra_terms(candidate)),
        logloss = NA_real_,
        delta_vs_base = NA_real_,
        max_abs_extra_coef = NA_real_,
        detail = conditionMessage(fitted)
      )
      cat(candidate, "failed:", conditionMessage(fitted), "\n")
    } else {
      idx <- match(va_wide$No, fitted$no)
      pred <- fitted$pred[idx, , drop = FALSE]
      loss <- log_loss_matrix(va_truth, pred)
      results[[candidate]] <- data.frame(
        candidate = candidate,
        family = spec$family,
        status = "ok",
        n_terms = length(fitted$extra_coef),
        logloss = loss,
        delta_vs_base = loss - baseline_loss,
        max_abs_extra_coef = max(abs(fitted$extra_coef)),
        detail = "continuous standardized products; no binning"
      )
      predictions[[candidate]] <- pred
      coefficients[[candidate]] <- data.frame(
        candidate = candidate,
        term = names(fitted$extra_coef),
        coefficient = unname(fitted$extra_coef)
      )
      cat(sprintf(
        "%s: %.6f; delta %+.6f; max |extra coef| %.4f\n",
        candidate, loss, loss - baseline_loss,
        max(abs(fitted$extra_coef))
      ))
    }
  }

  result <- do.call(rbind, results)
  coef_result <- do.call(rbind, coefficients)
  write_result_csv(
    result, file.path(output_dir, "triple_screen.csv")
  )
  if (!is.null(coef_result)) {
    write_result_csv(
      coef_result, file.path(output_dir, "triple_screen_coefficients.csv")
    )
  }
  saveRDS(
    list(
      result = result,
      predictions = predictions,
      baseline_pred = baseline_pred,
      validation_no = va_wide$No
    ),
    file.path(output_dir, "triple_screen.rds")
  )
  print(result)
}

if (stage == "cv") {
  screen <- read.csv(file.path(output_dir, "triple_screen.csv"))
  requested <- Sys.getenv("CODEX_CANDIDATES", "")
  if (nzchar(requested)) {
    candidates <- strsplit(requested, ",", fixed = TRUE)[[1]]
  } else {
    viable <- screen[
      screen$status == "ok" & screen$delta_vs_base < 0,
      , drop = FALSE
    ]
    if (nrow(viable) == 0) {
      cat("No screen candidate beat m8trpg; CV skipped.\n")
      quit(save = "no", status = 0)
    }
    candidates <- viable$candidate[order(viable$logloss)]
  }
  stopifnot(all(candidates %in% names(triple_candidate_specs)))
  cat("CV candidates:", paste(candidates, collapse = ", "), "\n")

  train_long <- reshape_choice_long(train)
  fold_map <- canonical_fold_map()
  saved <- readRDS("data_processed/oof_ensemble_v10.rds")
  baseline <- saved$oof_mlogit
  xgb <- saved$oof_xgb
  baseline_loss <- log_loss_matrix(truth, baseline)
  stopifnot(abs(baseline_loss - 1.1470212110518) < 1e-8)

  candidate_oof <- lapply(
    candidates,
    function(x) matrix(NA_real_, nrow(train), 4)
  )
  names(candidate_oof) <- candidates
  fold_results <- list()
  coefficient_results <- list()

  for (candidate in candidates) {
    for (fold in 1:5) {
      val_cases <- as.integer(names(fold_map)[fold_map == fold])
      tr_long <- train_long[
        !(train_long$Case %in% val_cases), , drop = FALSE
      ]
      va_long <- train_long[
        train_long$Case %in% val_cases, , drop = FALSE
      ]
      fitted <- fit_predict_candidate(tr_long, va_long, candidate)
      idx <- match(fitted$no, train$No)
      candidate_oof[[candidate]][idx, ] <- fitted$pred

      fold_results[[length(fold_results) + 1L]] <- data.frame(
        candidate = candidate,
        fold = fold,
        baseline_logloss = log_loss_matrix(
          truth[idx, ], baseline[idx, ]
        ),
        candidate_logloss = log_loss_matrix(
          truth[idx, ], candidate_oof[[candidate]][idx, ]
        ),
        gain = log_loss_matrix(truth[idx, ], baseline[idx, ]) -
          log_loss_matrix(
            truth[idx, ], candidate_oof[[candidate]][idx, ]
          ),
        max_abs_extra_coef = max(abs(fitted$extra_coef))
      )
      coefficient_results[[length(coefficient_results) + 1L]] <-
        data.frame(
          candidate = candidate,
          fold = fold,
          term = names(fitted$extra_coef),
          coefficient = unname(fitted$extra_coef)
        )
      cat(sprintf(
        "%s fold %d: base %.6f; candidate %.6f; gain %+.6f\n",
        candidate, fold,
        fold_results[[length(fold_results)]]$baseline_logloss,
        fold_results[[length(fold_results)]]$candidate_logloss,
        fold_results[[length(fold_results)]]$gain
      ))
    }
    stopifnot(!anyNA(candidate_oof[[candidate]]))
  }

  v11 <- 0.8 * baseline + 0.2 * xgb
  v11_loss <- log_loss_matrix(truth, v11)
  stopifnot(abs(v11_loss - 1.145094) < 5e-6)
  blend_grid <- seq(0, 1, by = 0.01)
  summary_rows <- list()
  bootstrap_rows <- list()

  for (candidate in candidates) {
    pred <- candidate_oof[[candidate]]
    fixed_blend <- 0.8 * pred + 0.2 * xgb
    grid_loss <- vapply(
      blend_grid,
      function(weight) {
        log_loss_matrix(
          truth, weight * pred + (1 - weight) * xgb
        )
      },
      numeric(1)
    )
    best_weight <- blend_grid[which.min(grid_loss)]
    best_blend <- best_weight * pred + (1 - best_weight) * xgb

    summary_rows[[length(summary_rows) + 1L]] <- data.frame(
      candidate = candidate,
      family = triple_candidate_specs[[candidate]]$family,
      n_terms = length(candidate_extra_terms(candidate)),
      baseline_mlogit_logloss = baseline_loss,
      candidate_mlogit_logloss = log_loss_matrix(truth, pred),
      mlogit_gain = baseline_loss - log_loss_matrix(truth, pred),
      v11_logloss = v11_loss,
      fixed_080_blend_logloss = log_loss_matrix(truth, fixed_blend),
      fixed_080_blend_gain =
        v11_loss - log_loss_matrix(truth, fixed_blend),
      diagnostic_best_weight = best_weight,
      diagnostic_best_blend_logloss = min(grid_loss),
      diagnostic_best_blend_gain = v11_loss - min(grid_loss)
    )

    comparisons <- list(
      mlogit = list(base = baseline, candidate = pred),
      fixed_080_blend = list(base = v11, candidate = fixed_blend),
      diagnostic_optimized_blend = list(
        base = v11, candidate = best_blend
      )
    )
    for (comparison in names(comparisons)) {
      boot <- respondent_bootstrap_gain(
        truth,
        comparisons[[comparison]]$base,
        comparisons[[comparison]]$candidate,
        train$Case
      )
      bootstrap_rows[[length(bootstrap_rows) + 1L]] <- cbind(
        data.frame(
          candidate = candidate,
          comparison = comparison
        ),
        boot
      )
    }
  }

  summary <- do.call(rbind, summary_rows)
  bootstrap <- do.call(rbind, bootstrap_rows)
  fold_result <- do.call(rbind, fold_results)
  coefficient_result <- do.call(rbind, coefficient_results)

  write_result_csv(
    fold_result, file.path(output_dir, "triple_cv_folds.csv")
  )
  write_result_csv(
    coefficient_result,
    file.path(output_dir, "triple_cv_coefficients.csv")
  )
  write_result_csv(
    summary, file.path(output_dir, "triple_cv.csv")
  )
  write_result_csv(
    bootstrap, file.path(output_dir, "triple_bootstrap.csv")
  )
  saveRDS(
    list(
      candidates = candidates,
      oof = candidate_oof,
      summary = summary,
      bootstrap = bootstrap,
      fold_results = fold_result
    ),
    file.path(output_dir, "triple_oof.rds")
  )
  print(summary)
  print(bootstrap)
}
