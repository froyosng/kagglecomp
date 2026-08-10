source("R/codex_shift_common.R")

stage <- Sys.getenv("CODEX_STAGE", "screen")
dir.create("data_processed/codex_shift", recursive = TRUE,
           showWarnings = FALSE)

train <- read.csv("csv files/train.csv")
test <- read.csv("csv files/test.csv")
truth <- as.matrix(train[, paste0("Ch", 1:4)])

if (stage == "screen") {
  split <- readRDS("data_processed/train_val_split.rds")
  tr_wide <- split$train_wide_tr
  va_wide <- split$train_wide_val
  tr_long <- split$train_long_tr
  va_long <- split$train_long_val

  # The saved long split predates chid/dummy construction.
  for (x in c("tr_long", "va_long")) {
    d <- get(x)
    d$chid <- paste(d$Case, d$Task, sep = "_")
    d$d2 <- as.integer(d$alt == 2L)
    d$d3 <- as.integer(d$alt == 3L)
    assign(x, d)
  }

  ratio_fit <- fit_density_ratio(tr_wide, test)
  ratio_tr <- predict_density_ratio(ratio_fit, tr_wide)
  ratio_va <- predict_density_ratio(ratio_fit, va_wide)
  eval_weights <- stabilize_ratio(ratio_va, cap = 20, power = 1)
  va_order <- order(va_wide$No)
  va_wide <- va_wide[va_order, , drop = FALSE]
  eval_row_weight <- eval_weights[as.character(va_wide$Case)]
  va_truth <- as.matrix(va_wide[, paste0("Ch", 1:4)])

  alpha_env <- Sys.getenv("CODEX_ALPHAS", "")
  alphas <- if (nzchar(alpha_env)) {
    as.numeric(strsplit(alpha_env, ",", fixed = TRUE)[[1]])
  } else {
    c(0, 0.25, 0.5, 0.75, 1)
  }
  results <- list()
  predictions <- list()
  for (i in seq_along(alphas)) {
    alpha <- alphas[i]
    fit_weights <- stabilize_ratio(ratio_tr, cap = 20, power = alpha)
    fitted <- fit_predict_m8trpg(
      tr_long, va_long, case_weights = fit_weights
    )
    idx <- match(va_wide$No, fitted$no)
    pred <- fitted$pred[idx, , drop = FALSE]
    predictions[[i]] <- pred
    results[[i]] <- data.frame(
      alpha = alpha,
      fit_weight_min = min(fit_weights),
      fit_weight_max = max(fit_weights),
      fit_effective_n = effective_sample_size(fit_weights),
      unweighted_logloss = log_loss_matrix(va_truth, pred),
      target_weighted_logloss = weighted_log_loss(
        va_truth, pred, eval_row_weight
      ),
      target_weighted_cap5 = weighted_log_loss(
        va_truth, pred,
        stabilize_ratio(ratio_va, cap = 5)[as.character(va_wide$Case)]
      ),
      target_weighted_cap10 = weighted_log_loss(
        va_truth, pred,
        stabilize_ratio(ratio_va, cap = 10)[as.character(va_wide$Case)]
      )
    )
    cat(sprintf(
      "alpha %.2f: ordinary %.6f; target-weighted %.6f; effective n %.1f\n",
      alpha, results[[i]]$unweighted_logloss,
      results[[i]]$target_weighted_logloss,
      results[[i]]$fit_effective_n
    ))
  }
  result <- do.call(rbind, results)
  write_result_csv(result,
                   "data_processed/codex_shift/importance_screen.csv")
  saveRDS(
    list(result = result, predictions = predictions,
         va_no = va_wide$No, eval_weights = eval_row_weight),
    "data_processed/codex_shift/importance_screen.rds"
  )
  print(result)
}

if (stage == "cv") {
  screen <- read.csv(
    "data_processed/codex_shift/importance_screen.csv"
  )
  alphas <- sort(unique(screen$alpha))
  cat("Confirming alphas:", paste(alphas, collapse = ", "), "\n")

  train_long <- reshape_choice_long(train)
  fold_map <- canonical_fold_map()
  saved <- readRDS("data_processed/oof_ensemble_v10.rds")
  baseline <- saved$oof_mlogit
  xgb <- saved$oof_xgb

  weighted_oof <- lapply(
    alphas,
    function(x) matrix(NA_real_, nrow(train), 4)
  )
  names(weighted_oof) <- as.character(alphas)
  eval_weights <- rep(NA_real_, nrow(train))
  fold_summary <- list()

  for (k in 1:5) {
    val_cases <- as.integer(names(fold_map)[fold_map == k])
    tr_wide <- train[!(train$Case %in% val_cases), , drop = FALSE]
    va_wide <- train[train$Case %in% val_cases, , drop = FALSE]
    tr_long <- train_long[!(train_long$Case %in% val_cases), , drop = FALSE]
    va_long <- train_long[train_long$Case %in% val_cases, , drop = FALSE]

    ratio_fit <- fit_density_ratio(tr_wide, test)
    ratio_tr <- predict_density_ratio(ratio_fit, tr_wide)
    ratio_va <- predict_density_ratio(ratio_fit, va_wide)
    eval_case_weights <- stabilize_ratio(
      ratio_va, cap = 20, power = 1
    )
    row_idx <- which(train$Case %in% val_cases)
    eval_weights[row_idx] <-
      eval_case_weights[as.character(train$Case[row_idx])]

    for (alpha in alphas) {
      fit_weights <- stabilize_ratio(
        ratio_tr, cap = 20, power = alpha
      )
      fitted <- fit_predict_m8trpg(
        tr_long, va_long, case_weights = fit_weights
      )
      fitted_idx <- match(fitted$no, train$No)
      weighted_oof[[as.character(alpha)]][fitted_idx, ] <- fitted$pred

      fold_summary[[length(fold_summary) + 1L]] <- data.frame(
        fold = k,
        alpha = alpha,
        fit_effective_n = effective_sample_size(fit_weights),
        eval_effective_n = effective_sample_size(eval_case_weights),
        baseline_target_loss = weighted_log_loss(
          truth[row_idx, ], baseline[row_idx, ], eval_weights[row_idx]
        ),
        refit_target_loss = weighted_log_loss(
          truth[row_idx, ],
          weighted_oof[[as.character(alpha)]][row_idx, ],
          eval_weights[row_idx]
        )
      )
      cat(sprintf(
        paste0(
          "fold %d alpha %.2f: baseline target %.6f; ",
          "refit target %.6f\n"
        ),
        k, alpha,
        fold_summary[[length(fold_summary)]]$baseline_target_loss,
        fold_summary[[length(fold_summary)]]$refit_target_loss
      ))
    }
  }
  stopifnot(
    all(vapply(weighted_oof, function(x) !anyNA(x), logical(1))),
    !anyNA(eval_weights)
  )

  weighted_blend_loss <- function(w, mlogit_pred, weights) {
    weighted_log_loss(
      truth, w * mlogit_pred + (1 - w) * xgb, weights
    )
  }
  blend_weights <- seq(0, 1, by = 0.01)
  baseline_curve <- vapply(
    blend_weights, weighted_blend_loss, numeric(1),
    mlogit_pred = baseline, weights = eval_weights
  )
  baseline_w <- blend_weights[which.min(baseline_curve)]
  baseline_blend <- baseline_w * baseline + (1 - baseline_w) * xgb

  summary_rows <- list()
  candidate_blends <- list()
  for (alpha in alphas) {
    pred <- weighted_oof[[as.character(alpha)]]
    curve <- vapply(
      blend_weights, weighted_blend_loss, numeric(1),
      mlogit_pred = pred, weights = eval_weights
    )
    blend_w <- blend_weights[which.min(curve)]
    candidate_blends[[as.character(alpha)]] <-
      blend_w * pred + (1 - blend_w) * xgb
    summary_rows[[length(summary_rows) + 1L]] <- data.frame(
      alpha = alpha,
      fit_ordinary_logloss = log_loss_matrix(truth, pred),
      fit_target_weighted_logloss = weighted_log_loss(
        truth, pred, eval_weights
      ),
      delta_target_vs_baseline = weighted_log_loss(
        truth, pred, eval_weights
      ) - weighted_log_loss(truth, baseline, eval_weights),
      blend_mlogit_weight = blend_w,
      blend_ordinary_logloss = log_loss_matrix(
        truth, candidate_blends[[as.character(alpha)]]
      ),
      blend_target_weighted_logloss = min(curve),
      delta_blend_target_vs_baseline = min(curve) - min(baseline_curve)
    )
  }
  summary <- do.call(rbind, summary_rows)

  # Paired respondent bootstraps use each alpha's target-optimal global blend
  # only as a diagnostic. A CI crossing zero cannot justify a submission.
  row_loss <- function(pred) {
    pred <- pmin(pmax(pred / rowSums(pred), 1e-15), 1 - 1e-15)
    -rowSums(truth * log(pred))
  }
  set.seed(4821)
  B <- 1000L
  boot_rows <- list()
  for (alpha in alphas) {
    for (comparison in c("mlogit", "blend")) {
      if (comparison == "mlogit") {
        base_pred <- baseline
        candidate_pred <- weighted_oof[[as.character(alpha)]]
      } else {
        base_pred <- baseline_blend
        candidate_pred <- candidate_blends[[as.character(alpha)]]
      }
      base_loss <- row_loss(base_pred)
      candidate_loss <- row_loss(candidate_pred)
      resp <- aggregate(
        cbind(
          weight = eval_weights,
          weighted_delta =
            eval_weights * (base_loss - candidate_loss)
        ),
        by = list(Case = train$Case),
        FUN = mean
      )
      boot <- numeric(B)
      for (b in seq_len(B)) {
        sampled <- sample(
          seq_len(nrow(resp)), nrow(resp), replace = TRUE
        )
        boot[b] <- sum(resp$weighted_delta[sampled]) /
          sum(resp$weight[sampled])
      }
      boot_rows[[length(boot_rows) + 1L]] <- data.frame(
        alpha = alpha,
        comparison = comparison,
        point_gain = weighted_log_loss(
          truth, base_pred, eval_weights
        ) - weighted_log_loss(
          truth, candidate_pred, eval_weights
        ),
        bootstrap_mean = mean(boot),
        bootstrap_sd = sd(boot),
        lower_95 = quantile(boot, 0.025),
        upper_95 = quantile(boot, 0.975),
        win_rate = mean(boot > 0)
      )
    }
  }
  bootstrap <- do.call(rbind, boot_rows)

  write_result_csv(
    do.call(rbind, fold_summary),
    "data_processed/codex_shift/importance_cv_folds.csv"
  )
  write_result_csv(
    summary, "data_processed/codex_shift/importance_cv.csv"
  )
  write_result_csv(
    bootstrap, "data_processed/codex_shift/importance_bootstrap.csv"
  )
  saveRDS(
    list(
      weighted_oof = weighted_oof,
      eval_weights = eval_weights,
      summary = summary,
      bootstrap = bootstrap,
      baseline_blend_weight = baseline_w
    ),
    "data_processed/codex_shift/importance_oof.rds"
  )
  print(summary)
  print(bootstrap)
}
