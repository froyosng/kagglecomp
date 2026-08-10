suppressPackageStartupMessages(library(xgboost))
source("R/codex_modeling_common.R")

stage <- Sys.getenv("CODEX_STAGE", "screen")
dir.create("data_processed/codex", recursive = TRUE, showWarnings = FALSE)

rank_params <- function(objective, eta, depth, min_child = 1,
                        subsample = 0.8, colsample = 0.8) {
  list(
    objective = objective,
    eval_metric = "ndcg@1",
    eta = eta,
    max_depth = depth,
    min_child_weight = min_child,
    subsample = subsample,
    colsample_bytree = colsample,
    tree_method = "hist",
    seed = 4821
  )
}

fit_ranker <- function(train_long, params, nrounds) {
  train_long <- sort_long_tasks(train_long)
  qid <- as.integer(factor(train_long$No, levels = unique(train_long$No)))
  dtrain <- xgb.DMatrix(
    rank_feature_matrix(train_long),
    label = as.numeric(train_long$chosen),
    qid = qid
  )
  xgb.train(params = params, data = dtrain, nrounds = nrounds, verbose = 0)
}

predict_ranker_margin <- function(model, long_df) {
  long_df <- sort_long_tasks(long_df)
  qid <- as.integer(factor(long_df$No, levels = unique(long_df$No)))
  dvalid <- xgb.DMatrix(rank_feature_matrix(long_df), qid = qid)
  as.numeric(predict(model, dvalid, outputmargin = TRUE))
}

if (stage == "screen") {
  split <- readRDS("data_processed/train_val_split.rds")
  tr <- sort_long_tasks(split$train_long_tr)
  va <- sort_long_tasks(split$train_long_val)
  truth <- long_truth_matrix(va)

  grid <- expand.grid(
    objective = c("rank:pairwise", "rank:ndcg"),
    eta = c(0.05, 0.10),
    depth = c(3L, 4L, 6L),
    nrounds = c(75L, 150L, 300L),
    stringsAsFactors = FALSE
  )
  grid$logloss <- NA_real_
  grid$scale <- NA_real_

  for (i in seq_len(nrow(grid))) {
    g <- grid[i, ]
    params <- rank_params(g$objective, g$eta, g$depth)
    model <- fit_ranker(tr, params, g$nrounds)
    margin <- predict_ranker_margin(model, va)
    calibrated <- best_margin_scale(margin, truth)
    grid$logloss[i] <- calibrated$logloss
    grid$scale[i] <- calibrated$scale
    cat(sprintf(
      "[%02d/%02d] %s eta=%.2f depth=%d rounds=%d scale=%.3f ll=%.6f\n",
      i, nrow(grid), g$objective, g$eta, g$depth, g$nrounds,
      calibrated$scale, calibrated$logloss
    ))
  }

  grid <- grid[order(grid$logloss), ]
  write_result_csv(grid, "data_processed/codex/rank_screen.csv")
  print(head(grid, 10))
}

if (stage == "cv") {
  train <- read.csv("csv files/train.csv")
  pat <- paste0("^(", paste(c(attrs, "Price", "Ch"), collapse = "|"), ")([1-4])$")
  varying <- grep(pat, names(train), value = TRUE)

  # Base reshape avoids a tidyverse dependency in this long-running script.
  long_parts <- lapply(1:4, function(a) {
    fixed <- train[, setdiff(names(train), varying), drop = FALSE]
    for (v in c(attrs, "Price", "Ch")) fixed[[v]] <- train[[paste0(v, a)]]
    fixed$alt <- a
    fixed$chosen <- as.integer(fixed$Ch == 1)
    fixed
  })
  train_long <- do.call(rbind, long_parts)
  train_long <- sort_long_tasks(train_long)

  screen <- read.csv("data_processed/codex/rank_screen.csv")
  # Confirm the two best materially distinct configurations, not near-duplicates.
  top <- screen[!duplicated(screen$objective), , drop = FALSE]
  if (nrow(top) < 2) top <- screen[1:min(2, nrow(screen)), , drop = FALSE]
  top <- top[1:min(2, nrow(top)), , drop = FALSE]

  fold_map <- canonical_fold_map()
  truth <- as.matrix(train[, paste0("Ch", 1:4)])
  saved <- readRDS("data_processed/oof_ensemble_v10.rds")
  oof_mlogit <- saved$oof_mlogit

  summary_rows <- list()
  all_oof <- list()
  for (m in seq_len(nrow(top))) {
    spec <- top[m, ]
    oof_margin <- rep(NA_real_, nrow(train) * 4L)

    for (k in 1:5) {
      val_cases <- as.integer(names(fold_map)[fold_map == k])
      tr <- train_long[!(train_long$Case %in% val_cases), , drop = FALSE]
      va <- train_long[train_long$Case %in% val_cases, , drop = FALSE]
      params <- rank_params(spec$objective, spec$eta, spec$depth)
      model <- fit_ranker(tr, params, spec$nrounds)
      margin <- predict_ranker_margin(model, va)
      row_index <- match(unique(va$No), train$No)
      long_index <- as.vector(rbind(
        4L * row_index - 3L, 4L * row_index - 2L,
        4L * row_index - 1L, 4L * row_index
      ))
      oof_margin[long_index] <- margin
      cat(sprintf("config %d fold %d done\n", m, k))
    }

    stopifnot(!anyNA(oof_margin))
    calibrated <- best_margin_scale(oof_margin, truth)

    # Cross-fit the probability-temperature scale: choose it on four folds'
    # margins and apply it to the fifth. This avoids using a held-out label to
    # calibrate its own ranking score.
    margin_matrix <- matrix(oof_margin, ncol = 4, byrow = TRUE)
    pred <- matrix(NA_real_, nrow(train), 4)
    fold_scales <- numeric(5)
    for (k in 1:5) {
      fit_rows <- which(unname(fold_map[as.character(train$Case)]) != k)
      val_rows <- which(unname(fold_map[as.character(train$Case)]) == k)
      fit_margin <- as.vector(t(margin_matrix[fit_rows, , drop = FALSE]))
      fold_cal <- best_margin_scale(
        fit_margin, truth[fit_rows, , drop = FALSE]
      )
      fold_scales[k] <- fold_cal$scale
      val_margin <- as.vector(t(margin_matrix[val_rows, , drop = FALSE]))
      pred[val_rows, ] <- softmax_margins(val_margin, fold_cal$scale)
    }
    stopifnot(!anyNA(pred))
    crossfit_ll <- log_loss_matrix(truth, pred)
    blend <- search_two_way_blend(oof_mlogit, pred, truth)
    summary_rows[[m]] <- data.frame(
      objective = spec$objective, eta = spec$eta, depth = spec$depth,
      nrounds = spec$nrounds, global_scale = calibrated$scale,
      fold_scales = paste(sprintf("%.3f", fold_scales), collapse = "/"),
      rank_logloss = crossfit_ll,
      best_mlogit_weight = blend$weight1,
      blend_logloss = blend$logloss
    )
    all_oof[[m]] <- list(spec = spec, margin = oof_margin, pred = pred,
                         global_scale = calibrated$scale,
                         fold_scales = fold_scales)
    cat(sprintf(
      "config %d pooled rank ll %.6f; blend %.6f at mlogit %.2f\n",
      m, crossfit_ll, blend$logloss, blend$weight1
    ))
  }
  summary <- do.call(rbind, summary_rows)
  write_result_csv(summary, "data_processed/codex/rank_cv.csv")
  saveRDS(all_oof, "data_processed/codex/rank_oof.rds")
  print(summary)
}
