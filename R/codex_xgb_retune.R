suppressPackageStartupMessages(library(xgboost))
source("R/codex_modeling_common.R")

stage <- Sys.getenv("CODEX_STAGE", "screen")
dir.create("data_processed/codex", recursive = TRUE, showWarnings = FALSE)

xgb_params <- function(row) {
  list(
    objective = "multi:softprob",
    num_class = 4,
    eval_metric = "mlogloss",
    eta = as.numeric(row$eta),
    max_depth = as.integer(row$max_depth),
    min_child_weight = as.numeric(row$min_child_weight),
    subsample = as.numeric(row$subsample),
    colsample_bytree = as.numeric(row$colsample_bytree),
    reg_alpha = as.numeric(row$reg_alpha),
    reg_lambda = as.numeric(row$reg_lambda),
    gamma = as.numeric(row$gamma),
    tree_method = "hist",
    seed = 4821
  )
}

make_screen_grid <- function() {
  baseline <- data.frame(
    eta = 0.10, max_depth = 4L, min_child_weight = 1,
    subsample = 0.8, colsample_bytree = 0.8,
    reg_alpha = 0, reg_lambda = 1, gamma = 0,
    nrounds = 73L
  )

  candidates <- expand.grid(
    eta = c(0.03, 0.05, 0.08, 0.10),
    max_depth = 2:6,
    min_child_weight = c(1, 3, 8, 15),
    subsample = c(0.65, 0.8, 1.0),
    colsample_bytree = c(0.65, 0.8, 1.0),
    reg_alpha = c(0, 0.25, 1, 3),
    reg_lambda = c(0.5, 1, 5, 15),
    gamma = c(0, 0.1, 0.5)
  )
  set.seed(7402)
  # Twenty-three seeded space-filling draws plus the exact baseline keep this
  # a cheap screen while covering every requested hyperparameter.
  candidates <- candidates[sample(seq_len(nrow(candidates)), 23), ]
  multiplier <- sample(c(0.75, 1, 1.5), nrow(candidates), replace = TRUE)
  candidates$nrounds <- as.integer(round((7.3 / candidates$eta) * multiplier))
  rbind(baseline, candidates)
}

fit_multiclass <- function(train_wide, spec) {
  y <- max.col(train_wide[, paste0("Ch", 1:4)]) - 1L
  dtrain <- xgb.DMatrix(wide_feature_matrix(train_wide), label = y)
  xgb.train(
    params = xgb_params(spec), data = dtrain,
    nrounds = as.integer(spec$nrounds), verbose = 0
  )
}

predict_multiclass <- function(model, wide_df) {
  pred <- predict(model, xgb.DMatrix(wide_feature_matrix(wide_df)))
  as.matrix(pred)
}

if (stage == "screen") {
  split <- readRDS("data_processed/train_val_split.rds")
  tr <- split$train_wide_tr
  va <- split$train_wide_val
  truth <- as.matrix(va[, paste0("Ch", 1:4)])
  grid <- make_screen_grid()
  grid$logloss <- NA_real_

  for (i in seq_len(nrow(grid))) {
    model <- fit_multiclass(tr, grid[i, ])
    pred <- predict_multiclass(model, va)
    grid$logloss[i] <- log_loss_matrix(truth, pred)
    cat(sprintf(
      "[%02d/%02d] eta=%.2f depth=%d child=%g sub=%.2f col=%.2f a=%g l=%g gamma=%g rounds=%d ll=%.6f\n",
      i, nrow(grid), grid$eta[i], grid$max_depth[i],
      grid$min_child_weight[i], grid$subsample[i],
      grid$colsample_bytree[i], grid$reg_alpha[i],
      grid$reg_lambda[i], grid$gamma[i], grid$nrounds[i],
      grid$logloss[i]
    ))
  }
  grid <- grid[order(grid$logloss), ]
  write_result_csv(grid, "data_processed/codex/xgb_retune_screen.csv")
  print(head(grid, 10))
}

if (stage == "cv") {
  train <- read.csv("csv files/train.csv")
  truth <- as.matrix(train[, paste0("Ch", 1:4)])
  fold_map <- canonical_fold_map()
  saved <- readRDS("data_processed/oof_ensemble_v10.rds")
  oof_mlogit <- saved$oof_mlogit

  screen <- read.csv("data_processed/codex/xgb_retune_screen.csv")
  top <- screen[1:min(3, nrow(screen)), , drop = FALSE]
  summary_rows <- list()
  all_oof <- list()

  for (m in seq_len(nrow(top))) {
    spec <- top[m, ]
    oof <- matrix(NA_real_, nrow(train), 4)
    for (k in 1:5) {
      val_cases <- as.integer(names(fold_map)[fold_map == k])
      tr <- train[!(train$Case %in% val_cases), , drop = FALSE]
      va <- train[train$Case %in% val_cases, , drop = FALSE]
      model <- fit_multiclass(tr, spec)
      pred <- predict_multiclass(model, va)
      idx <- match(va$No, train$No)
      oof[idx, ] <- pred
      cat(sprintf("config %d fold %d done\n", m, k))
    }
    stopifnot(!anyNA(oof))
    ll <- log_loss_matrix(truth, oof)
    blend <- search_two_way_blend(oof_mlogit, oof, truth)
    summary_rows[[m]] <- cbind(
      spec,
      cv_logloss = ll,
      best_mlogit_weight = blend$weight1,
      blend_logloss = blend$logloss
    )
    all_oof[[m]] <- list(spec = spec, pred = oof)
    cat(sprintf(
      "config %d pooled xgb ll %.6f; blend %.6f at mlogit %.2f\n",
      m, ll, blend$logloss, blend$weight1
    ))
  }

  summary <- do.call(rbind, summary_rows)
  write_result_csv(summary, "data_processed/codex/xgb_retune_cv.csv")
  saveRDS(all_oof, "data_processed/codex/xgb_retune_oof.rds")
  print(summary)
}
