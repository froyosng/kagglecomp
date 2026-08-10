suppressPackageStartupMessages({
  library(glmnet)
  library(survival)
})
source("R/codex_modeling_common.R")

stage <- Sys.getenv("CODEX_STAGE", "screen")
dir.create("data_processed/codex", recursive = TRUE, showWarnings = FALSE)

cox_scaler_vars <- c(
  "incomea", "agea", "milesa", "nighta",
  "genderind", "Urbind", "educind"
)

make_cox_design <- function(df, ctr, scl, attr_max) {
  df <- sort_long_tasks(df)
  inside <- as.integer(as.integer(df$alt) != 4L)
  d2 <- as.integer(as.integer(df$alt) == 2L)
  d3 <- as.integer(as.integer(df$alt) == 3L)
  z <- sweep(sweep(as.matrix(df[, cox_scaler_vars]), 2, ctr, "-"), 2, scl, "/")

  core <- list()
  for (a in attrs) {
    for (lev in seq_len(attr_max[[a]])) {
      core[[paste0(a, "_", lev)]] <- as.numeric(df[[a]] == lev)
    }
  }
  core$Price <- as.numeric(df$Price)
  core$d2 <- d2
  core$d3 <- d3

  candidate <- list()
  for (v in cox_scaler_vars) {
    candidate[[paste0("P_", v)]] <- as.numeric(df$Price) * z[, v]
    candidate[[paste0("In_", v)]] <- inside * z[, v]
  }
  for (s in 2:6) {
    candidate[[paste0("P_seg", s)]] <-
      as.numeric(df$Price) * as.numeric(df$segmentind == s)
    candidate[[paste0("In_seg", s)]] <-
      inside * as.numeric(df$segmentind == s)
  }
  for (a in attrs) {
    for (s in 2:6) {
      candidate[[paste0(a, "_seg", s)]] <-
        as.numeric(df[[a]]) * as.numeric(df$segmentind == s)
    }
    for (v in c("incomea", "agea", "milesa", "nighta")) {
      candidate[[paste0(a, "_", v)]] <- as.numeric(df[[a]]) * z[, v]
    }
  }

  stopifnot(length(core) == 63L, length(candidate) == 195L)
  x <- do.call(cbind, c(core, candidate))
  storage.mode(x) <- "double"
  list(
    x = x,
    penalty = c(rep(0, length(core)), rep(1, length(candidate))),
    n_core = length(core)
  )
}

inner_fold_id <- function(df, seed) {
  cases <- unique(df$Case)
  set.seed(seed)
  f <- sample(rep(1:5, length.out = length(cases)))
  names(f) <- cases
  unname(f[as.character(df$Case)])
}

fit_cox_cv <- function(train_long, attr_max, seed) {
  train_long <- sort_long_tasks(train_long)
  ctr <- vapply(train_long[, cox_scaler_vars, drop = FALSE], mean, numeric(1))
  scl <- vapply(train_long[, cox_scaler_vars, drop = FALSE], sd, numeric(1))
  scl[scl == 0] <- 1
  design <- make_cox_design(train_long, ctr, scl, attr_max)
  y <- stratifySurv(
    Surv(rep(1, nrow(train_long)), as.integer(train_long$chosen)),
    as.factor(train_long$No)
  )
  foldid <- inner_fold_id(train_long, seed)
  cvfit <- cv.glmnet(
    x = design$x, y = y, family = "cox", alpha = 1,
    penalty.factor = design$penalty, standardize = FALSE,
    foldid = foldid, type.measure = "deviance",
    grouped = FALSE, nlambda = 60, cox.ties = "breslow"
  )
  list(cvfit = cvfit, ctr = ctr, scl = scl, attr_max = attr_max,
       n_core = design$n_core)
}

predict_cox_choice <- function(fit, new_long, s) {
  new_long <- sort_long_tasks(new_long)
  design <- make_cox_design(
    new_long, fit$ctr, fit$scl, fit$attr_max
  )
  eta <- as.numeric(predict(fit$cvfit$glmnet.fit, newx = design$x,
                            s = s, type = "link"))
  softmax_margins(eta, scale = 1)
}

selected_interactions <- function(fit, s) {
  b <- as.matrix(coef(fit$cvfit$glmnet.fit, s = s))
  sum(abs(b[(fit$n_core + 1):nrow(b), 1]) > 1e-10)
}

reshape_train_long <- function(train) {
  fixed_names <- names(train)[!grepl(
    paste0("^(", paste(c(attrs, "Price", "Ch"), collapse = "|"), ")[1-4]$"),
    names(train)
  )]
  parts <- lapply(1:4, function(a) {
    out <- train[, fixed_names, drop = FALSE]
    for (v in c(attrs, "Price", "Ch")) out[[v]] <- train[[paste0(v, a)]]
    out$alt <- a
    out$chosen <- as.integer(out$Ch == 1)
    out
  })
  sort_long_tasks(do.call(rbind, parts))
}

if (stage == "screen") {
  split <- readRDS("data_processed/train_val_split.rds")
  tr <- sort_long_tasks(split$train_long_tr)
  va <- sort_long_tasks(split$train_long_val)
  truth <- long_truth_matrix(va)
  attr_max <- as.list(vapply(rbind(tr, va)[, attrs], max, numeric(1)))

  fit <- fit_cox_cv(tr, attr_max, seed = 4821)
  pred_min <- predict_cox_choice(fit, va, fit$cvfit$lambda.min)
  pred_1se <- predict_cox_choice(fit, va, fit$cvfit$lambda.1se)
  result <- data.frame(
    lambda_rule = c("min", "1se"),
    lambda = c(fit$cvfit$lambda.min, fit$cvfit$lambda.1se),
    selected_interactions = c(
      selected_interactions(fit, fit$cvfit$lambda.min),
      selected_interactions(fit, fit$cvfit$lambda.1se)
    ),
    logloss = c(
      log_loss_matrix(truth, pred_min),
      log_loss_matrix(truth, pred_1se)
    )
  )
  write_result_csv(result, "data_processed/codex/cox_screen.csv")
  print(result)
}

if (stage == "cv") {
  train <- read.csv("csv files/train.csv")
  train_long <- reshape_train_long(train)
  truth <- as.matrix(train[, paste0("Ch", 1:4)])
  attr_max <- as.list(vapply(train_long[, attrs], max, numeric(1)))
  fold_map <- canonical_fold_map()
  saved <- readRDS("data_processed/oof_ensemble_v10.rds")

  oof_min <- matrix(NA_real_, nrow(train), 4)
  oof_1se <- matrix(NA_real_, nrow(train), 4)
  fold_details <- list()

  for (k in 1:5) {
    val_cases <- as.integer(names(fold_map)[fold_map == k])
    tr <- train_long[!(train_long$Case %in% val_cases), , drop = FALSE]
    va <- train_long[train_long$Case %in% val_cases, , drop = FALSE]
    fit <- fit_cox_cv(tr, attr_max, seed = 4821 + k)
    pred_min <- predict_cox_choice(fit, va, fit$cvfit$lambda.min)
    pred_1se <- predict_cox_choice(fit, va, fit$cvfit$lambda.1se)
    idx <- match(unique(va$No), train$No)
    oof_min[idx, ] <- pred_min
    oof_1se[idx, ] <- pred_1se
    fold_details[[k]] <- data.frame(
      fold = k,
      lambda_min = fit$cvfit$lambda.min,
      lambda_1se = fit$cvfit$lambda.1se,
      selected_min = selected_interactions(fit, fit$cvfit$lambda.min),
      selected_1se = selected_interactions(fit, fit$cvfit$lambda.1se)
    )
    cat(sprintf("fold %d done; min selected %d; 1se selected %d\n",
                k, fold_details[[k]]$selected_min,
                fold_details[[k]]$selected_1se))
    rm(fit)
    gc()
  }
  stopifnot(!anyNA(oof_min), !anyNA(oof_1se))

  rank_oof <- readRDS("data_processed/codex/rank_oof.rds")[[1]]$pred
  tuned_oof <- readRDS("data_processed/codex/xgb_retune_oof.rds")[[1]]$pred

  components <- list(
    cox_min = oof_min,
    cox_1se = oof_1se
  )
  results <- list()
  for (nm in names(components)) {
    pcox <- components[[nm]]
    three_original <- search_three_way_blend(
      saved$oof_mlogit, saved$oof_xgb, pcox, truth, step = 0.01
    )
    three_rank <- search_three_way_blend(
      saved$oof_mlogit, rank_oof, pcox, truth, step = 0.01
    )
    three_tuned <- search_three_way_blend(
      saved$oof_mlogit, tuned_oof, pcox, truth, step = 0.01
    )
    results[[nm]] <- rbind(
      cbind(cox_rule = nm, partner = "original_xgb", three_original),
      cbind(cox_rule = nm, partner = "rank_ndcg", three_rank),
      cbind(cox_rule = nm, partner = "retuned_xgb", three_tuned)
    )
  }

  # Also test whether the two new tree objectives complement each other.
  tree_three <- search_three_way_blend(
    saved$oof_mlogit, rank_oof, tuned_oof, truth, step = 0.01
  )
  results$tree_three <- cbind(
    cox_rule = "none", partner = "rank_plus_retuned", tree_three
  )

  summary <- do.call(rbind, results)
  component_scores <- data.frame(
    component = c("mlogit", "original_xgb", "rank_ndcg",
                  "retuned_xgb", "cox_min", "cox_1se"),
    logloss = c(
      log_loss_matrix(truth, saved$oof_mlogit),
      log_loss_matrix(truth, saved$oof_xgb),
      log_loss_matrix(truth, rank_oof),
      log_loss_matrix(truth, tuned_oof),
      log_loss_matrix(truth, oof_min),
      log_loss_matrix(truth, oof_1se)
    )
  )
  write_result_csv(do.call(rbind, fold_details),
                   "data_processed/codex/cox_cv_folds.csv")
  write_result_csv(component_scores,
                   "data_processed/codex/component_scores.csv")
  write_result_csv(summary,
                   "data_processed/codex/three_way_blends.csv")
  saveRDS(list(oof_min = oof_min, oof_1se = oof_1se),
          "data_processed/codex/cox_oof.rds")
  print(component_scores)
  print(summary)
}
