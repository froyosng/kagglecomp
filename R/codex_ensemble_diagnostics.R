source("R/codex_modeling_common.R")

dir.create("data_processed/codex", recursive = TRUE, showWarnings = FALSE)

train <- read.csv("csv files/train.csv")
truth <- as.matrix(train[, paste0("Ch", 1:4)])
fold_map <- canonical_fold_map()
row_fold <- unname(fold_map[as.character(train$Case)])

base <- readRDS("data_processed/oof_ensemble_v10.rds")
rank <- readRDS("data_processed/codex/rank_oof.rds")[[1]]$pred
tuned <- readRDS("data_processed/codex/xgb_retune_oof.rds")[[1]]$pred
cox <- readRDS("data_processed/codex/cox_oof.rds")$oof_min

components <- list(
  mlogit = base$oof_mlogit,
  original_xgb = base$oof_xgb,
  rank_ndcg = rank,
  retuned_xgb = tuned,
  cox_min = cox
)

softmax_weights <- function(theta, k) {
  z <- c(theta, 0)
  z <- z - max(z)
  exp(z) / sum(exp(z))
}

blend_components <- function(component_list, weights) {
  out <- component_list[[1]] * weights[1]
  if (length(component_list) > 1) {
    for (j in 2:length(component_list)) {
      out <- out + component_list[[j]] * weights[j]
    }
  }
  out
}

fit_weights <- function(component_list, actual, rows) {
  k <- length(component_list)
  objective <- function(theta) {
    w <- softmax_weights(theta, k)
    pred <- blend_components(lapply(component_list, function(p) p[rows, , drop = FALSE]), w)
    log_loss_matrix(actual[rows, , drop = FALSE], pred)
  }
  # The arithmetic-mixture loss is convex in simplex weights. A mlogit-heavy
  # deterministic start converges reliably and avoids repeated expensive scans.
  start <- c(log(4), rep(0, max(0, k - 2L)))
  best <- optim(start, objective, method = "BFGS",
                control = list(maxit = 250, reltol = 1e-10))
  list(weights = softmax_weights(best$par, k), logloss = best$value)
}

evaluate_pool <- function(pool_names) {
  pool <- components[pool_names]
  global_fit <- fit_weights(pool, truth, seq_len(nrow(train)))
  global_pred <- blend_components(pool, global_fit$weights)

  crossfit_pred <- matrix(NA_real_, nrow(train), 4)
  fold_weights <- matrix(NA_real_, 5, length(pool_names),
                         dimnames = list(paste0("fold", 1:5), pool_names))
  for (k in 1:5) {
    fit_rows <- which(row_fold != k)
    val_rows <- which(row_fold == k)
    meta_fit <- fit_weights(pool, truth, fit_rows)
    fold_weights[k, ] <- meta_fit$weights
    crossfit_pred[val_rows, ] <- blend_components(
      lapply(pool, function(p) p[val_rows, , drop = FALSE]),
      meta_fit$weights
    )
  }
  stopifnot(!anyNA(crossfit_pred))
  list(
    pool = paste(pool_names, collapse = "+"),
    global_weights = global_fit$weights,
    global_logloss = log_loss_matrix(truth, global_pred),
    crossfit_logloss = log_loss_matrix(truth, crossfit_pred),
    global_pred = global_pred,
    crossfit_pred = crossfit_pred,
    fold_weights = fold_weights
  )
}

pools <- list(
  c("mlogit", "original_xgb"),
  c("mlogit", "rank_ndcg"),
  c("mlogit", "retuned_xgb"),
  c("mlogit", "original_xgb", "cox_min"),
  c("mlogit", "rank_ndcg", "cox_min"),
  c("mlogit", "retuned_xgb", "cox_min"),
  c("mlogit", "rank_ndcg", "retuned_xgb"),
  c("mlogit", "rank_ndcg", "retuned_xgb", "cox_min"),
  c("mlogit", "original_xgb", "rank_ndcg", "retuned_xgb", "cox_min")
)

evaluated <- lapply(pools, evaluate_pool)
summary <- do.call(rbind, lapply(evaluated, function(x) {
  data.frame(
    pool = x$pool,
    global_logloss = x$global_logloss,
    crossfit_logloss = x$crossfit_logloss,
    weights = paste(sprintf("%.4f", x$global_weights), collapse = "/")
  )
}))
summary <- summary[order(summary$crossfit_logloss), ]

best_pool <- summary$pool[1]
best <- evaluated[[match(best_pool, vapply(evaluated, `[[`, character(1), "pool"))]]
v11_pred <- 0.8 * components$mlogit + 0.2 * components$original_xgb
retuned_two <- evaluated[[match(
  "mlogit+retuned_xgb",
  vapply(evaluated, `[[`, character(1), "pool")
)]]$crossfit_pred

set.seed(4821)
cases <- unique(train$Case)
B <- 1000L
boot <- matrix(NA_real_, B, 2,
               dimnames = list(NULL, c("gain_vs_v11", "gain_vs_retuned_two")))

row_loss <- function(pred) {
  pred <- pmin(pmax(pred / rowSums(pred), 1e-15), 1 - 1e-15)
  -rowSums(truth * log(pred))
}
candidate_row_loss <- row_loss(best$crossfit_pred)
v11_row_loss <- row_loss(v11_pred)
retuned_row_loss <- row_loss(retuned_two)
case_gain_v11 <- tapply(v11_row_loss - candidate_row_loss, train$Case, mean)
case_gain_retuned <- tapply(
  retuned_row_loss - candidate_row_loss, train$Case, mean
)

for (b in seq_len(B)) {
  sampled <- as.character(sample(cases, length(cases), replace = TRUE))
  boot[b, 1] <- mean(case_gain_v11[sampled])
  boot[b, 2] <- mean(case_gain_retuned[sampled])
}

bootstrap_summary <- data.frame(
  comparison = colnames(boot),
  mean_gain = colMeans(boot),
  sd = apply(boot, 2, sd),
  lower_95 = apply(boot, 2, quantile, probs = 0.025),
  upper_95 = apply(boot, 2, quantile, probs = 0.975),
  win_rate = colMeans(boot > 0)
)

fold_weight_rows <- do.call(rbind, lapply(evaluated, function(x) {
  do.call(rbind, lapply(seq_len(nrow(x$fold_weights)), function(k) {
    data.frame(
      pool = x$pool,
      fold = k,
      component = colnames(x$fold_weights),
      weight = as.numeric(x$fold_weights[k, ])
    )
  }))
}))

write_result_csv(summary, "data_processed/codex/ensemble_meta_cv.csv")
write_result_csv(bootstrap_summary,
                 "data_processed/codex/ensemble_bootstrap.csv")
write_result_csv(fold_weight_rows,
                 "data_processed/codex/ensemble_fold_weights.csv")
saveRDS(list(summary = summary, best = best,
             bootstrap = boot, bootstrap_summary = bootstrap_summary),
        "data_processed/codex/ensemble_diagnostics.rds")

cat("Meta-CV ensemble comparison:\n")
print(summary)
cat("\nBest pool:", best_pool, "\n")
cat("Fold-specific weights:\n")
print(best$fold_weights)
cat("\nRespondent bootstrap of fixed meta-crossfit predictions:\n")
print(bootstrap_summary)
