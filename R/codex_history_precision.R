# High-precision uncertainty audit for the design-exposure history candidates.
#
# This is deliberately separate from codex_design_history.R so that the costly
# mlogit fits do not need to be repeated. It uses the saved canonical-fold OOF
# predictions, repeats the respondent-clustered bootstrap at high precision,
# and reports both ordinary and multiplicity-adjusted intervals.

options(stringsAsFactors = FALSE)

output_dir <- "data_processed/codex_behavioral_round"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

log_loss_matrix <- function(truth, pred) {
  pred <- pmin(pmax(pred, 1e-15), 1 - 1e-15)
  -mean(rowSums(truth * log(pred)))
}

row_log_loss <- function(truth, pred) {
  pred <- pmin(pmax(pred, 1e-15), 1 - 1e-15)
  -rowSums(truth * log(pred))
}

case_mean_gain <- function(truth, baseline, candidate, respondent) {
  row_gain <- row_log_loss(truth, baseline) -
    row_log_loss(truth, candidate)
  unname(tapply(row_gain, respondent, mean))
}

bootstrap_case_means <- function(gain_by_case, n_boot = 100000L,
                                 seed = 4821L, chunk_size = 2000L) {
  set.seed(seed)
  n_case <- length(gain_by_case)
  result <- numeric(n_boot)
  starts <- seq.int(1L, n_boot, by = chunk_size)
  for (start in starts) {
    stop_at <- min(n_boot, start + chunk_size - 1L)
    n_this <- stop_at - start + 1L
    sampled <- matrix(
      sample(gain_by_case, n_case * n_this, replace = TRUE),
      nrow = n_case,
      ncol = n_this
    )
    result[start:stop_at] <- colMeans(sampled)
  }
  result
}

summarize_bootstrap <- function(gain_by_case, point_gain,
                                family_size = 8L,
                                n_boot = 100000L,
                                seed = 4821L) {
  boot <- bootstrap_case_means(
    gain_by_case,
    n_boot = n_boot,
    seed = seed
  )
  alpha_family <- 0.05 / family_size
  normal_half_width <- qnorm(0.975) * sd(boot)
  data.frame(
    point_gain = point_gain,
    bootstrap_mean = mean(boot),
    bootstrap_sd = sd(boot),
    lower_95 = unname(quantile(boot, 0.025)),
    upper_95 = unname(quantile(boot, 0.975)),
    lower_99 = unname(quantile(boot, 0.005)),
    upper_99 = unname(quantile(boot, 0.995)),
    lower_bonferroni_95 = unname(
      quantile(boot, alpha_family / 2)
    ),
    upper_bonferroni_95 = unname(
      quantile(boot, 1 - alpha_family / 2)
    ),
    normal_lower_95 = point_gain - normal_half_width,
    normal_upper_95 = point_gain + normal_half_width,
    win_rate = mean(boot > 0),
    n_boot = n_boot,
    family_size = family_size
  )
}

train <- read.csv("csv files/train.csv")
truth <- as.matrix(train[, paste0("Ch", 1:4)])
saved_base <- readRDS("data_processed/oof_ensemble_v10.rds")
saved_history <- readRDS(
  file.path(output_dir, "design_history_oof.rds")
)

baseline <- saved_base$oof_mlogit
xgb <- saved_base$oof_xgb
v11 <- 0.8 * baseline + 0.2 * xgb
stopifnot(
  abs(log_loss_matrix(truth, baseline) - 1.1470212110518) < 1e-8,
  abs(log_loss_matrix(truth, v11) - 1.145094212987) < 1e-8,
  identical(nrow(truth), nrow(baseline)),
  !anyNA(saved_history$oof)
)

candidates <- saved_history$candidates
family_size <- length(candidates)
fixed_points <- data.frame(
  candidate = candidates,
  point_gain = vapply(
    candidates,
    function(candidate) {
      pred <- 0.8 * saved_history$oof[[candidate]] + 0.2 * xgb
      log_loss_matrix(truth, v11) - log_loss_matrix(truth, pred)
    },
    numeric(1)
  )
)

# The best fixed-weight candidate is also checked with an honest outer-fold
# shrinkage choice. For each held-out fold, q is selected using only the other
# four OOF folds. q=0 recovers v11 and q=1 uses the full history candidate.
best_candidate <- fixed_points$candidate[
  which.max(fixed_points$point_gain)
]
best_oof <- saved_history$oof[[best_candidate]]
best_fixed_pred <- 0.8 * best_oof + 0.2 * xgb
best_fixed_gain <- log_loss_matrix(truth, v11) -
  log_loss_matrix(truth, best_fixed_pred)
best_fixed_summary <- cbind(
  data.frame(
    candidate = best_candidate,
    comparison = "fixed_080_blend"
  ),
  summarize_bootstrap(
    case_mean_gain(truth, v11, best_fixed_pred, train$Case),
    point_gain = best_fixed_gain,
    family_size = family_size,
    seed = 4821L
  )
)
fold_map <- saved_base$fold_of_case
fold_of_row <- unname(fold_map[as.character(train$Case)])
stopifnot(!anyNA(fold_of_row), all(fold_of_row %in% 1:5))

q_grid <- seq(0, 1, by = 0.05)
crossfit_pred <- matrix(NA_real_, nrow(train), 4)
weight_rows <- vector("list", 5L)

for (fold in 1:5) {
  tune_rows <- fold_of_row != fold
  validation_rows <- fold_of_row == fold
  tune_losses <- vapply(
    q_grid,
    function(q) {
      shrunk_mlogit <- (1 - q) * baseline + q * best_oof
      blend <- 0.8 * shrunk_mlogit + 0.2 * xgb
      log_loss_matrix(truth[tune_rows, ], blend[tune_rows, ])
    },
    numeric(1)
  )
  best_q <- q_grid[which.min(tune_losses)]
  shrunk_mlogit <- (1 - best_q) * baseline + best_q * best_oof
  fold_pred <- 0.8 * shrunk_mlogit + 0.2 * xgb
  crossfit_pred[validation_rows, ] <- fold_pred[validation_rows, ]
  weight_rows[[fold]] <- data.frame(
    fold = fold,
    selected_q = best_q,
    tuning_logloss = min(tune_losses)
  )
}
stopifnot(!anyNA(crossfit_pred))

crossfit_point_gain <- log_loss_matrix(truth, v11) -
  log_loss_matrix(truth, crossfit_pred)
crossfit_summary <- cbind(
  data.frame(
    candidate = best_candidate,
    comparison = "fold_crossfit_shrink"
  ),
  summarize_bootstrap(
    case_mean_gain(truth, v11, crossfit_pred, train$Case),
    point_gain = crossfit_point_gain,
    family_size = family_size,
    seed = 14821L
  )
)

precision_summary <- rbind(best_fixed_summary, crossfit_summary)
write.csv(
  fixed_points,
  file.path(output_dir, "design_history_fixed_points.csv"),
  row.names = FALSE
)
write.csv(
  precision_summary,
  file.path(output_dir, "design_history_precision.csv"),
  row.names = FALSE
)
write.csv(
  do.call(rbind, weight_rows),
  file.path(output_dir, "design_history_crossfit_weights.csv"),
  row.names = FALSE
)
saveRDS(
  list(
    best_candidate = best_candidate,
    crossfit_prediction = crossfit_pred,
    crossfit_weights = do.call(rbind, weight_rows),
    precision_summary = precision_summary
  ),
  file.path(output_dir, "design_history_precision.rds")
)

print(precision_summary, digits = 9)
print(do.call(rbind, weight_rows), digits = 9)
