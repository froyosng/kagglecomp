# High-precision audit of the selected MLP ensemble using saved canonical-fold
# OOF predictions. This reconstructs the fold-cross-fitted blend exactly,
# verifies the headline losses, and quantifies sensitivity to the nine-config
# screen that selected the architecture.

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

bootstrap_case_means <- function(gain_by_case, n_boot = 100000L,
                                 seed = 4821L,
                                 chunk_size = 2000L) {
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

train <- read.csv("csv files/train.csv")
truth <- as.matrix(train[, paste0("Ch", 1:4)])
saved_base <- readRDS("data_processed/oof_ensemble_v10.rds")
saved_mlp <- readRDS(file.path(output_dir, "mlp_oof.rds"))
weights <- read.csv(
  file.path(output_dir, "mlp_cv_weights.csv"),
  stringsAsFactors = FALSE
)

stopifnot(
  length(saved_mlp$candidates) == 1L,
  nrow(weights) == 5L,
  length(unique(weights$candidate)) == 1L
)
candidate <- saved_mlp$candidates[[1]]
mlp <- saved_mlp$oof[[candidate]]
baseline <- saved_base$oof_mlogit
xgb <- saved_base$oof_xgb
v11 <- 0.8 * baseline + 0.2 * xgb
fold_of_row <- unname(
  saved_base$fold_of_case[as.character(train$Case)]
)

stopifnot(
  !anyNA(mlp),
  !anyNA(fold_of_row),
  max(abs(rowSums(mlp) - 1)) < 1e-10,
  all(mlp > 0),
  abs(log_loss_matrix(truth, baseline) - 1.1470212110518) < 1e-8,
  abs(log_loss_matrix(truth, v11) - 1.14509421298673) < 1e-10,
  abs(log_loss_matrix(truth, mlp) - 1.19054334979533) < 1e-10
)

crossfit <- matrix(NA_real_, nrow(train), 4L)
fold_rows <- vector("list", 5L)
for (fold in 1:5) {
  rows <- fold_of_row == fold
  weight <- weights$mlp_weight[weights$fold == fold]
  stopifnot(length(weight) == 1L)
  crossfit[rows, ] <- (1 - weight) * v11[rows, ] +
    weight * mlp[rows, ]
  fold_rows[[fold]] <- data.frame(
    fold = fold,
    mlp_weight = weight,
    v11_logloss = log_loss_matrix(truth[rows, ], v11[rows, ]),
    mlp_logloss = log_loss_matrix(truth[rows, ], mlp[rows, ]),
    blend_logloss =
      log_loss_matrix(truth[rows, ], crossfit[rows, ]),
    blend_gain =
      log_loss_matrix(truth[rows, ], v11[rows, ]) -
      log_loss_matrix(truth[rows, ], crossfit[rows, ])
  )
}
stopifnot(
  !anyNA(crossfit),
  abs(log_loss_matrix(truth, crossfit) -
    1.14378944178118) < 1e-10
)

case_gain <- unname(tapply(
  row_log_loss(truth, v11) -
    row_log_loss(truth, crossfit),
  train$Case,
  mean
))
point_gain <- mean(case_gain)
boot <- bootstrap_case_means(case_gain)
family_size <- 9L
alpha_family <- 0.05 / family_size
normal_half_width <- qnorm(0.975) * sd(boot)

precision <- data.frame(
  candidate = candidate,
  v11_logloss = log_loss_matrix(truth, v11),
  crossfit_blend_logloss = log_loss_matrix(truth, crossfit),
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
  n_boot = length(boot),
  family_size = family_size
)

write.csv(
  precision,
  file.path(output_dir, "mlp_precision.csv"),
  row.names = FALSE
)
write.csv(
  do.call(rbind, fold_rows),
  file.path(output_dir, "mlp_precision_folds.csv"),
  row.names = FALSE
)
saveRDS(
  list(
    candidate = candidate,
    crossfit_prediction = crossfit,
    precision = precision,
    fold_results = do.call(rbind, fold_rows)
  ),
  file.path(output_dir, "mlp_precision.rds")
)

print(precision, digits = 9)
print(do.call(rbind, fold_rows), digits = 9)
