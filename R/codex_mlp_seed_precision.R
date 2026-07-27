# High-precision paired audit of 20-seed MLP averaging versus the original
# five-seed candidate. The same respondent bootstrap draws are used for every
# comparison so changes in interval width are directly comparable.

options(stringsAsFactors = FALSE)

output_dir <- "data_processed/codex_mlp_seed_bagging"
n_boot <- 100000L
family_size <- 9L

log_loss_matrix <- function(truth, prediction) {
  prediction <- pmin(pmax(prediction, 1e-15), 1 - 1e-15)
  -mean(rowSums(truth * log(prediction)))
}

row_log_loss <- function(truth, prediction) {
  prediction <- pmin(pmax(prediction, 1e-15), 1 - 1e-15)
  -rowSums(truth * log(prediction))
}

train <- read.csv("csv files/train.csv")
truth <- as.matrix(train[, paste0("Ch", 1:4)])
saved_base <- readRDS("data_processed/oof_ensemble_v10.rds")
saved_seed <- readRDS(
  file.path(output_dir, "mlp_seed_bagging_oof.rds")
)
baseline <- saved_base$oof_mlogit
xgb <- saved_base$oof_xgb
v11 <- 0.8 * baseline + 0.2 * xgb

mlp_5 <- saved_seed$oof_by_count[[5L]]
mlp_20 <- saved_seed$oof_by_count[[20L]]
blend_5 <- saved_seed$blend_by_count[[5L]]$prediction
blend_20 <- saved_seed$blend_by_count[[20L]]$prediction

stopifnot(
  abs(log_loss_matrix(truth, v11) - 1.14509421298673) < 1e-10,
  abs(log_loss_matrix(truth, mlp_5) - 1.19054334979533) < 1e-10,
  abs(log_loss_matrix(truth, blend_5) - 1.14378944178118) < 1e-10,
  !anyNA(mlp_20),
  !anyNA(blend_20),
  max(abs(rowSums(mlp_20) - 1)) < 1e-10,
  # v11 contains single-precision xgboost probabilities, so its blends
  # inherit harmless row-sum error around 1e-8.
  max(abs(rowSums(blend_20) - 1)) < 1e-6
)

case_gain <- function(reference, candidate) {
  unname(tapply(
    row_log_loss(truth, reference) -
      row_log_loss(truth, candidate),
    train$Case,
    mean
  ))
}

gain_matrix <- cbind(
  component_20_vs_5 = case_gain(mlp_5, mlp_20),
  blend_5_vs_v11 = case_gain(v11, blend_5),
  blend_20_vs_v11 = case_gain(v11, blend_20),
  blend_20_vs_5 = case_gain(blend_5, blend_20)
)
stopifnot(nrow(gain_matrix) == length(unique(train$Case)))

set.seed(4821L)
n_case <- nrow(gain_matrix)
comparison_count <- ncol(gain_matrix)
boot <- matrix(
  NA_real_,
  nrow = n_boot,
  ncol = comparison_count,
  dimnames = list(NULL, colnames(gain_matrix))
)
chunk_size <- 1000L
for (start in seq.int(1L, n_boot, by = chunk_size)) {
  stop_at <- min(n_boot, start + chunk_size - 1L)
  n_this <- stop_at - start + 1L
  sampled_index <- matrix(
    sample.int(
      n_case,
      n_case * n_this,
      replace = TRUE
    ),
    nrow = n_case,
    ncol = n_this
  )
  for (comparison in seq_len(comparison_count)) {
    sampled_gain <- matrix(
      gain_matrix[sampled_index, comparison],
      nrow = n_case,
      ncol = n_this
    )
    boot[start:stop_at, comparison] <-
      colMeans(sampled_gain)
  }
}

alpha_family <- 0.05 / family_size
precision <- do.call(rbind, lapply(
  seq_len(comparison_count),
  function(index) {
    values <- boot[, index]
    point <- mean(gain_matrix[, index])
    normal_half_width <- qnorm(0.975) * sd(values)
    data.frame(
      comparison = colnames(gain_matrix)[[index]],
      point_gain = point,
      bootstrap_mean = mean(values),
      bootstrap_sd = sd(values),
      lower_95 = unname(quantile(values, 0.025)),
      upper_95 = unname(quantile(values, 0.975)),
      width_95 = unname(
        quantile(values, 0.975) -
          quantile(values, 0.025)
      ),
      lower_99 = unname(quantile(values, 0.005)),
      upper_99 = unname(quantile(values, 0.995)),
      lower_bonferroni_9_95 = unname(
        quantile(values, alpha_family / 2)
      ),
      upper_bonferroni_9_95 = unname(
        quantile(values, 1 - alpha_family / 2)
      ),
      normal_lower_95 = point - normal_half_width,
      normal_upper_95 = point + normal_half_width,
      win_rate = mean(values > 0),
      n_boot = n_boot
    )
  }
))

fold_of_row <- unname(
  saved_base$fold_of_case[as.character(train$Case)]
)
fold_rows <- do.call(rbind, lapply(c(5L, 20L), function(seed_count) {
  component <- saved_seed$oof_by_count[[seed_count]]
  blend <- saved_seed$blend_by_count[[seed_count]]$prediction
  weights <- saved_seed$blend_by_count[[seed_count]]$weights
  do.call(rbind, lapply(1:5, function(fold) {
    rows <- fold_of_row == fold
    data.frame(
      n_seeds = seed_count,
      fold = fold,
      mlp_weight = weights[[fold]],
      component_logloss =
        log_loss_matrix(truth[rows, ], component[rows, ]),
      blend_logloss =
        log_loss_matrix(truth[rows, ], blend[rows, ]),
      v11_logloss =
        log_loss_matrix(truth[rows, ], v11[rows, ]),
      blend_gain =
        log_loss_matrix(truth[rows, ], v11[rows, ]) -
        log_loss_matrix(truth[rows, ], blend[rows, ])
    )
  }))
}))

write.csv(
  precision,
  file.path(output_dir, "mlp_seed_precision.csv"),
  row.names = FALSE
)
write.csv(
  fold_rows,
  file.path(output_dir, "mlp_seed_precision_folds.csv"),
  row.names = FALSE
)
saveRDS(
  list(
    precision = precision,
    fold_results = fold_rows,
    bootstrap = boot,
    gain_by_case = gain_matrix,
    blend_5 = blend_5,
    blend_20 = blend_20
  ),
  file.path(output_dir, "mlp_seed_precision.rds")
)

print(precision, digits = 9)
print(fold_rows, digits = 9)
