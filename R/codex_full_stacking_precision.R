# High-precision respondent bootstrap for the completed full-stacking round.
# Includes arithmetic sanity-check blends because the augmented eight-component
# arithmetic pool unexpectedly produced the round's best point estimate.

options(stringsAsFactors = FALSE)

source("R/codex_modeling_common.R")

output_dir <- "data_processed/codex_deep_stack"
saved <- readRDS(file.path(output_dir, "full_stacking.rds"))
train <- read.csv("csv files/train.csv")
truth <- as.matrix(train[, paste0("Ch", 1:4)])
current <- saved$current
results <- saved$results

old_five <- readRDS(
  "data_processed/codex_behavioral_round/mlp_extended_ensemble.rds"
)$best$crossfit_pred
deep_joint <- readRDS(
  file.path(output_dir, "torch_deep_oof.rds")
)$predictions$joint

stopifnot(
  abs(log_loss_matrix(truth, current) -
    1.14378944178118) < 1e-8,
  abs(log_loss_matrix(truth, old_five) -
    1.14312870734996) < 1e-8,
  abs(log_loss_matrix(truth, deep_joint) -
    1.143030195718) < 2e-8,
  abs(log_loss_matrix(
    truth, results$augmented8_arithmetic$prediction
  ) - 1.14211190984165) < 1e-10
)

row_loss <- function(prediction) {
  -rowSums(
    truth * log(pmax(prediction / rowSums(prediction), 1e-15))
  )
}

candidate_names <- names(results)
comparison_rows <- list()
for (name in candidate_names) {
  comparison_rows[[paste0(name, "_vs_current")]] <-
    row_loss(current) - row_loss(results[[name]]$prediction)
}
comparison_rows$augmented8_arithmetic_vs_old_five <-
  row_loss(old_five) -
  row_loss(results$augmented8_arithmetic$prediction)
comparison_rows$augmented8_arithmetic_vs_deep_joint <-
  row_loss(deep_joint) -
  row_loss(results$augmented8_arithmetic$prediction)

case_gain <- do.call(cbind, lapply(comparison_rows, function(value) {
  unname(tapply(value, train$Case, mean))
}))

set.seed(4821L)
n_boot <- 100000L
n_case <- nrow(case_gain)
boot <- matrix(
  NA_real_, n_boot, ncol(case_gain),
  dimnames = list(NULL, colnames(case_gain))
)
for (start in seq.int(1L, n_boot, by = 1000L)) {
  stop_at <- min(n_boot, start + 999L)
  n_this <- stop_at - start + 1L
  sampled <- matrix(
    sample.int(n_case, n_case * n_this, replace = TRUE),
    nrow = n_case
  )
  for (index in seq_len(ncol(case_gain))) {
    boot[start:stop_at, index] <- colMeans(matrix(
      case_gain[sampled, index],
      nrow = n_case
    ))
  }
}

# Six model variants were evaluated in the full stacking round:
# two pools x arithmetic/log-pool/meta-xgboost.
family_size <- length(candidate_names)
family_alpha <- 0.05 / family_size
summary <- do.call(rbind, lapply(seq_len(ncol(boot)), function(index) {
  value <- boot[, index]
  data.frame(
    comparison = colnames(boot)[[index]],
    point_gain = mean(case_gain[, index]),
    bootstrap_sd = sd(value),
    lower_95 = unname(quantile(value, 0.025)),
    upper_95 = unname(quantile(value, 0.975)),
    lower_99 = unname(quantile(value, 0.005)),
    upper_99 = unname(quantile(value, 0.995)),
    lower_bonferroni = unname(
      quantile(value, family_alpha / 2)
    ),
    upper_bonferroni = unname(
      quantile(value, 1 - family_alpha / 2)
    ),
    win_rate = mean(value > 0),
    n_boot = n_boot,
    family_size = family_size
  )
}))

write.csv(
  summary,
  file.path(output_dir, "full_stacking_precision.csv"),
  row.names = FALSE
)
saveRDS(
  list(
    summary = summary,
    bootstrap = boot,
    case_gain = case_gain,
    family_size = family_size
  ),
  file.path(output_dir, "full_stacking_precision.rds")
)
print(summary, digits = 9)
