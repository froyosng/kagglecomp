source("R/log_loss.R")

attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
           "PP","KA","SC","TS","NV","MA","LB","AF","HU","Price")
design_cols <- as.vector(outer(attrs, 1:4, paste0))

oof <- readRDS("data_processed/oof_ensemble_v10.rds")
oof_mlogit <- oof$oof_mlogit
oof_xgb <- oof$oof_xgb
oof_truth <- oof$oof_truth
fold_of_case <- oof$fold_of_case

train <- read.csv("csv files/train.csv")
train$fp <- do.call(paste, c(train[, design_cols], sep = "_"))
train$Choice <- max.col(train[, c("Ch1","Ch2","Ch3","Ch4")])
fold_of_row <- fold_of_case[as.character(train$Case)]

w <- 0.80
oof_ensemble <- w * oof_mlogit + (1 - w) * oof_xgb
oof_ensemble <- oof_ensemble / rowSums(oof_ensemble)

# Design-cell empirical shrinkage: for each of the 5 folds, compute empirical
# choice shares per exact design fingerprint using ONLY the other 4 folds'
# respondents (proper cross-fitting -- a held-out respondent's own choice
# never contributes to their own prediction, mirroring exactly how this would
# work at real test time: test respondents are disjoint from train, but 98.5%
# of test designs recur in train per the fingerprint check). Blend:
# p_blend = (n_cell * p_empirical + alpha * p_model) / (n_cell + alpha)
blend_with_shrinkage <- function(p_model, alpha) {
  p_blend <- p_model
  for (k in 1:5) {
    train_idx <- which(fold_of_row != k)
    held_idx <- which(fold_of_row == k)

    cell_tab <- table(train$fp[train_idx], train$Choice[train_idx])
    n_cell <- rowSums(cell_tab)
    p_empirical <- cell_tab / n_cell  # row-normalized

    fp_held <- train$fp[held_idx]
    match_idx <- match(fp_held, rownames(cell_tab))
    has_match <- !is.na(match_idx)

    for (j in which(has_match)) {
      row <- held_idx[j]
      cell <- match_idx[j]
      n <- n_cell[cell]
      p_emp <- as.numeric(p_empirical[cell, ])
      p_blend[row, ] <- (n * p_emp + alpha * p_model[row, ]) / (n + alpha)
    }
  }
  p_blend / rowSums(p_blend)
}

cat("Baseline (no shrinkage): mlogit =", round(log_loss(oof_truth, oof_mlogit), 6),
    " ensemble =", round(log_loss(oof_truth, oof_ensemble), 6), "\n\n")

alphas <- c(2, 5, 10, 20, 40, 80, 160, Inf)
cat("=== Shrinkage on mlogit (m8trpg) alone ===\n")
for (a in alphas) {
  p <- if (is.infinite(a)) oof_mlogit else blend_with_shrinkage(oof_mlogit, a)
  cat("alpha =", a, " log loss =", round(log_loss(oof_truth, p), 6), "\n")
}

cat("\n=== Shrinkage on the full ensemble (mlogit+xgb) ===\n")
for (a in alphas) {
  p <- if (is.infinite(a)) oof_ensemble else blend_with_shrinkage(oof_ensemble, a)
  cat("alpha =", a, " log loss =", round(log_loss(oof_truth, p), 6), "\n")
}

cat("\nCoverage: fraction of held-out tasks whose design recurs elsewhere in that fold's training portion:\n")
cov <- numeric(5)
for (k in 1:5) {
  train_idx <- which(fold_of_row != k)
  held_idx <- which(fold_of_row == k)
  cov[k] <- mean(train$fp[held_idx] %in% train$fp[train_idx])
}
print(round(cov, 4))
