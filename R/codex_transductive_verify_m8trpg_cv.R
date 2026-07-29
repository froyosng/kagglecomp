# Correctness check (infrastructure, not a hypothesis result): reproduce the
# official m8trpg 5-fold OOF log loss (1.1470212110518, from
# data_processed/oof_ensemble_v10.rds) using this track's OWN reimplementation
# (R/codex_transductive_common.R: to_long / make_features_m8trpg / m8trpg_formula,
# with recode_fn = NULL so it is byte-for-byte the same specification as
# R/cv_ensemble_v10.R). If this doesn't match closely, nothing built on top of this
# pipeline (quantile-matching, self-training) can be trusted.

source("R/codex_transductive_common.R")
library(mlogit)

t0 <- Sys.time()
train <- read.csv("csv files/train.csv")
truth <- as.matrix(train[, paste0("Ch", 1:4)])
train_long <- to_long(train)

saved_base <- readRDS("data_processed/oof_ensemble_v10.rds")
fold_of_case <- saved_base$fold_of_case
fold_of_row <- unname(fold_of_case[as.character(train$Case)])
stopifnot(!anyNA(fold_of_row), all(table(fold_of_case) == 227))

scaler_vars <- c("incomea","agea","milesa","nighta","genderind","Urbind","educind")
oof_mlogit_check <- matrix(NA_real_, nrow(train), 4)

for (k in 1:5) {
  val_cases_k <- names(fold_of_case)[fold_of_case == k]
  tr_k <- train_long[!(as.character(train_long$Case) %in% val_cases_k), ]
  va_k <- train_long[as.character(train_long$Case) %in% val_cases_k, ]

  std_ctr <- sapply(tr_k[scaler_vars], mean, na.rm = TRUE)
  std_scl <- sapply(tr_k[scaler_vars], sd, na.rm = TRUE)
  std_scl[std_scl == 0] <- 1

  tr_feat <- make_features_m8trpg(tr_k, std_ctr, std_scl, recode_fn = NULL)
  va_feat <- make_features_m8trpg(va_k, std_ctr, std_scl, recode_fn = NULL)

  mdat_tr <- dfidx(tr_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
  mdat_va <- dfidx(va_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
  mod_k <- mlogit(m8trpg_formula(), data = mdat_tr)
  pred_k <- predict(mod_k, newdata = mdat_va)

  va_map <- unique(va_k[, c("chid", "No")])
  row_idx <- match(va_map$No, train$No)
  pred_ordered <- pred_k[match(va_map$chid, rownames(pred_k)), ]
  oof_mlogit_check[row_idx, ] <- pred_ordered

  cat("Fold", k, "n_train_resp =", length(unique(tr_k$Case)),
      " n_val_resp =", length(unique(va_k$Case)),
      " fold ll =", round(log_loss_matrix(truth[row_idx, ], pred_ordered), 6),
      " elapsed =", round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s\n")
}

stopifnot(!anyNA(oof_mlogit_check))
ll_check <- log_loss_matrix(truth, oof_mlogit_check)
ll_official <- log_loss_matrix(truth, saved_base$oof_mlogit)
cat("\nReimplemented pooled OOF m8trpg log loss:", sprintf("%.10f", ll_check), "\n")
cat("Officially saved pooled OOF m8trpg log loss:", sprintf("%.10f", ll_official), "\n")
cat("Absolute difference:", sprintf("%.2e", abs(ll_check - ll_official)), "\n")
cat("Correlation between reimplemented and saved OOF (col-wise, alt1):",
    cor(oof_mlogit_check[, 1], saved_base$oof_mlogit[, 1]), "\n")
cat("Max abs cell difference:", max(abs(oof_mlogit_check - saved_base$oof_mlogit)), "\n")

saveRDS(list(oof_mlogit_check = oof_mlogit_check, fold_of_row = fold_of_row),
        "data_processed/codex_transductive/m8trpg_reimplementation_check.rds")

cat("\nTotal elapsed:", round(as.numeric(Sys.time() - t0, units = "mins"), 2), "min\n")
