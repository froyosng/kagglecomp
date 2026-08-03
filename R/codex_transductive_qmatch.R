# Candidate A: quantile-mapped moment matching on m8trpg's four continuous
# Price/inside interaction covariates (income, age, mileage, night), per
# codex_transductive_preregister.md Section 3. Produces a 5-fold OOF mlogit
# prediction matrix comparable to the officially-saved oof_mlogit, differing ONLY
# in how income/age/mileage/night are recoded before standardizing.

source("R/codex_transductive_common.R")
library(mlogit)

t0 <- Sys.time()
train <- read.csv("csv files/train.csv")
test <- read.csv("csv files/test.csv")
truth <- as.matrix(train[, paste0("Ch", 1:4)])
train_long <- to_long(train)

saved_base <- readRDS("data_processed/oof_ensemble_v10.rds")
fold_of_case <- saved_base$fold_of_case
fold_of_row <- unname(fold_of_case[as.character(train$Case)])
stopifnot(!anyNA(fold_of_row), all(table(fold_of_case) == 227))

qmatch_vars <- c("incomea", "agea", "milesa", "nighta")
scaler_vars <- c("incomea","agea","milesa","nighta","genderind","Urbind","educind")

# Fixed target distribution: test.csv's respondent-level covariate values, same
# reference regardless of CV fold (test.csv has no labels to leak; its covariates
# are legitimately observable at prediction time for every fold).
test_resp <- test[!duplicated(test$Case), c("Case", qmatch_vars)]
stopifnot(nrow(test_resp) == 263L)
target_values_list <- setNames(lapply(qmatch_vars, function(v) test_resp[[v]]), qmatch_vars)
test_mean <- sapply(qmatch_vars, function(v) mean(test_resp[[v]]))
test_sd <- sapply(qmatch_vars, function(v) sd(test_resp[[v]]))

oof_mlogit_qmatch <- matrix(NA_real_, nrow(train), 4)
fold_diagnostics <- vector("list", 5L)

for (k in 1:5) {
  val_cases_k <- names(fold_of_case)[fold_of_case == k]
  tr_k <- train_long[!(as.character(train_long$Case) %in% val_cases_k), ]
  va_k <- train_long[as.character(train_long$Case) %in% val_cases_k, ]

  # Source distribution for this fold: respondent-level covariate values among
  # THIS FOLD'S ~908 training respondents only (never the held-out 227).
  tr_resp <- tr_k[!duplicated(tr_k$Case), c("Case", qmatch_vars)]
  stopifnot(nrow(tr_resp) == 908L)
  source_values_list <- setNames(lapply(qmatch_vars, function(v) tr_resp[[v]]), qmatch_vars)

  recode_fn <- function(varname, x) {
    quantile_match(x, source_values_list[[varname]], target_values_list[[varname]])
  }

  # Standardization: the four quantile-matched covariates are already on test's
  # raw scale (by construction), so standardize them with TEST's fixed mean/sd;
  # the three untouched covariates (gender/urb/educ) keep the ordinary fold-
  # specific training mean/sd, exactly as in the current-best model.
  std_ctr <- sapply(scaler_vars, function(v) {
    if (v %in% qmatch_vars) test_mean[[v]] else mean(tr_k[[v]], na.rm = TRUE)
  })
  std_scl <- sapply(scaler_vars, function(v) {
    if (v %in% qmatch_vars) test_sd[[v]] else sd(tr_k[[v]], na.rm = TRUE)
  })
  std_scl[std_scl == 0] <- 1
  names(std_ctr) <- scaler_vars
  names(std_scl) <- scaler_vars

  tr_feat <- make_features_m8trpg(tr_k, std_ctr, std_scl, recode_fn = recode_fn)
  va_feat <- make_features_m8trpg(va_k, std_ctr, std_scl, recode_fn = recode_fn)

  mdat_tr <- dfidx(tr_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
  mdat_va <- dfidx(va_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
  mod_k <- mlogit(m8trpg_formula(), data = mdat_tr)
  pred_k <- predict(mod_k, newdata = mdat_va)

  va_map <- unique(va_k[, c("chid", "No")])
  row_idx <- match(va_map$No, train$No)
  pred_ordered <- pred_k[match(va_map$chid, rownames(pred_k)), ]
  oof_mlogit_qmatch[row_idx, ] <- pred_ordered

  fold_ll <- log_loss_matrix(truth[row_idx, ], pred_ordered)
  fold_diagnostics[[k]] <- data.frame(
    fold = k,
    income_shift_example = as.numeric(quantile_match(median(tr_resp$incomea),
                                                       source_values_list$incomea,
                                                       target_values_list$incomea)) -
      median(tr_resp$incomea),
    fold_logloss = fold_ll
  )
  cat("Fold", k, "qmatch fold ll =", round(fold_ll, 6),
      " elapsed =", round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s\n")
}

stopifnot(!anyNA(oof_mlogit_qmatch), all(oof_mlogit_qmatch > 0),
          max(abs(rowSums(oof_mlogit_qmatch) - 1)) < 1e-8)

ll_qmatch <- log_loss_matrix(truth, oof_mlogit_qmatch)
ll_baseline <- log_loss_matrix(truth, saved_base$oof_mlogit)
cat("\nPooled OOF m8trpg (qmatch) log loss:", sprintf("%.10f", ll_qmatch), "\n")
cat("Pooled OOF m8trpg (current best) log loss:", sprintf("%.10f", ll_baseline), "\n")
cat("mlogit-only gain (baseline - qmatch, positive = qmatch better):",
    sprintf("%.6f", ll_baseline - ll_qmatch), "\n")

dir.create("data_processed/codex_transductive", recursive = TRUE, showWarnings = FALSE)
saveRDS(list(oof_mlogit_qmatch = oof_mlogit_qmatch, fold_of_row = fold_of_row,
             fold_diagnostics = do.call(rbind, fold_diagnostics)),
        "data_processed/codex_transductive/qmatch_oof.rds")
write.csv(do.call(rbind, fold_diagnostics),
          "data_processed/codex_transductive/qmatch_fold_diagnostics.csv", row.names = FALSE)

cat("\nTotal elapsed:", round(as.numeric(Sys.time() - t0, units = "mins"), 2), "min\n")
