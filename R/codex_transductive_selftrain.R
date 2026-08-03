# Candidate B: confident self-training on m8trpg, per
# codex_transductive_preregister.md Section 4.
#
# Per fold k:
#  1. Fit VANILLA m8trpg on the fold's ~908 real training respondents.
#  2. Score test.csv (263 respondents, no labels) with that fold-specific model.
#  3. Select confident test TASKS: max predicted prob > 0.85 (fixed, pre-
#     registered threshold, not swept).
#  4. Augment the fold's training data with those confident tasks' full 4-
#     alternative rows, pseudo-"chosen" = argmax, likelihood weight 1 (same as
#     real rows).
#  5. Refit m8trpg on (908 real respondents) union (confident pseudo test tasks).
#  6. Predict on the SAME held-out 227 REAL respondents used everywhere else in
#     this track -- never on the pseudo-labeled rows themselves for the verdict.
#
# Anti-circularity stress test included at the end: for fold 1, explicitly
# compute and print the WRONG, circular number (loss of the augmented model
# evaluated on the very pseudo-labeled rows it was fit to reproduce) side by side
# with the honest held-out number, so the two are never confused in the write-up.

source("R/codex_transductive_common.R")
library(mlogit)

t0 <- Sys.time()
train <- read.csv("csv files/train.csv")
test <- read.csv("csv files/test.csv")
truth <- as.matrix(train[, paste0("Ch", 1:4)])
train_long <- to_long(train)

# placeholder outcome columns so to_long()/dfidx() accept test.csv (predict()
# ignores the outcome column entirely; test's real Ch1-4 do not exist)
test_ph <- test
test_ph$Ch1 <- 1; test_ph$Ch2 <- 0; test_ph$Ch3 <- 0; test_ph$Ch4 <- 0
test_long <- to_long(test_ph)
test_map <- unique(test_long[, c("chid", "No")])

saved_base <- readRDS("data_processed/oof_ensemble_v10.rds")
fold_of_case <- saved_base$fold_of_case
fold_of_row <- unname(fold_of_case[as.character(train$Case)])
stopifnot(!anyNA(fold_of_row), all(table(fold_of_case) == 227))

scaler_vars <- c("incomea","agea","milesa","nighta","genderind","Urbind","educind")
CONF_THRESHOLD <- 0.85

oof_mlogit_selftrain <- matrix(NA_real_, nrow(train), 4)
fold_diag <- vector("list", 5L)
circularity_demo <- NULL

for (k in 1:5) {
  val_cases_k <- names(fold_of_case)[fold_of_case == k]
  tr_k <- train_long[!(as.character(train_long$Case) %in% val_cases_k), ]
  va_k <- train_long[as.character(train_long$Case) %in% val_cases_k, ]

  std_ctr <- sapply(tr_k[scaler_vars], mean, na.rm = TRUE)
  std_scl <- sapply(tr_k[scaler_vars], sd, na.rm = TRUE)
  std_scl[std_scl == 0] <- 1

  # --- Step 1: vanilla model on this fold's real training respondents ---
  tr_feat <- make_features_m8trpg(tr_k, std_ctr, std_scl, recode_fn = NULL)
  mdat_tr <- dfidx(tr_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
  mod_vanilla_k <- mlogit(m8trpg_formula(), data = mdat_tr)

  # --- Step 2: score test.csv with this fold-specific vanilla model ---
  test_feat_k <- make_features_m8trpg(test_long, std_ctr, std_scl, recode_fn = NULL)
  mdat_test_k <- dfidx(test_feat_k, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
  pred_test_k <- predict(mod_vanilla_k, newdata = mdat_test_k)
  pred_test_ordered <- pred_test_k[match(test_map$chid, rownames(pred_test_k)), ]
  stopifnot(!anyNA(pred_test_ordered), all(abs(rowSums(pred_test_ordered) - 1) < 1e-8))

  # --- Step 3: confident tasks ---
  max_prob <- apply(pred_test_ordered, 1, max)
  argmax_alt <- apply(pred_test_ordered, 1, which.max)
  confident <- max_prob > CONF_THRESHOLD
  n_confident <- sum(confident)
  confident_chid <- test_map$chid[confident]
  confident_argmax <- setNames(argmax_alt[confident], confident_chid)

  # --- Step 4: build augmented training long-data ---
  pseudo_rows <- test_long[test_long$chid %in% confident_chid, ]
  pseudo_rows$chosen <- as.integer(pseudo_rows$alt == confident_argmax[pseudo_rows$chid])
  stopifnot(all(tapply(pseudo_rows$chosen, pseudo_rows$chid, sum) == 1L))

  tr_k_aug <- rbind(
    tr_k[, intersect(names(tr_k), names(pseudo_rows))],
    pseudo_rows[, intersect(names(tr_k), names(pseudo_rows))]
  )
  stopifnot(length(unique(tr_k_aug$chid)) == length(unique(tr_k$chid)) + length(unique(pseudo_rows$chid)))

  # --- Step 5: refit on augmented data (scaler frozen from REAL respondents only) ---
  tr_aug_feat <- make_features_m8trpg(tr_k_aug, std_ctr, std_scl, recode_fn = NULL)
  mdat_tr_aug <- dfidx(tr_aug_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
  mod_aug_k <- mlogit(m8trpg_formula(), data = mdat_tr_aug)

  # --- Step 6: predict on the SAME held-out 227 real respondents ---
  va_feat <- make_features_m8trpg(va_k, std_ctr, std_scl, recode_fn = NULL)
  mdat_va <- dfidx(va_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
  pred_va_aug <- predict(mod_aug_k, newdata = mdat_va)

  va_map <- unique(va_k[, c("chid", "No")])
  row_idx <- match(va_map$No, train$No)
  pred_ordered <- pred_va_aug[match(va_map$chid, rownames(pred_va_aug)), ]
  oof_mlogit_selftrain[row_idx, ] <- pred_ordered

  fold_ll <- log_loss_matrix(truth[row_idx, ], pred_ordered)
  fold_diag[[k]] <- data.frame(fold = k, n_confident_tasks = n_confident,
                                pct_confident = n_confident / nrow(test_map) * 100,
                                fold_logloss_honest = fold_ll)
  cat("Fold", k, ": n_confident_test_tasks =", n_confident,
      sprintf("(%.1f%% of %d)", n_confident / nrow(test_map) * 100, nrow(test_map)),
      " honest held-out fold ll =", round(fold_ll, 6),
      " elapsed =", round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s\n")

  # --- Anti-circularity stress test, fold 1 only ---
  if (k == 1 && n_confident > 0) {
    pseudo_feat <- make_features_m8trpg(pseudo_rows, std_ctr, std_scl, recode_fn = NULL)
    mdat_pseudo <- dfidx(pseudo_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
    pred_on_pseudo <- predict(mod_aug_k, newdata = mdat_pseudo)
    pseudo_truth <- matrix(0, nrow(pred_on_pseudo), 4)
    pseudo_chid_order <- rownames(pred_on_pseudo)
    for (i in seq_along(pseudo_chid_order)) {
      pseudo_truth[i, confident_argmax[pseudo_chid_order[i]]] <- 1
    }
    circular_ll <- log_loss_matrix(pseudo_truth, pred_on_pseudo)
    circularity_demo <- data.frame(
      fold = 1,
      n_pseudo_tasks = nrow(pred_on_pseudo),
      circular_logloss_on_own_pseudo_labels = circular_ll,
      honest_logloss_on_genuine_holdout = fold_ll
    )
    cat("\n[Anti-circularity stress test, fold 1] loss evaluated on the model's OWN\n",
        "pseudo-labeled rows (WRONG, circular, NOT used for the verdict):",
        sprintf("%.6f", circular_ll), "\n",
        "vs. honest loss on genuine held-out train respondents (the real verdict input):",
        sprintf("%.6f", fold_ll), "\n\n", sep = "")
  }
}

stopifnot(!anyNA(oof_mlogit_selftrain), all(oof_mlogit_selftrain > 0),
          max(abs(rowSums(oof_mlogit_selftrain) - 1)) < 1e-8)

ll_selftrain <- log_loss_matrix(truth, oof_mlogit_selftrain)
ll_baseline <- log_loss_matrix(truth, saved_base$oof_mlogit)
cat("\nPooled OOF m8trpg (self-train) log loss:", sprintf("%.10f", ll_selftrain), "\n")
cat("Pooled OOF m8trpg (current best) log loss:", sprintf("%.10f", ll_baseline), "\n")
cat("mlogit-only gain (baseline - selftrain, positive = selftrain better):",
    sprintf("%.6f", ll_baseline - ll_selftrain), "\n")

dir.create("data_processed/codex_transductive", recursive = TRUE, showWarnings = FALSE)
saveRDS(list(oof_mlogit_selftrain = oof_mlogit_selftrain, fold_of_row = fold_of_row,
             fold_diag = do.call(rbind, fold_diag), circularity_demo = circularity_demo,
             conf_threshold = CONF_THRESHOLD),
        "data_processed/codex_transductive/selftrain_oof.rds")
write.csv(do.call(rbind, fold_diag),
          "data_processed/codex_transductive/selftrain_fold_diagnostics.csv", row.names = FALSE)
write.csv(circularity_demo,
          "data_processed/codex_transductive/selftrain_circularity_stress_test.csv", row.names = FALSE)

cat("\nTotal elapsed:", round(as.numeric(Sys.time() - t0, units = "mins"), 2), "min\n")
