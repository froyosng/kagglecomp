source("R/codex_shared_utility_common.R")

dir.create(
  shared_utility_output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

add_year_features <- function(features) {
  for (year_level in 2:7) {
    indicator <- as.integer(features$yearind == year_level)
    features[[paste0("P_year", year_level)]] <-
      features$Price_num * indicator
    features[[paste0("In_year", year_level)]] <-
      features$inside * indicator
  }
  features
}

year_formula <- function() {
  base <- m8trpg_formula("none")
  base_rhs <- sub("\\| 0$", "", as.character(base)[3])
  extra <- c(
    paste0("P_year", 2:7),
    paste0("In_year", 2:7)
  )
  formula <- as.formula(paste(
    "chosen ~", base_rhs, "+",
    paste(extra, collapse = " + "), "| 0"
  ))
  environment(formula) <- environment()
  formula
}

fit_predict_year <- function(train_long, valid_long) {
  train_long <- ensure_choice_long(train_long)
  valid_long <- ensure_choice_long(valid_long)
  scaler <- choice_scaler(train_long)
  train_features <- add_year_features(make_m8trpg_features(
    train_long, scaler$ctr, scaler$scl, "none"
  ))
  valid_features <- add_year_features(make_m8trpg_features(
    valid_long, scaler$ctr, scaler$scl, "none"
  ))
  formula <- year_formula()
  model <- mlogit(
    formula,
    data = train_features,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen"
  )

  indexed <- dfidx(
    valid_features,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen"
  )
  class(indexed) <- c("dfidx_mlogit", class(indexed))
  model_frame <- model.frame(indexed, formula, balanced = TRUE)
  design <- model.matrix(model_frame, rhs = 1:3)
  coefficient <- coef(model)
  stopifnot(all(names(coefficient) %in% colnames(design)))
  raw_margin <- as.numeric(
    design[, names(coefficient), drop = FALSE] %*% coefficient
  )
  task_map <- unique(valid_features[, c("chid", "No")])
  task_map <- task_map[order(task_map$No), , drop = FALSE]
  margin_chid <- as.character(dfidx::idx(model_frame, 1))
  margin_alt <- as.integer(as.character(dfidx::idx(model_frame, 2)))
  margin_order <- order(
    match(margin_chid, task_map$chid),
    margin_alt
  )
  stopifnot(
    identical(
      margin_chid[margin_order],
      rep(task_map$chid, each = 4L)
    ),
    identical(
      margin_alt[margin_order],
      rep(1:4, times = nrow(task_map))
    )
  )
  extra_names <- c(
    paste0("P_year", 2:7),
    paste0("In_year", 2:7)
  )
  list(
    pred = softmax_margins(raw_margin[margin_order]),
    no = task_map$No,
    extra_coef = coefficient[extra_names]
  )
}

train <- read.csv("csv files/train.csv")
year_counts <- as.data.frame(table(
  yearind = train$yearind[!duplicated(train$Case)]
))
year_counts$yearind <- as.integer(as.character(year_counts$yearind))
stopifnot(
  identical(sort(year_counts$yearind), 1:7),
  min(year_counts$Freq) >= 40
)
write_result_csv(
  year_counts,
  file.path(shared_utility_output_dir, "yearind_counts.csv")
)

# Canonical single-split screen.
split_data <- readRDS("data_processed/train_val_split.rds")
screen_cache <- readRDS(
  "data_processed/codex_behavioral_round/mlp_screen.rds"
)
screen_fit <- fit_predict_year(
  split_data$train_long_tr,
  split_data$train_long_val
)
screen_pred <- screen_fit$pred[
  match(screen_cache$validation_no, screen_fit$no), ,
  drop = FALSE
]
# Recover the exact m8trpg screen prediction and assert the established score.
screen_base_fit <- fit_m8trpg_model(
  ensure_choice_long(split_data$train_long_tr)
)
screen_base_result <- predict_m8trpg_margin(
  screen_base_fit,
  ensure_choice_long(split_data$train_long_val)
)
screen_baseline <- aligned_prediction(
  screen_base_result,
  screen_cache$validation_no
)
stopifnot(
  abs(log_loss_matrix(screen_cache$truth, screen_baseline) -
    1.15968144721113) < 1e-8
)
screen_summary <- data.frame(
  baseline_logloss =
    log_loss_matrix(screen_cache$truth, screen_baseline),
  year_logloss =
    log_loss_matrix(screen_cache$truth, screen_pred),
  gain =
    log_loss_matrix(screen_cache$truth, screen_baseline) -
    log_loss_matrix(screen_cache$truth, screen_pred)
)
write_result_csv(
  screen_summary,
  file.path(shared_utility_output_dir, "yearind_screen.csv")
)
cat(sprintf(
  "yearind single-split: baseline %.9f, candidate %.9f, gain %+.9f\n",
  screen_summary$baseline_logloss,
  screen_summary$year_logloss,
  screen_summary$gain
))

# This candidate is cheap enough to confirm on all canonical folds even if its
# one split is weak; doing so prevents an arbitrary split from closing the only
# never-tested seven-level covariate.
train_long <- reshape_choice_long(train)
base_cache <- readRDS("data_processed/oof_ensemble_v10.rds")
truth <- base_cache$oof_truth
base_mlogit <- base_cache$oof_mlogit
original_xgb <- base_cache$oof_xgb
v11 <- 0.8 * base_mlogit + 0.2 * original_xgb
fold_map <- base_cache$fold_of_case
row_fold <- unname(fold_map[as.character(train$Case)])
mlp_oof <- readRDS(
  "data_processed/codex_behavioral_round/mlp_oof.rds"
)$oof[[1]]
mlp_precision <- readRDS(
  "data_processed/codex_behavioral_round/mlp_precision.rds"
)
current <- mlp_precision$crossfit_prediction
mlp_weight_by_fold <- mlp_precision$fold_results$mlp_weight[
  match(1:5, mlp_precision$fold_results$fold)
]
row_mlp_weight <- mlp_weight_by_fold[row_fold]
stopifnot(
  abs(log_loss_matrix(truth, base_mlogit) - 1.147021) < 1e-6,
  abs(log_loss_matrix(truth, v11) - 1.145094) < 1e-6,
  abs(log_loss_matrix(truth, current) - 1.143789442) < 1e-8
)

oof_year <- matrix(NA_real_, nrow(train), 4L)
coefficient_rows <- list()
fold_rows <- list()
for (fold in 1:5) {
  valid_cases <- as.integer(names(fold_map)[fold_map == fold])
  fold_train <- train_long[
    !(train_long$Case %in% valid_cases), ,
    drop = FALSE
  ]
  fold_valid <- train_long[
    train_long$Case %in% valid_cases, ,
    drop = FALSE
  ]
  fitted <- fit_predict_year(fold_train, fold_valid)
  valid_rows <- which(row_fold == fold)
  oof_year[valid_rows, ] <- fitted$pred[
    match(train$No[valid_rows], fitted$no), ,
    drop = FALSE
  ]
  coefficient_rows[[fold]] <- data.frame(
    fold = fold,
    term = names(fitted$extra_coef),
    coefficient = as.numeric(fitted$extra_coef)
  )
  fold_rows[[fold]] <- data.frame(
    fold = fold,
    baseline_logloss = log_loss_matrix(
      truth[valid_rows, , drop = FALSE],
      base_mlogit[valid_rows, , drop = FALSE]
    ),
    candidate_logloss = log_loss_matrix(
      truth[valid_rows, , drop = FALSE],
      oof_year[valid_rows, , drop = FALSE]
    )
  )
  fold_rows[[fold]]$gain <-
    fold_rows[[fold]]$baseline_logloss -
    fold_rows[[fold]]$candidate_logloss
  cat(sprintf(
    "yearind fold %d: baseline %.9f, candidate %.9f, gain %+.9f\n",
    fold,
    fold_rows[[fold]]$baseline_logloss,
    fold_rows[[fold]]$candidate_logloss,
    fold_rows[[fold]]$gain
  ))
}
stopifnot(!anyNA(oof_year))

replacement_v11 <- 0.8 * oof_year + 0.2 * original_xgb
adjusted_current <-
  (1 - row_mlp_weight) * replacement_v11 +
  row_mlp_weight * mlp_oof
summary <- data.frame(
  baseline_mlogit_logloss = log_loss_matrix(truth, base_mlogit),
  year_mlogit_logloss = log_loss_matrix(truth, oof_year),
  mlogit_gain =
    log_loss_matrix(truth, base_mlogit) -
    log_loss_matrix(truth, oof_year),
  v11_logloss = log_loss_matrix(truth, v11),
  replacement_v11_logloss =
    log_loss_matrix(truth, replacement_v11),
  replacement_v11_gain =
    log_loss_matrix(truth, v11) -
    log_loss_matrix(truth, replacement_v11),
  current_logloss = log_loss_matrix(truth, current),
  adjusted_current_logloss =
    log_loss_matrix(truth, adjusted_current),
  adjusted_current_gain =
    log_loss_matrix(truth, current) -
    log_loss_matrix(truth, adjusted_current)
)
bootstrap <- respondent_bootstrap_comparison(
  truth,
  current,
  adjusted_current,
  train$Case
)
write_result_csv(
  summary,
  file.path(shared_utility_output_dir, "yearind_cv_summary.csv")
)
write_result_csv(
  do.call(rbind, fold_rows),
  file.path(shared_utility_output_dir, "yearind_cv_folds.csv")
)
write_result_csv(
  do.call(rbind, coefficient_rows),
  file.path(shared_utility_output_dir, "yearind_cv_coefficients.csv")
)
write_result_csv(
  bootstrap,
  file.path(shared_utility_output_dir, "yearind_bootstrap.csv")
)
saveRDS(
  list(
    summary = summary,
    fold_results = do.call(rbind, fold_rows),
    coefficients = do.call(rbind, coefficient_rows),
    bootstrap = bootstrap,
    oof = oof_year,
    replacement_v11 = replacement_v11,
    adjusted_current = adjusted_current
  ),
  file.path(shared_utility_output_dir, "yearind_cv.rds")
)
print(summary, digits = 9)
print(bootstrap, digits = 9)
