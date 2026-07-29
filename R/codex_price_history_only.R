# Price-only design-history anchoring: isolates the price-gap-only mechanism
# from the both_k3 near-miss (R/codex_history_prior_smoothing.R), whose
# attribute-familiarity term flipped sign across folds while its price term
# stayed negative and stable. See codex_price_history_preregister.md for the
# frozen family (k in {3,9,27}), CV/repeated-CV design, and promotion rule --
# all fixed before this script was ever run.
#
# Every stage below hard-asserts a known cached number before trusting a new
# one, matching this project's established verification discipline.

suppressPackageStartupMessages({
  library(mlogit)
  library(dfidx)
})

old_history_stage <- Sys.getenv(
  "CODEX_HISTORY_PRIOR_STAGE", unset = NA_character_
)
Sys.setenv(CODEX_HISTORY_PRIOR_STAGE = "define")
source("R/codex_history_prior_smoothing.R")
if (is.na(old_history_stage)) {
  Sys.unsetenv("CODEX_HISTORY_PRIOR_STAGE")
} else {
  Sys.setenv(CODEX_HISTORY_PRIOR_STAGE = old_history_stage)
}

price_history_output_dir <- "data_processed/codex_price_history"
dir.create(
  price_history_output_dir, recursive = TRUE, showWarnings = FALSE
)

price_only_strengths <- c(3L, 9L, 27L)
price_only_family_size <- length(price_only_strengths)
price_only_candidate_name <- function(strength) {
  paste0("price_only_k", strength)
}

price_only_formula <- function(strength) {
  base <- m8trpg_formula("none")
  base_rhs <- sub("\\| 0$", "", as.character(base)[3])
  term <- paste0("hist_prior_price_gap_k", strength)
  formula <- as.formula(paste("chosen ~", base_rhs, "+", term, "| 0"))
  environment(formula) <- environment()
  formula
}

fit_predict_price_only <- function(source_long, target_long, strength) {
  scaler <- choice_scaler(source_long)
  source_features <- make_m8trpg_features(
    source_long, scaler$ctr, scaler$scl, "none"
  )
  target_features <- make_m8trpg_features(
    target_long, scaler$ctr, scaler$scl, "none"
  )
  formula <- price_only_formula(strength)
  model <- mlogit(
    formula,
    data = source_features,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen"
  )
  indexed <- dfidx(
    target_features,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen"
  )
  class(indexed) <- c("dfidx_mlogit", class(indexed))
  frame <- model.frame(indexed, formula, balanced = TRUE)
  design <- model.matrix(frame, rhs = 1:3)
  coefficient <- coef(model)
  stopifnot(all(names(coefficient) %in% colnames(design)))
  margin <- as.numeric(
    design[, names(coefficient), drop = FALSE] %*% coefficient
  )
  task_map <- unique(target_features[, c("chid", "No")])
  task_map <- task_map[order(task_map$No), , drop = FALSE]
  margin_chid <- as.character(dfidx::idx(frame, 1))
  margin_alt <- as.integer(as.character(dfidx::idx(frame, 2)))
  margin_order <- order(
    match(margin_chid, task_map$chid), margin_alt
  )
  stopifnot(
    identical(
      margin_chid[margin_order], rep(task_map$chid, each = 4L)
    ),
    identical(
      margin_alt[margin_order], rep(1:4, times = nrow(task_map))
    )
  )
  term_name <- paste0("hist_prior_price_gap_k", strength)
  list(
    pred = softmax_margins(margin[margin_order]),
    no = as.integer(task_map$No),
    extra_coefficient = unname(coefficient[[term_name]])
  )
}

repeat_fold_map <- function(cases, seed) {
  cases <- as.integer(cases)
  set.seed(seed)
  fold <- sample(rep(1:5, length.out = length(cases)))
  names(fold) <- as.character(cases)
  stopifnot(
    length(fold) == length(cases),
    all(table(fold) %in% c(227L))
  )
  fold
}

train <- read.csv("csv files/train.csv")
train <- train[order(train$No), , drop = FALSE]
stopifnot(identical(train$No, seq_len(nrow(train))))
truth <- as.matrix(train[, paste0("Ch", 1:4), drop = FALSE])
cases <- unique(train$Case)
stopifnot(length(cases) == 1135L, all(table(train$Case) == 19L))
full_long <- reshape_choice_long(train)

base <- readRDS("data_processed/oof_ensemble_v10.rds")
fold_map_canonical <- base$fold_of_case
shallow_canonical <- readRDS(
  "data_processed/codex_behavioral_round/mlp_oof.rds"
)$oof[["h08_d0.100"]]
v11_canonical <- 0.8 * base$oof_mlogit + 0.2 * base$oof_xgb
current_canonical <- 0.85 * v11_canonical + 0.15 * shallow_canonical
stopifnot(
  abs(
    log_loss_matrix(truth, current_canonical) - 1.143686618134879
  ) < 1e-10
)

repeat_additional_seeds <- c(1907L, 2719L, 6151L, 8293L, 104729L)
repeat_all_seeds <- c(4821L, repeat_additional_seeds)
n_repeats <- length(repeat_all_seeds)

candidate_names <- vapply(
  price_only_strengths, price_only_candidate_name, character(1)
)

## ---- Validation: reconstructed folds must reproduce the already-logged
## both_k3 result exactly, both for the canonical split and for one
## additional repeat seed. If this fails, the fold/prior/feature plumbing
## below is wrong and nothing else in this script should be trusted. ----

validate_canonical_fold1 <- function() {
  validation_cases <- as.integer(
    names(fold_map_canonical)[fold_map_canonical == 1]
  )
  source_raw <- full_long[
    !(full_long$Case %in% validation_cases), , drop = FALSE
  ]
  target_raw <- full_long[
    full_long$Case %in% validation_cases, , drop = FALSE
  ]
  prior <- history_design_prior(source_raw)
  source_hist <- add_prior_smoothed_history(source_raw, prior)
  target_hist <- add_prior_smoothed_history(target_raw, prior)
  fitted <- fit_predict_history_prior(
    source_hist, target_hist, "both_k3"
  )
  target_rows <- train$Case %in% validation_cases
  target_no <- train$No[target_rows]
  reconstructed_mlogit <- fitted$pred[
    match(target_no, fitted$no), , drop = FALSE
  ]
  # history_prior_cv.rds$predictions stores the BLENDED ensemble prediction
  # (0.85*(0.8*history_mlogit+0.2*xgb)+0.15*shallow), not the raw mlogit --
  # reproduce that same blend before comparing, matching
  # R/codex_history_prior_smoothing.R's "cv" stage exactly.
  reconstructed_blend <- 0.85 * (
    0.8 * reconstructed_mlogit +
      0.2 * base$oof_xgb[target_rows, , drop = FALSE]
  ) + 0.15 * shallow_canonical[target_rows, , drop = FALSE]
  cached <- readRDS(
    "data_processed/codex_overnight_queue/history_prior_cv.rds"
  )
  cached_fold1 <- cached$predictions[["both_k3"]][
    target_rows, , drop = FALSE
  ]
  max_diff <- max(abs(reconstructed_blend - cached_fold1))
  cat(sprintf(
    "Validation A (canonical fold 1, both_k3): max abs diff = %.3e\n",
    max_diff
  ))
  stopifnot(max_diff < 1e-8)
}

validate_repeat_seed1907_fold1 <- function() {
  fold_map <- repeat_fold_map(cases, 1907L)
  validation_cases <- as.integer(
    names(fold_map)[fold_map == 1]
  )
  source_raw <- full_long[
    !(full_long$Case %in% validation_cases), , drop = FALSE
  ]
  target_raw <- full_long[
    full_long$Case %in% validation_cases, , drop = FALSE
  ]
  prior <- history_design_prior(source_raw)
  source_hist <- add_prior_smoothed_history(source_raw, prior)
  target_hist <- add_prior_smoothed_history(target_raw, prior)
  fitted <- fit_predict_history_prior(
    source_hist, target_hist, "both_k3"
  )
  checkpoint <- readRDS(
    "data_processed/codex_repeat_cv/checkpoints/seed_1907_fold_1.rds"
  )
  target_no <- checkpoint$validation_no
  reconstructed <- fitted$pred[
    match(target_no, fitted$no), , drop = FALSE
  ]
  max_diff <- max(abs(
    reconstructed - checkpoint$predictions$history_mlogit
  ))
  cat(sprintf(
    paste0(
      "Validation B (repeat seed 1907, fold 1, both_k3): ",
      "max abs diff = %.3e\n"
    ),
    max_diff
  ))
  stopifnot(max_diff < 1e-8)
}

cat("Running plumbing validation before any price-only fit...\n")
validate_canonical_fold1()
validate_repeat_seed1907_fold1()
cat("Both validations passed -- fold/prior/feature reconstruction is",
    "confirmed correct.\n\n")

## ---- Canonical CV (seed 4821) ----

fold_rows <- list()
candidate_oof <- lapply(
  candidate_names, function(x) matrix(NA_real_, nrow(train), 4L)
)
names(candidate_oof) <- candidate_names

run_canonical <- function() {
  for (fold in 1:5) {
    cat(sprintf(
      "canonical seed 4821, fold %d/5\n", fold
    ))
    flush.console()
    validation_cases <- as.integer(
      names(fold_map_canonical)[fold_map_canonical == fold]
    )
    source_raw <- full_long[
      !(full_long$Case %in% validation_cases), , drop = FALSE
    ]
    target_raw <- full_long[
      full_long$Case %in% validation_cases, , drop = FALSE
    ]
    prior <- history_design_prior(source_raw)
    source_hist <- add_prior_smoothed_history(source_raw, prior)
    target_hist <- add_prior_smoothed_history(target_raw, prior)
    target_rows <- train$Case %in% validation_cases
    target_no <- train$No[target_rows]

    for (strength in price_only_strengths) {
      candidate <- price_only_candidate_name(strength)
      fitted <- fit_predict_price_only(
        source_hist, target_hist, strength
      )
      history_mlogit <- fitted$pred[
        match(target_no, fitted$no), , drop = FALSE
      ]
      candidate_fold <- 0.85 * (
        0.8 * history_mlogit +
          0.2 * base$oof_xgb[target_rows, , drop = FALSE]
      ) + 0.15 * shallow_canonical[target_rows, , drop = FALSE]
      candidate_oof[[candidate]][target_rows, ] <<- candidate_fold
      fold_truth <- truth[target_rows, , drop = FALSE]
      fold_current <- current_canonical[target_rows, , drop = FALSE]
      fold_rows[[length(fold_rows) + 1L]] <<- data.frame(
        candidate = candidate,
        repeat_seed = 4821L,
        repeat_index = 0L,
        fold = fold,
        baseline_loss = log_loss_matrix(fold_truth, fold_current),
        candidate_loss = log_loss_matrix(fold_truth, candidate_fold),
        gain = log_loss_matrix(fold_truth, fold_current) -
          log_loss_matrix(fold_truth, candidate_fold),
        coefficient = fitted$extra_coefficient
      )
    }
  }
}
run_canonical()

## ---- Repeated CV: 5 additional seeds, reusing cached xgb/MLP components ----

repeat_case_gain <- list()
for (candidate in candidate_names) {
  repeat_case_gain[[candidate]] <- matrix(
    NA_real_, nrow = length(cases), ncol = n_repeats,
    dimnames = list(as.character(cases), as.character(repeat_all_seeds))
  )
}

## canonical repeat's per-respondent gain, filled in after the loop below
## (needs candidate_oof fully populated first)

run_additional_repeat <- function(seed, repeat_index) {
  checkpoint_path_ok <- TRUE
  fold_map <- repeat_fold_map(cases, seed)
  for (fold in 1:5) {
    cat(sprintf(
      "repeat seed %d, fold %d/5\n", seed, fold
    ))
    flush.console()
    validation_cases <- as.integer(
      names(fold_map)[fold_map == fold]
    )
    source_raw <- full_long[
      !(full_long$Case %in% validation_cases), , drop = FALSE
    ]
    target_raw <- full_long[
      full_long$Case %in% validation_cases, , drop = FALSE
    ]
    prior <- history_design_prior(source_raw)
    source_hist <- add_prior_smoothed_history(source_raw, prior)
    target_hist <- add_prior_smoothed_history(target_raw, prior)
    target_rows <- train$Case %in% validation_cases
    target_no <- train$No[target_rows]

    checkpoint <- readRDS(sprintf(
      "data_processed/codex_repeat_cv/checkpoints/seed_%d_fold_%d.rds",
      seed, fold
    ))
    stopifnot(
      identical(
        as.integer(checkpoint$validation_no), as.integer(target_no)
      )
    )
    xgb_component <- checkpoint$predictions$original_xgb
    mlp_component <- checkpoint$predictions$shallow_mlp
    mlogit_component <- checkpoint$predictions$mlogit
    baseline_fold <- 0.85 * (
      0.8 * mlogit_component + 0.2 * xgb_component
    ) + 0.15 * mlp_component
    fold_truth <- truth[target_rows, , drop = FALSE]

    for (strength in price_only_strengths) {
      candidate <- price_only_candidate_name(strength)
      fitted <- fit_predict_price_only(
        source_hist, target_hist, strength
      )
      history_mlogit <- fitted$pred[
        match(target_no, fitted$no), , drop = FALSE
      ]
      candidate_fold <- 0.85 * (
        0.8 * history_mlogit + 0.2 * xgb_component
      ) + 0.15 * mlp_component
      fold_rows[[length(fold_rows) + 1L]] <<- data.frame(
        candidate = candidate,
        repeat_seed = seed,
        repeat_index = repeat_index,
        fold = fold,
        baseline_loss = log_loss_matrix(fold_truth, baseline_fold),
        candidate_loss = log_loss_matrix(fold_truth, candidate_fold),
        gain = log_loss_matrix(fold_truth, baseline_fold) -
          log_loss_matrix(fold_truth, candidate_fold),
        coefficient = fitted$extra_coefficient
      )
      row_gain <- (
        -rowSums(
          fold_truth * log(pmax(baseline_fold, 1e-15))
        )
      ) - (
        -rowSums(
          fold_truth * log(pmax(candidate_fold, 1e-15))
        )
      )
      case_gain_fold <- tapply(
        row_gain, train$Case[target_rows], mean
      )
      repeat_case_gain[[candidate]][
        names(case_gain_fold), as.character(seed)
      ] <<- case_gain_fold
    }
  }
}

repeat_checkpoint <- file.path(
  price_history_output_dir, "repeat_progress.rds"
)
completed_seeds <- integer(0)
if (file.exists(repeat_checkpoint)) {
  saved <- readRDS(repeat_checkpoint)
  fold_rows <- saved$fold_rows
  repeat_case_gain <- saved$repeat_case_gain
  candidate_oof <- saved$candidate_oof
  completed_seeds <- saved$completed_seeds
  cat("Resumed from checkpoint; completed seeds:",
      paste(completed_seeds, collapse = ", "), "\n")
}

if (!(4821L %in% completed_seeds)) {
  completed_seeds <- c(completed_seeds, 4821L)
}

for (index in seq_along(repeat_additional_seeds)) {
  seed <- repeat_additional_seeds[[index]]
  if (seed %in% completed_seeds) {
    cat("Repeat seed", seed, "already completed, skipping.\n")
    next
  }
  run_additional_repeat(seed, index)
  completed_seeds <- c(completed_seeds, seed)
  saveRDS(
    list(
      fold_rows = fold_rows,
      repeat_case_gain = repeat_case_gain,
      candidate_oof = candidate_oof,
      completed_seeds = completed_seeds
    ),
    repeat_checkpoint
  )
  invisible(gc())
}

## ---- Fill in the canonical repeat's per-respondent gain ----

for (candidate in candidate_names) {
  prediction <- candidate_oof[[candidate]]
  stopifnot(!anyNA(prediction))
  row_gain <- (
    -rowSums(truth * log(pmax(current_canonical, 1e-15)))
  ) - (
    -rowSums(truth * log(pmax(prediction, 1e-15)))
  )
  case_gain <- tapply(row_gain, train$Case, mean)
  repeat_case_gain[[candidate]][names(case_gain), "4821"] <- case_gain
}

for (candidate in candidate_names) {
  stopifnot(!anyNA(repeat_case_gain[[candidate]]))
}

## ---- Summaries: fold table, repeat table, bootstrap ----

fold_summary <- do.call(rbind, fold_rows)
write.csv(
  fold_summary,
  file.path(price_history_output_dir, "price_history_folds.csv"),
  row.names = FALSE
)

repeat_rows <- list()
for (candidate in candidate_names) {
  cg <- repeat_case_gain[[candidate]]
  for (seed in repeat_all_seeds) {
    repeat_rows[[length(repeat_rows) + 1L]] <- data.frame(
      candidate = candidate,
      repeat_seed = seed,
      mean_case_gain = mean(cg[, as.character(seed)])
    )
  }
}
repeat_summary <- do.call(rbind, repeat_rows)
write.csv(
  repeat_summary,
  file.path(
    price_history_output_dir, "price_history_repeat_summary.csv"
  ),
  row.names = FALSE
)

bootstrap_average_gain <- function(case_gain_matrix, family_size,
                                    n_boot = 100000L, seed = 4821L) {
  stopifnot(
    nrow(case_gain_matrix) == 1135L,
    ncol(case_gain_matrix) == n_repeats,
    !anyNA(case_gain_matrix)
  )
  case_gain <- rowMeans(case_gain_matrix)
  set.seed(seed)
  bootstrap <- numeric(n_boot)
  n_case <- length(case_gain)
  for (start in seq.int(1L, n_boot, by = 1000L)) {
    stop_at <- min(n_boot, start + 999L)
    n_this <- stop_at - start + 1L
    sampled <- matrix(
      sample.int(n_case, n_case * n_this, replace = TRUE),
      nrow = n_case
    )
    bootstrap[start:stop_at] <- colMeans(matrix(
      case_gain[sampled], nrow = n_case
    ))
  }
  family_alpha <- 0.05 / family_size
  context_family_alpha <- 0.05 / 16L
  list(
    summary = data.frame(
      point_gain = mean(case_gain),
      bootstrap_mean = mean(bootstrap),
      bootstrap_sd = sd(bootstrap),
      lower_95 = unname(quantile(bootstrap, 0.025)),
      upper_95 = unname(quantile(bootstrap, 0.975)),
      lower_99 = unname(quantile(bootstrap, 0.005)),
      upper_99 = unname(quantile(bootstrap, 0.995)),
      lower_family3 = unname(quantile(bootstrap, family_alpha / 2)),
      upper_family3 = unname(quantile(bootstrap, 1 - family_alpha / 2)),
      lower_family16_context = unname(quantile(
        bootstrap, context_family_alpha / 2
      )),
      upper_family16_context = unname(quantile(
        bootstrap, 1 - context_family_alpha / 2
      )),
      win_rate = mean(bootstrap > 0),
      n_boot = n_boot
    ),
    case_gain = case_gain
  )
}

bootstrap_rows <- list()
decision_rows <- list()
for (candidate in candidate_names) {
  bootstrap <- bootstrap_average_gain(
    repeat_case_gain[[candidate]], family_size = price_only_family_size
  )
  bootstrap_rows[[candidate]] <- cbind(
    data.frame(candidate = candidate), bootstrap$summary
  )

  positive_repeats <- sum(colMeans(repeat_case_gain[[candidate]]) > 0)
  candidate_folds <- fold_summary[fold_summary$candidate == candidate, ]
  stopifnot(nrow(candidate_folds) == n_repeats * 5L)
  negative_coefficient_count <- sum(candidate_folds$coefficient < 0)

  decision_rows[[candidate]] <- data.frame(
    candidate = candidate,
    point_gain = bootstrap$summary$point_gain,
    lower_family3 = bootstrap$summary$lower_family3,
    positive_repeats = positive_repeats,
    negative_coefficient_folds = negative_coefficient_count,
    total_folds = nrow(candidate_folds),
    promote = (
      bootstrap$summary$point_gain > 0 &&
        bootstrap$summary$lower_family3 > 0 &&
        positive_repeats >= 5L &&
        negative_coefficient_count >= 27L
    )
  )
}
bootstrap_summary <- do.call(rbind, bootstrap_rows)
decision <- do.call(rbind, decision_rows)

write.csv(
  bootstrap_summary,
  file.path(price_history_output_dir, "price_history_bootstrap.csv"),
  row.names = FALSE
)
write.csv(
  decision,
  file.path(price_history_output_dir, "price_history_decision.csv"),
  row.names = FALSE
)
saveRDS(
  list(
    fold_summary = fold_summary,
    repeat_summary = repeat_summary,
    bootstrap_summary = bootstrap_summary,
    decision = decision,
    repeat_case_gain = repeat_case_gain,
    candidate_oof = candidate_oof
  ),
  file.path(price_history_output_dir, "price_history_results.rds")
)

print(fold_summary, digits = 10, row.names = FALSE)
print(repeat_summary, digits = 10, row.names = FALSE)
print(bootstrap_summary, digits = 10, row.names = FALSE)
print(decision, digits = 10, row.names = FALSE)
