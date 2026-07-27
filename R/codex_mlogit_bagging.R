## Bootstrap-bag the dominant m8trpg component while preserving respondent
## panels. Each bootstrap draw samples whole respondents with replacement and
## relabels every copied respondent/task before dfidx construction.

suppressPackageStartupMessages({
  library(mlogit)
  library(dfidx)
})
source("R/codex_shift_common.R")

stage <- Sys.getenv("CODEX_STAGE", "screen")
n_bags <- as.integer(Sys.getenv("CODEX_N_BAGS", "15"))
stopifnot(stage %in% c("screen", "cv"), n_bags >= 1L)

output_dir <- "data_processed/codex_final_round"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

ensure_choice_ids <- function(df) {
  df <- as.data.frame(df)
  if (!("chid" %in% names(df))) {
    df$chid <- paste(df$Case, df$Task, sep = "_")
  }
  if (!("d2" %in% names(df))) df$d2 <- as.integer(df$alt == 2L)
  if (!("d3" %in% names(df))) df$d3 <- as.integer(df$alt == 3L)
  df
}

assert_panel_integrity <- function(df, expected_respondents = NULL) {
  df <- as.data.frame(df)
  stopifnot(
    all(c("Case", "Task", "No", "chid", "alt") %in% names(df)),
    !anyNA(df[, c("Case", "Task", "No", "chid", "alt")]),
    !anyDuplicated(df[, c("chid", "alt")])
  )
  chid_counts <- table(df$chid)
  stopifnot(
    length(chid_counts) * 4L == nrow(df),
    all(chid_counts == 4L)
  )
  chid_case <- unique(df[, c("chid", "Case"), drop = FALSE])
  stopifnot(!anyDuplicated(chid_case$chid))
  case_task <- table(df$Case, df$Task)
  stopifnot(all(case_task == 4L))
  if (!is.null(expected_respondents)) {
    stopifnot(
      length(unique(df$Case)) == expected_respondents,
      nrow(df) == expected_respondents * 19L * 4L
    )
  }
  invisible(TRUE)
}

bootstrap_respondents <- function(train_long, seed) {
  train_long <- ensure_choice_ids(train_long)
  source_cases <- sort(unique(train_long$Case))
  set.seed(seed)
  sampled_cases <- sample(
    source_cases, length(source_cases), replace = TRUE
  )

  copies <- vector("list", length(sampled_cases))
  for (draw in seq_along(sampled_cases)) {
    source_case <- sampled_cases[[draw]]
    block <- train_long[
      train_long$Case == source_case,
      ,
      drop = FALSE
    ]
    block <- block[order(block$Task, block$alt), , drop = FALSE]
    stopifnot(
      nrow(block) == 19L * 4L,
      all(table(block$Task) == 4L)
    )

    # No is also relabeled because feature construction sorts on No/alt.
    # Leaving duplicated source No values would interleave alternatives from
    # repeated bootstrap copies even if Case/chid themselves were unique.
    synthetic_case <- draw
    synthetic_task_no <- match(block$Task, sort(unique(block$Task)))
    block$Case <- synthetic_case
    block$No <- (draw - 1L) * 19L + synthetic_task_no
    block$chid <- paste0("boot_", draw, "_task_", block$Task)
    block$bootstrap_source_case <- source_case
    copies[[draw]] <- block
  }

  out <- do.call(rbind, copies)
  rownames(out) <- NULL
  assert_panel_integrity(out, length(sampled_cases))
  attr(out, "sampled_cases") <- sampled_cases
  out
}

run_bagged_fit <- function(train_long, valid_long, fold_label) {
  valid_long <- ensure_choice_ids(valid_long)
  assert_panel_integrity(valid_long)

  predictions <- vector("list", n_bags)
  attempts <- data.frame()
  success <- 0L
  attempt <- 0L
  max_attempts <- n_bags + 10L

  while (success < n_bags && attempt < max_attempts) {
    attempt <- attempt + 1L
    bag_seed <- 730000L + as.integer(fold_label) * 1000L + attempt
    boot <- bootstrap_respondents(train_long, bag_seed)

    started <- proc.time()[["elapsed"]]
    fitted <- tryCatch(
      fit_predict_m8trpg(boot, valid_long),
      error = function(e) e
    )
    elapsed <- proc.time()[["elapsed"]] - started

    if (inherits(fitted, "error")) {
      attempts <- rbind(
        attempts,
        data.frame(
          fold = fold_label, attempt = attempt, seed = bag_seed,
          success = FALSE, elapsed_seconds = elapsed,
          detail = conditionMessage(fitted)
        )
      )
      cat(sprintf(
        "fold %s attempt %d failed: %s\n",
        fold_label, attempt, conditionMessage(fitted)
      ))
      next
    }

    success <- success + 1L
    predictions[[success]] <- fitted$pred
    attempts <- rbind(
      attempts,
      data.frame(
        fold = fold_label, attempt = attempt, seed = bag_seed,
        success = TRUE, elapsed_seconds = elapsed,
        detail = "ok"
      )
    )
    cat(sprintf(
      "fold %s bag %d/%d fitted in %.1fs\n",
      fold_label, success, n_bags, elapsed
    ))
    flush.console()
    rm(boot, fitted)
    gc()
  }

  if (success < n_bags) {
    stop(sprintf(
      "Only %d of %d requested bootstrap fits succeeded in fold %s",
      success, n_bags, fold_label
    ))
  }

  cumulative <- vector("list", n_bags)
  running <- matrix(0, nrow(predictions[[1]]), 4)
  for (b in seq_len(n_bags)) {
    running <- running + predictions[[b]]
    cumulative[[b]] <- running / b
  }
  list(cumulative = cumulative, attempts = attempts)
}

row_loss <- function(truth, pred, eps = 1e-15) {
  pred <- pred / rowSums(pred)
  pred <- pmin(pmax(pred, eps), 1 - eps)
  -rowSums(truth * log(pred))
}

bootstrap_comparison <- function(truth, baseline, candidate, case,
                                 label, replicates = 2000L) {
  respondent_gain <- tapply(
    row_loss(truth, baseline) - row_loss(truth, candidate),
    case,
    mean
  )
  set.seed(4821)
  boot <- replicate(
    replicates,
    mean(sample(
      respondent_gain, length(respondent_gain), replace = TRUE
    ))
  )
  data.frame(
    comparison = label,
    point_gain = mean(respondent_gain),
    bootstrap_mean = mean(boot),
    bootstrap_sd = sd(boot),
    lower_95 = unname(quantile(boot, 0.025)),
    upper_95 = unname(quantile(boot, 0.975)),
    win_rate = mean(boot > 0)
  )
}

if (stage == "screen") {
  split <- readRDS("data_processed/train_val_split.rds")
  tr <- ensure_choice_ids(split$train_long_tr)
  va <- ensure_choice_ids(split$train_long_val)
  va_wide <- split$train_wide_val
  va_wide <- va_wide[order(va_wide$No), , drop = FALSE]
  truth <- as.matrix(va_wide[, paste0("Ch", 1:4)])

  baseline_fit <- fit_predict_m8trpg(tr, va)
  baseline <- baseline_fit$pred[
    match(va_wide$No, baseline_fit$no), , drop = FALSE
  ]
  baseline_loss <- log_loss_matrix(truth, baseline)
  stopifnot(abs(baseline_loss - 1.15968144721113) < 1e-8)

  bagged <- run_bagged_fit(tr, va, fold_label = 0L)
  learning <- data.frame(
    n_bags = seq_len(n_bags),
    baseline_logloss = baseline_loss,
    bagged_logloss = vapply(
      bagged$cumulative,
      function(pred) log_loss_matrix(truth, pred),
      numeric(1)
    )
  )
  learning$gain <- learning$baseline_logloss - learning$bagged_logloss

  write_result_csv(
    learning,
    file.path(output_dir, "mlogit_bagging_screen.csv")
  )
  write_result_csv(
    bagged$attempts,
    file.path(output_dir, "mlogit_bagging_screen_attempts.csv")
  )
  saveRDS(
    list(
      baseline = baseline,
      cumulative = bagged$cumulative,
      truth = truth,
      validation_no = va_wide$No
    ),
    file.path(output_dir, "mlogit_bagging_screen.rds")
  )
  print(learning, digits = 7)
}

if (stage == "cv") {
  train <- read.csv("csv files/train.csv")
  truth <- as.matrix(train[, paste0("Ch", 1:4)])
  train_long <- reshape_choice_long(train)
  saved <- readRDS("data_processed/oof_ensemble_v10.rds")
  baseline <- saved$oof_mlogit
  xgb_single <- saved$oof_xgb
  fold_map <- saved$fold_of_case
  xgb_bagged <- readRDS(
    "data_processed/codex_shift/seed_bagging_oof.rds"
  )$bagged_xgb

  stopifnot(
    abs(log_loss_matrix(truth, baseline) - 1.1470212110518) < 1e-8,
    abs(log_loss_matrix(truth, xgb_single) - 1.178668) < 1e-4,
    abs(log_loss_matrix(truth, xgb_bagged) - 1.176836) < 1e-4,
    abs(log_loss_matrix(
      truth, 0.8 * baseline + 0.2 * xgb_single
    ) - 1.145094) < 5e-6
  )

  oof_by_n <- lapply(
    seq_len(n_bags),
    function(x) matrix(NA_real_, nrow(train), 4)
  )
  attempt_rows <- list()

  for (fold in 1:5) {
    val_cases <- as.integer(names(fold_map)[fold_map == fold])
    tr <- train_long[
      !(train_long$Case %in% val_cases), , drop = FALSE
    ]
    va <- train_long[
      train_long$Case %in% val_cases, , drop = FALSE
    ]
    bagged <- run_bagged_fit(tr, va, fold_label = fold)
    validation_no <- sort(unique(va$No))
    idx <- match(validation_no, train$No)
    stopifnot(
      !anyNA(idx),
      length(idx) == nrow(bagged$cumulative[[1]])
    )
    for (b in seq_len(n_bags)) {
      oof_by_n[[b]][idx, ] <- bagged$cumulative[[b]]
    }
    attempt_rows[[fold]] <- bagged$attempts
    saveRDS(
      list(
        fold = fold,
        cumulative = bagged$cumulative,
        validation_index = idx,
        attempts = bagged$attempts
      ),
      file.path(
        output_dir,
        sprintf("mlogit_bagging_fold_%d.rds", fold)
      )
    )
  }
  stopifnot(all(vapply(oof_by_n, function(x) !anyNA(x), logical(1))))

  v11 <- 0.8 * baseline + 0.2 * xgb_single
  learning <- data.frame(
    n_bags = seq_len(n_bags),
    mlogit_logloss = vapply(
      oof_by_n,
      function(pred) log_loss_matrix(truth, pred),
      numeric(1)
    ),
    blend_single_xgb_logloss = vapply(
      oof_by_n,
      function(pred) {
        log_loss_matrix(truth, 0.8 * pred + 0.2 * xgb_single)
      },
      numeric(1)
    ),
    blend_bagged_xgb_logloss = vapply(
      oof_by_n,
      function(pred) {
        log_loss_matrix(truth, 0.8 * pred + 0.2 * xgb_bagged)
      },
      numeric(1)
    )
  )
  learning$mlogit_gain <-
    log_loss_matrix(truth, baseline) - learning$mlogit_logloss
  learning$blend_single_xgb_gain <-
    log_loss_matrix(truth, v11) - learning$blend_single_xgb_logloss
  learning$blend_bagged_xgb_gain <-
    log_loss_matrix(truth, v11) - learning$blend_bagged_xgb_logloss

  final_mlogit <- oof_by_n[[n_bags]]
  fixed_single <- 0.8 * final_mlogit + 0.2 * xgb_single
  fixed_both <- 0.8 * final_mlogit + 0.2 * xgb_bagged
  bootstrap <- rbind(
    bootstrap_comparison(
      truth, baseline, final_mlogit, train$Case,
      "bagged_mlogit_vs_single_mlogit"
    ),
    bootstrap_comparison(
      truth, v11, fixed_single, train$Case,
      "bagged_mlogit_blend_vs_v11"
    ),
    bootstrap_comparison(
      truth, v11, fixed_both, train$Case,
      "both_components_bagged_vs_v11"
    )
  )

  write_result_csv(
    learning,
    file.path(output_dir, "mlogit_bagging_cv.csv")
  )
  write_result_csv(
    do.call(rbind, attempt_rows),
    file.path(output_dir, "mlogit_bagging_attempts.csv")
  )
  write_result_csv(
    bootstrap,
    file.path(output_dir, "mlogit_bagging_bootstrap.csv")
  )
  saveRDS(
    list(
      n_bags = n_bags,
      oof_by_n = oof_by_n,
      final_mlogit = final_mlogit,
      fixed_single_xgb = fixed_single,
      fixed_both_bagged = fixed_both,
      learning = learning,
      bootstrap = bootstrap
    ),
    file.path(output_dir, "mlogit_bagging_oof.rds")
  )
  print(learning, digits = 7)
  print(bootstrap, digits = 7)
}
