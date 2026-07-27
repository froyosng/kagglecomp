source("R/codex_shift_common.R")

stage <- Sys.getenv("CODEX_STAGE", "screen")
dir.create("data_processed/codex_shift", recursive = TRUE,
           showWarnings = FALSE)

train <- read.csv("csv files/train.csv")
truth <- as.matrix(train[, paste0("Ch", 1:4)])

prepare_saved_long <- function(d) {
  d$chid <- paste(d$Case, d$Task, sep = "_")
  d$d2 <- as.integer(d$alt == 2L)
  d$d3 <- as.integer(d$alt == 3L)
  d
}

if (stage == "screen") {
  split <- readRDS("data_processed/train_val_split.rds")
  tr_long <- prepare_saved_long(split$train_long_tr)
  va_long <- prepare_saved_long(split$train_long_val)
  va_wide <- split$train_wide_val
  va_wide <- va_wide[order(va_wide$No), , drop = FALSE]
  va_truth <- as.matrix(va_wide[, paste0("Ch", 1:4)])

  baseline_fit <- fit_predict_m8trpg(
    tr_long, va_long,
    case_weights = NULL,
    attr_rank_scope = "none"
  )
  baseline_pred <- baseline_fit$pred[
    match(va_wide$No, baseline_fit$no), , drop = FALSE
  ]

  scopes <- c("inside", "all")
  results <- list()
  predictions <- list()
  for (scope in scopes) {
    fitted <- tryCatch(
      fit_predict_m8trpg(
        tr_long, va_long,
        case_weights = NULL,
        attr_rank_scope = scope
      ),
      error = function(e) e
    )
    if (inherits(fitted, "error")) {
      results[[scope]] <- data.frame(
        scope = scope,
        status = "fit_failed",
        logloss = NA_real_,
        delta_vs_base = NA_real_,
        detail = conditionMessage(fitted)
      )
      cat(scope, "fit failed:", conditionMessage(fitted), "\n")
    } else {
      idx <- match(va_wide$No, fitted$no)
      pred <- fitted$pred[idx, , drop = FALSE]
      predictions[[scope]] <- pred
      ll <- log_loss_matrix(va_truth, pred)
      results[[scope]] <- data.frame(
        scope = scope,
        status = "ok",
        logloss = ll,
        delta_vs_base = ll - log_loss_matrix(va_truth, baseline_pred),
        detail = "38 min/max indicators; ties retain multiple flags"
      )
      cat(sprintf(
        "%s ranks: ll %.6f; delta vs base %+.6f\n",
        scope, ll, results[[scope]]$delta_vs_base
      ))
    }
  }

  result <- do.call(rbind, results)
  write_result_csv(
    result, "data_processed/codex_shift/attribute_rank_screen.csv"
  )
  saveRDS(
    list(result = result, predictions = predictions,
         baseline_pred = baseline_pred, va_no = va_wide$No),
    "data_processed/codex_shift/attribute_rank_screen.rds"
  )
  print(result)
}

if (stage == "cv") {
  screen <- read.csv(
    "data_processed/codex_shift/attribute_rank_screen.csv"
  )
  viable <- screen[
    screen$status == "ok" & screen$delta_vs_base < 0,
    , drop = FALSE
  ]
  if (nrow(viable) == 0) {
    cat("No attribute-rank candidate beat the single-split baseline; CV skipped.\n")
    quit(save = "no", status = 0)
  }
  scope <- viable$scope[which.min(viable$logloss)]
  cat("Confirming scope:", scope, "\n")

  train_long <- reshape_choice_long(train)
  fold_map <- canonical_fold_map()
  saved <- readRDS("data_processed/oof_ensemble_v10.rds")
  baseline <- saved$oof_mlogit
  rank_oof <- matrix(NA_real_, nrow(train), 4)

  for (k in 1:5) {
    val_cases <- as.integer(names(fold_map)[fold_map == k])
    tr_long <- train_long[
      !(train_long$Case %in% val_cases), , drop = FALSE
    ]
    va_long <- train_long[
      train_long$Case %in% val_cases, , drop = FALSE
    ]
    fitted <- fit_predict_m8trpg(
      tr_long, va_long,
      case_weights = NULL,
      attr_rank_scope = scope
    )
    idx <- match(fitted$no, train$No)
    rank_oof[idx, ] <- fitted$pred
    cat(sprintf(
      "fold %d done: base %.6f; rank %.6f\n",
      k,
      log_loss_matrix(truth[idx, ], baseline[idx, ]),
      log_loss_matrix(truth[idx, ], rank_oof[idx, ])
    ))
  }
  stopifnot(!anyNA(rank_oof))

  row_loss <- function(pred) {
    pred <- pmin(pmax(pred / rowSums(pred), 1e-15), 1 - 1e-15)
    -rowSums(truth * log(pred))
  }
  respondent_delta <- tapply(
    row_loss(baseline) - row_loss(rank_oof),
    train$Case, mean
  )
  set.seed(4821)
  boot <- replicate(
    1000,
    mean(sample(
      respondent_delta, length(respondent_delta), replace = TRUE
    ))
  )
  result <- data.frame(
    scope = scope,
    baseline_logloss = log_loss_matrix(truth, baseline),
    rank_logloss = log_loss_matrix(truth, rank_oof),
    point_gain = mean(respondent_delta),
    bootstrap_mean = mean(boot),
    bootstrap_sd = sd(boot),
    lower_95 = quantile(boot, 0.025),
    upper_95 = quantile(boot, 0.975),
    win_rate = mean(boot > 0)
  )
  write_result_csv(
    result, "data_processed/codex_shift/attribute_rank_cv.csv"
  )
  saveRDS(
    list(scope = scope, oof = rank_oof, result = result),
    "data_processed/codex_shift/attribute_rank_oof.rds"
  )
  print(result)
}
