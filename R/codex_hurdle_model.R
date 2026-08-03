# Dedicated hurdle (opt-out / conditional-bundle) model.
#
# Pre-registration:
#   codex_hurdle_model_preregister.md
#
# Smoke test (1 outer fold only):
#   Sys.setenv(HURDLE_SMOKE = "1")
#   source("R/codex_hurdle_model.R")
#   Sys.unsetenv("HURDLE_SMOKE")
#
# Full run:
#   Sys.unsetenv("HURDLE_SMOKE")
#   Sys.setenv(HURDLE_REPEATED = "auto")
#   source("R/codex_hurdle_model.R")

suppressPackageStartupMessages({
  library(glmnet)
  library(survival)
})

options(stringsAsFactors = FALSE)

experiment_id <- "hurdle_optout_bundle_v1"
output_dir <- file.path("data_processed", "codex_hurdle_model")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

smoke_mode <- identical(Sys.getenv("HURDLE_SMOKE", "0"), "1")
repeated_mode <- tolower(Sys.getenv("HURDLE_REPEATED", "auto"))
stopifnot(repeated_mode %in% c("auto", "never", "always"))

canonical_seed <- 4821L
additional_seeds <- c(1907L, 2719L, 6151L, 8293L, 104729L)
all_outer_seeds <- c(canonical_seed, additional_seeds)

canonical_v14_target <- 1.1435331472708
near_miss_lower_limit <- -0.00075
metric_tolerance <- 1e-8

bootstrap_replicates <- if (smoke_mode) {
  200L
} else {
  as.integer(Sys.getenv("HURDLE_N_BOOT", "100000"))
}

v14_dir <- file.path("data_processed", "codex_set_context_network")
blend_weight_grid <- seq(0, 0.40, by = 0.02)

attrs <- c(
  "CC", "GN", "NS", "BU", "FA", "LD", "BZ", "FC", "FP", "RP",
  "PP", "KA", "SC", "TS", "NV", "MA", "LB", "AF", "HU"
)
q_scaler_vars <- c(
  "incomea", "agea", "milesa", "nighta",
  "genderind", "Urbind", "educind"
)
difficulty_vars <- c(
  "n_attrs_varying", "price_min", "price_max",
  "price_spread", "price_mean", "price_cv"
)

input_files <- c(
  file.path("csv files", "train.csv"),
  file.path("data_processed", "oof_ensemble_v10.rds"),
  file.path(v14_dir, "canonical_result.rds")
)
missing_files <- input_files[!file.exists(input_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing required project file(s):\n  ",
    paste(missing_files, collapse = "\n  "),
    "\nRun this script from the kagglecomp repository root."
  )
}

## ---- generic helpers (verbatim pattern from codex_two_head_ensemble_v2.R /
## codex_set_context_network_v2.R, kept self-contained on purpose) ----

validate_probability <- function(prediction, n_rows = NULL) {
  prediction <- as.matrix(prediction)
  if (!is.null(n_rows)) {
    stopifnot(identical(dim(prediction), c(as.integer(n_rows), 4L)))
  } else {
    stopifnot(ncol(prediction) == 4L)
  }
  stopifnot(
    !anyNA(prediction),
    all(is.finite(prediction)),
    min(prediction) >= -1e-12
  )
  prediction <- pmax(prediction, 1e-15)
  prediction <- prediction / rowSums(prediction)
  stopifnot(max(abs(rowSums(prediction) - 1)) < 1e-10)
  prediction
}

log_loss_matrix_local <- function(truth, prediction) {
  prediction <- validate_probability(prediction, nrow(truth))
  -mean(rowSums(as.matrix(truth) * log(pmax(prediction, 1e-15))))
}

row_log_loss_local <- function(truth, prediction) {
  prediction <- validate_probability(prediction, nrow(truth))
  -rowSums(as.matrix(truth) * log(pmax(prediction, 1e-15)))
}

balanced_fold_map <- function(cases, folds, seed) {
  cases <- sort(unique(as.integer(cases)))
  set.seed(as.integer(seed))
  assignment <- sample(rep(seq_len(folds), length.out = length(cases)))
  names(assignment) <- as.character(cases)
  assignment
}

respondent_gain <- function(truth, baseline, candidate, case) {
  row_gain <- row_log_loss_local(truth, baseline) -
    row_log_loss_local(truth, candidate)
  output <- tapply(row_gain, case, mean)
  output[order(as.integer(names(output)))]
}

bootstrap_case_means <- function(
    case_gain, replicates, seed = canonical_seed, chunk_size = 1000L) {
  case_gain <- as.numeric(case_gain)
  stopifnot(length(case_gain) > 1L, all(is.finite(case_gain)), replicates > 0L)
  set.seed(as.integer(seed))
  output <- numeric(replicates)
  start <- 1L
  while (start <= replicates) {
    count <- min(chunk_size, replicates - start + 1L)
    index <- matrix(
      sample.int(length(case_gain), length(case_gain) * count, replace = TRUE),
      nrow = length(case_gain), ncol = count
    )
    output[start:(start + count - 1L)] <- colMeans(matrix(
      case_gain[index], nrow = length(case_gain), ncol = count
    ))
    start <- start + count
  }
  output
}

summarize_case_gain <- function(case_gain, replicates, seed) {
  bootstrap <- bootstrap_case_means(case_gain, replicates, seed)
  list(
    summary = data.frame(
      point_gain = mean(case_gain),
      bootstrap_mean = mean(bootstrap),
      bootstrap_sd = sd(bootstrap),
      lower_95 = unname(quantile(bootstrap, 0.025)),
      upper_95 = unname(quantile(bootstrap, 0.975)),
      lower_99 = unname(quantile(bootstrap, 0.005)),
      upper_99 = unname(quantile(bootstrap, 0.995)),
      win_rate = mean(bootstrap > 0),
      n_boot = length(bootstrap)
    ),
    bootstrap = bootstrap
  )
}

bootstrap_summary <- function(truth, baseline, candidate, case, replicates, seed) {
  case_gain <- respondent_gain(truth, baseline, candidate, case)
  summarized <- summarize_case_gain(case_gain, replicates, seed)
  list(
    summary = summarized$summary,
    respondent_gain = case_gain,
    bootstrap = summarized$bootstrap
  )
}

fit_scaler <- function(values) {
  values <- as.matrix(values)
  centre <- colMeans(values)
  scale <- apply(values, 2L, sd)
  scale[!is.finite(scale) | scale == 0] <- 1
  list(centre = centre, scale = scale)
}

standardize_matrix <- function(values, scaler) {
  values <- as.matrix(values)
  sweep(sweep(values, 2L, scaler$centre, "-"), 2L, scaler$scale, "/")
}

crossfit_blend <- function(truth, baseline, component, row_fold) {
  baseline <- validate_probability(baseline, nrow(truth))
  component <- validate_probability(component, nrow(truth))
  candidate <- matrix(NA_real_, nrow(truth), 4L)
  weight_rows <- list()
  for (fold in 1:5) {
    tuning_rows <- row_fold != fold
    validation_rows <- row_fold == fold
    losses <- vapply(blend_weight_grid, function(weight) {
      log_loss_matrix_local(
        truth[tuning_rows, , drop = FALSE],
        (1 - weight) * baseline[tuning_rows, , drop = FALSE] +
          weight * component[tuning_rows, , drop = FALSE]
      )
    }, numeric(1))
    best_index <- which.min(losses)
    best_weight <- blend_weight_grid[[best_index]]
    candidate[validation_rows, ] <-
      (1 - best_weight) * baseline[validation_rows, , drop = FALSE] +
      best_weight * component[validation_rows, , drop = FALSE]
    weight_rows[[fold]] <- data.frame(
      fold = fold, hurdle_weight = best_weight,
      tuning_logloss = losses[[best_index]]
    )
  }
  list(
    prediction = validate_probability(candidate, nrow(truth)),
    weights = do.call(rbind, weight_rows)
  )
}

## ---- task-difficulty features (model-free, design-only) ----

compute_task_difficulty <- function(wide) {
  varying <- matrix(FALSE, nrow(wide), length(attrs))
  for (index in seq_along(attrs)) {
    attribute <- attrs[[index]]
    columns <- as.matrix(wide[, paste0(attribute, 1:3), drop = FALSE])
    varying[, index] <- !(columns[, 1L] == columns[, 2L] &
      columns[, 2L] == columns[, 3L])
  }
  price <- as.matrix(wide[, paste0("Price", 1:3), drop = FALSE])
  price_min <- apply(price, 1L, min)
  price_max <- apply(price, 1L, max)
  price_mean <- rowMeans(price)
  data.frame(
    n_attrs_varying = rowSums(varying),
    price_min = price_min,
    price_max = price_max,
    price_spread = price_max - price_min,
    price_mean = price_mean,
    price_cv = (price_max - price_min) / price_mean
  )
}

## ---- q head: task-level binary opt-out model with a frozen v14 offset ----

build_q_design <- function(wide, difficulty, cont_scaler, diff_scaler) {
  z_cont <- standardize_matrix(wide[, q_scaler_vars, drop = FALSE], cont_scaler)
  colnames(z_cont) <- paste0("z_", q_scaler_vars)
  z_diff <- standardize_matrix(difficulty, diff_scaler)
  colnames(z_diff) <- paste0("z_", difficulty_vars)
  Task_c <- (as.numeric(wide$Task) - 10) / 9
  segment_d <- model.matrix(~factor(segmentind, levels = 1:6), data = wide)[, -1, drop = FALSE]
  region_d <- model.matrix(~factor(regionind, levels = 1:5), data = wide)[, -1, drop = FALSE]
  ppark_d <- model.matrix(~factor(pparkind, levels = 1:5), data = wide)[, -1, drop = FALSE]
  design <- cbind(Task_c = Task_c, z_diff, z_cont, segment_d, region_d, ppark_d)
  storage.mode(design) <- "double"
  design
}

fit_q_head <- function(train_wide, holdout_wide, q_v14_train, q_v14_holdout, inner_foldid) {
  difficulty_train <- compute_task_difficulty(train_wide)
  difficulty_holdout <- compute_task_difficulty(holdout_wide)
  cont_scaler <- fit_scaler(train_wide[, q_scaler_vars, drop = FALSE])
  diff_scaler <- fit_scaler(difficulty_train)

  x_train <- build_q_design(train_wide, difficulty_train, cont_scaler, diff_scaler)
  x_holdout <- build_q_design(holdout_wide, difficulty_holdout, cont_scaler, diff_scaler)
  stopifnot(identical(colnames(x_train), colnames(x_holdout)))

  y_train <- as.numeric(train_wide$Ch4)
  offset_train <- qlogis(pmin(pmax(q_v14_train, 1e-8), 1 - 1e-8))
  offset_holdout <- qlogis(pmin(pmax(q_v14_holdout, 1e-8), 1 - 1e-8))

  fitted <- cv.glmnet(
    x = x_train, y = y_train, family = "binomial",
    alpha = 0, offset = offset_train, foldid = inner_foldid,
    standardize = FALSE
  )
  prediction <- as.numeric(predict(
    fitted, newx = x_holdout, newoffset = offset_holdout,
    s = "lambda.min", type = "response"
  ))
  prediction <- pmin(pmax(prediction, 1e-8), 1 - 1e-8)
  list(prediction = prediction, lambda_min = fitted$lambda.min)
}

## ---- r head: 3-way conditional logit on inside-only rows ----

to_long <- function(wide) {
  pattern <- paste0("^(", paste(c(attrs, "Price", "Ch"), collapse = "|"), ")([1-4])$")
  fixed_names <- names(wide)[!grepl(pattern, names(wide))]
  parts <- lapply(1:4, function(alternative) {
    piece <- wide[, fixed_names, drop = FALSE]
    for (variable in c(attrs, "Price", "Ch")) {
      piece[[variable]] <- wide[[paste0(variable, alternative)]]
    }
    piece$alt <- alternative
    piece$chosen <- as.integer(piece$Ch == 1)
    piece$chid <- paste(wide$Case, wide$Task, sep = "_")[seq_len(nrow(piece))]
    piece
  })
  output <- do.call(rbind, parts)
  output <- output[order(output$No, output$alt), , drop = FALSE]
  rownames(output) <- NULL
  output
}

r_scaler_vars <- c("incomea", "agea", "milesa", "nighta")

make_r_features <- function(long, ctr, scl) {
  long <- as.data.frame(long)
  # Structural fix caught by the smoke test's qr()-rank diagnostic, not a
  # post-hoc data peek: this partial-profile design fixes each alternative at
  # exactly 9 of 19 active (non-reference-level) attributes (cleaning_log.md
  # finding #2). With the opt-out alternative removed, every remaining row has
  # this same total, so the 19 attributes' one-hot dummy columns sum to an
  # exact constant (9) for every observation -- a genuine rank-1-deficient
  # direction in the design (the conditional-logit analogue of the original
  # Price-factor collinearity, which was only avoided before because the
  # opt-out alternative's all-zero profile broke the "always 9" pattern).
  # Any single column from the 19-attribute block can be dropped to fix this
  # without changing the fitted choice probabilities (the likelihood is exactly
  # flat along that one direction, so any complementary linear constraint
  # recovers the identical maximum), exactly as when the original fix dropped
  # one arbitrary price-level dummy. Folding HU's level 2 into its reference
  # level (0) is one such harmless, arbitrary choice (any of the 19 attributes'
  # levels would work identically); HU's own level-2 effect becomes
  # inestimable on its own and is absorbed into the shared baseline instead.
  long$HU <- ifelse(long$HU == 2L, 0L, long$HU)
  z <- sweep(sweep(long[, r_scaler_vars, drop = FALSE], 2, ctr, "-"), 2, scl, "/")
  long$inside <- 1L
  long$d2 <- as.integer(long$alt == 2L)
  long$d3 <- as.integer(long$alt == 3L)
  long$Price_num <- as.numeric(long$Price)
  for (level in 2:12) long[[paste0("Pr_lvl", level)]] <- as.integer(long$Price_num == level)
  long$Task_c <- (as.numeric(long$Task) - 10) / 9
  long$P_income <- long$Price_num * z$incomea
  long$P_age <- long$Price_num * z$agea
  long$P_miles <- long$Price_num * z$milesa
  long$P_night <- long$Price_num * z$nighta
  long$In_income <- long$inside * z$incomea
  long$In_age <- long$inside * z$agea
  long$In_miles <- long$inside * z$milesa
  long$In_night <- long$inside * z$nighta
  long$In_gender <- long$inside * long$genderind
  long$In_urb <- long$inside * long$Urbind
  long$In_educ <- long$inside * long$educind
  for (segment in 2:6) {
    long[[paste0("P_seg", segment)]] <- long$Price_num * (long$segmentind == segment)
    long[[paste0("In_seg", segment)]] <- long$inside * (long$segmentind == segment)
  }
  long$P_task <- long$Price_num * long$Task_c
  long$In_task <- long$inside * long$Task_c
  for (level in 2:5) {
    long[[paste0("P_region", level)]] <- long$Price_num * (long$regionind == level)
    long[[paste0("In_region", level)]] <- long$inside * (long$regionind == level)
    long[[paste0("P_ppark", level)]] <- long$Price_num * (long$pparkind == level)
    long[[paste0("In_ppark", level)]] <- long$inside * (long$pparkind == level)
  }
  price_min_by_chid <- tapply(long$Price_num, long$chid, min)
  price_max_by_chid <- tapply(long$Price_num, long$chid, max)
  long$price_min <- unname(price_min_by_chid[long$chid])
  long$price_max <- unname(price_max_by_chid[long$chid])
  long$is_cheapest <- as.integer(long$Price_num == long$price_min)
  long$is_dearest <- as.integer(long$Price_num == long$price_max)
  long$price_gap_min <- long$Price_num - long$price_min
  long$price_gap_max <- long$price_max - long$Price_num
  long
}

# NOTE: unlike m8trpg's full 4-way spec, every "In_*" (inside x covariate)
# term is dropped here. Once alt 4 (opt-out) is removed, `inside` is
# identically 1 for every remaining alternative in every task, so an
# `inside x covariate` product no longer varies across alternatives within a
# choice set -- conditional logit only identifies effects through
# within-task utility differences, so a term constant across a task's
# alternatives has no likelihood content and produces an unidentified
# (singular-Hessian) coefficient. This is exactly the mechanism the opt-out
# margin's heterogeneity is supposed to work through (and is now handled by
# the q head instead); only the alternative-varying `Price x covariate` and
# design terms remain meaningful for the conditional 3-way choice.
int_terms <- c(
  "P_income", "P_age", "P_miles", "P_night",
  paste0("P_seg", 2:6),
  "P_task",
  paste0("P_region", 2:5),
  paste0("P_ppark", 2:5),
  "is_cheapest", "is_dearest", "price_gap_min", "price_gap_max"
)

## r is estimated via glmnet's stratified-Cox equivalence to the conditional
## logit likelihood (family="cox", one stratum per choice task), the same
## already-validated technique this project used for its regularized
## conditional-logit interaction search (R/codex_glmnet_cox_ensemble.R).
## Plain unpenalized mlogit MLE hit an exactly-singular Hessian on this
## smaller, restricted (inside-only, single-outer-fold) subsample even after
## fixing every rank deficiency a direct qr() check could find -- consistent
## with quasi-complete separation in a sparser interaction cell, not a design
## bug. Ridge-penalized Cox estimation is the natural fix: it is numerically
## robust to exactly this failure mode by construction, and it directly
## delivers the "heavy regularization" the pre-registration calls for,
## instead of patching the MLE with ad hoc term removal.
build_r_matrix <- function(features, attr_max) {
  parts <- list()
  for (attribute in attrs) {
    max_level <- attr_max[[attribute]]
    if (max_level >= 1L) {
      for (level in seq_len(max_level)) {
        parts[[paste0(attribute, "_", level)]] <- as.numeric(features[[attribute]] == level)
      }
    }
  }
  for (level in 2:12) {
    parts[[paste0("Pr_lvl", level)]] <- as.numeric(features[[paste0("Pr_lvl", level)]])
  }
  parts$d2 <- as.numeric(features$d2)
  parts$d3 <- as.numeric(features$d3)
  for (term in int_terms) parts[[term]] <- as.numeric(features[[term]])
  design <- do.call(cbind, parts)
  storage.mode(design) <- "double"
  design
}

softmax3 <- function(eta) {
  z <- matrix(as.numeric(eta), ncol = 3L, byrow = TRUE)
  z <- z - apply(z, 1L, max)
  ez <- exp(z)
  ez / rowSums(ez)
}

fit_r_head <- function(train_wide, holdout_wide, fold_map, inner_labels) {
  inside_train <- train_wide[train_wide$Ch4 == 0L, , drop = FALSE]
  stopifnot(nrow(inside_train) > 0L)
  train_long <- to_long(inside_train)
  train_long <- train_long[train_long$alt != 4L, , drop = FALSE]
  stopifnot(all(table(train_long$chid) == 3L))

  holdout_long <- to_long(holdout_wide)
  holdout_long <- holdout_long[holdout_long$alt != 4L, , drop = FALSE]
  stopifnot(all(table(holdout_long$chid) == 3L))

  ctr <- sapply(train_long[r_scaler_vars], mean, na.rm = TRUE)
  scl <- sapply(train_long[r_scaler_vars], sd, na.rm = TRUE)
  scl[scl == 0] <- 1

  train_feat <- make_r_features(train_long, ctr, scl)
  holdout_feat <- make_r_features(holdout_long, ctr, scl)
  attr_max <- vapply(attrs, function(attribute) {
    max(c(train_feat[[attribute]], holdout_feat[[attribute]]))
  }, numeric(1))

  x_train <- build_r_matrix(train_feat, attr_max)
  x_holdout <- build_r_matrix(holdout_feat, attr_max)
  stopifnot(identical(colnames(x_train), colnames(x_holdout)))

  y_train <- stratifySurv(
    survival::Surv(rep(1, nrow(train_feat)), as.integer(train_feat$chosen)),
    as.factor(train_feat$No)
  )
  foldid <- match(unname(fold_map[as.character(train_feat$Case)]), inner_labels)
  stopifnot(!anyNA(foldid), length(unique(foldid)) == length(inner_labels))

  fitted <- cv.glmnet(
    x = x_train, y = y_train, family = "cox", alpha = 0,
    standardize = FALSE, foldid = foldid, type.measure = "deviance",
    grouped = FALSE, cox.ties = "breslow"
  )
  eta_holdout <- as.numeric(predict(
    fitted$glmnet.fit, newx = x_holdout, s = fitted$lambda.min, type = "link"
  ))
  prediction_by_task <- softmax3(eta_holdout)
  holdout_task_order <- unique(holdout_feat$No)
  prediction <- prediction_by_task[match(holdout_wide$No, holdout_task_order), , drop = FALSE]
  stopifnot(!anyNA(prediction), nrow(prediction) == nrow(holdout_wide))
  colnames(prediction) <- NULL
  list(prediction = prediction / rowSums(prediction), lambda_min = fitted$lambda.min)
}

## ---- outer fold loop: fit q + r on training folds, predict on the held-out fold ----

run_nested_outer <- function(train, fold_map, v14_oof_q, outer_fold) {
  row_fold <- unname(fold_map[as.character(train$Case)])
  outer_rows <- which(row_fold == outer_fold)
  train_rows <- which(row_fold != outer_fold)
  stopifnot(
    length(outer_rows) > 0L, length(train_rows) > 0L,
    length(intersect(
      unique(train$Case[outer_rows]), unique(train$Case[train_rows])
    )) == 0L
  )

  train_wide <- train[train_rows, , drop = FALSE]
  holdout_wide <- train[outer_rows, , drop = FALSE]

  inner_labels <- setdiff(1:5, outer_fold)
  inner_foldid <- match(row_fold[train_rows], inner_labels)
  stopifnot(!anyNA(inner_foldid), length(unique(inner_foldid)) == 4L)

  q_fit <- fit_q_head(
    train_wide, holdout_wide,
    q_v14_train = v14_oof_q[train_rows],
    q_v14_holdout = v14_oof_q[outer_rows],
    inner_foldid = inner_foldid
  )
  r_fit <- fit_r_head(train_wide, holdout_wide, fold_map, inner_labels)

  q_hat <- q_fit$prediction
  prediction <- cbind((1 - q_hat) * r_fit$prediction, q_hat)
  prediction <- validate_probability(prediction, length(outer_rows))

  list(
    outer_rows = outer_rows,
    prediction = prediction,
    q_lambda_min = q_fit$lambda_min,
    r_lambda_min = r_fit$lambda_min
  )
}

run_hurdle_oof <- function(train, fold_map, v14_oof_q) {
  row_fold <- unname(fold_map[as.character(train$Case)])
  prediction <- matrix(NA_real_, nrow(train), 4L)
  q_lambdas <- numeric(5L)
  r_lambdas <- numeric(5L)
  for (outer_fold in 1:5) {
    cat(sprintf("  hurdle outer fold %d/5\n", outer_fold))
    result <- run_nested_outer(train, fold_map, v14_oof_q, outer_fold)
    prediction[result$outer_rows, ] <- result$prediction
    q_lambdas[[outer_fold]] <- result$q_lambda_min
    r_lambdas[[outer_fold]] <- result$r_lambda_min
    flush.console()
  }
  list(
    prediction = validate_probability(prediction, nrow(train)),
    row_fold = row_fold,
    q_lambdas = q_lambdas,
    r_lambdas = r_lambdas
  )
}

## ---- v14 artifact loaders (same files codex_two_head_ensemble_v2.R verified) ----

load_v14_artifact <- function(seed) {
  path <- if (seed == canonical_seed) {
    file.path(v14_dir, "canonical_result.rds")
  } else {
    file.path(v14_dir, sprintf("repeat_result_%d.rds", seed))
  }
  if (!file.exists(path)) stop("Missing verified v14 artifact: ", path)
  object <- readRDS(path)
  required <- c("experiment_id", "fold_map", "candidate_prediction")
  stopifnot(all(required %in% names(object)))
  list(
    fold_map = object$fold_map,
    candidate_prediction = validate_probability(object$candidate_prediction)
  )
}

## ---- main experiment ----

run_smoke_test <- function(train, truth) {
  v14 <- load_v14_artifact(canonical_seed)
  fold_map <- v14$fold_map
  result <- run_nested_outer(train, fold_map, v14$candidate_prediction[, 4L], outer_fold = 1L)
  baseline_loss <- log_loss_matrix_local(
    truth[result$outer_rows, , drop = FALSE],
    v14$candidate_prediction[result$outer_rows, , drop = FALSE]
  )
  candidate_loss <- log_loss_matrix_local(truth[result$outer_rows, , drop = FALSE], result$prediction)
  cat(sprintf(
    "\nSmoke test: fold-1 v14 loss %.6f, fold-1 hurdle loss %.6f, q lambda.min %.6g, r lambda.min %.6g\n",
    baseline_loss, candidate_loss, result$q_lambda_min, result$r_lambda_min
  ))
  saveRDS(
    list(experiment_id = experiment_id, result = result, baseline_loss = baseline_loss, candidate_loss = candidate_loss),
    file.path(output_dir, "smoke_result.rds")
  )
  invisible(result)
}

run_experiment <- function() {
  train <- read.csv(file.path("csv files", "train.csv"))
  train <- train[order(train$No), , drop = FALSE]
  rownames(train) <- NULL
  truth <- as.matrix(train[, paste0("Ch", 1:4), drop = FALSE])
  stopifnot(
    nrow(train) == 21565L,
    length(unique(train$Case)) == 1135L,
    all(table(train$Case) == 19L),
    all(rowSums(truth) == 1L)
  )

  cat("\nRunning", experiment_id, "\n")

  if (smoke_mode) {
    return(run_smoke_test(train, truth))
  }

  v14_registry <- lapply(all_outer_seeds, load_v14_artifact)
  names(v14_registry) <- as.character(all_outer_seeds)
  canonical_v14 <- v14_registry[[as.character(canonical_seed)]]

  canonical_baseline_loss <- log_loss_matrix_local(truth, canonical_v14$candidate_prediction)
  stopifnot(abs(canonical_baseline_loss - canonical_v14_target) <= metric_tolerance)

  hurdle_oof <- run_hurdle_oof(train, canonical_v14$fold_map, canonical_v14$candidate_prediction[, 4L])
  blended <- crossfit_blend(truth, canonical_v14$candidate_prediction, hurdle_oof$prediction, hurdle_oof$row_fold)
  candidate_loss <- log_loss_matrix_local(truth, blended$prediction)
  hurdle_component_loss <- log_loss_matrix_local(truth, hurdle_oof$prediction)

  canonical_bootstrap <- bootstrap_summary(
    truth, canonical_v14$candidate_prediction, blended$prediction,
    train$Case, bootstrap_replicates, canonical_seed
  )
  canonical_summary <- cbind(
    data.frame(
      experiment = experiment_id, stage = "canonical",
      baseline_logloss = canonical_baseline_loss,
      hurdle_component_logloss = hurdle_component_loss,
      candidate_logloss = candidate_loss
    ),
    canonical_bootstrap$summary
  )
  canonical_summary$canonical_pass <-
    canonical_summary$point_gain > 0 & canonical_summary$lower_95 > 0
  canonical_summary$near_miss <-
    canonical_summary$point_gain > 0 &
    canonical_summary$lower_95 <= 0 &
    canonical_summary$lower_95 >= near_miss_lower_limit

  write.csv(canonical_summary, file.path(output_dir, "canonical_summary.csv"), row.names = FALSE)
  write.csv(blended$weights, file.path(output_dir, "canonical_weights.csv"), row.names = FALSE)
  saveRDS(
    list(
      experiment_id = experiment_id, summary = canonical_summary,
      baseline_prediction = canonical_v14$candidate_prediction,
      hurdle_prediction = hurdle_oof$prediction,
      candidate_prediction = blended$prediction,
      fold_map = canonical_v14$fold_map, weights = blended$weights,
      respondent_gain = canonical_bootstrap$respondent_gain,
      bootstrap = canonical_bootstrap$bootstrap,
      q_lambdas = hurdle_oof$q_lambdas
    ),
    file.path(output_dir, "canonical_result.rds")
  )
  cat("\nCanonical result:\n")
  print(canonical_summary, digits = 9)

  run_repeated <- repeated_mode == "always" ||
    (repeated_mode == "auto" &&
      (isTRUE(canonical_summary$canonical_pass) || isTRUE(canonical_summary$near_miss)))

  if (!run_repeated) {
    reason <- if (repeated_mode == "never") {
      "Repeated CV disabled by HURDLE_REPEATED=never."
    } else {
      "Repeated CV not triggered: canonical result was neither a pass nor the pre-registered near miss."
    }
    verdict <- paste(
      "REJECT hurdle model; retain exact v14.", reason,
      sprintf(
        "Canonical gain %.9f; 95%% CI [%.9f, %.9f].",
        canonical_summary$point_gain, canonical_summary$lower_95, canonical_summary$upper_95
      )
    )
    writeLines(verdict, file.path(output_dir, "verdict.txt"))
    cat("\n", verdict, "\n", sep = "")
    return(invisible(list(canonical = canonical_summary, repeated = NULL, verdict = verdict)))
  }

  cat("\nRepeated-CV escalation activated (", repeated_mode, ").\n", sep = "")
  case_levels <- sort(unique(train$Case))
  seed_names <- as.character(all_outer_seeds)
  case_gain_matrix <- matrix(
    NA_real_, length(case_levels), length(seed_names),
    dimnames = list(as.character(case_levels), seed_names)
  )
  case_gain_matrix[, as.character(canonical_seed)] <- canonical_bootstrap$respondent_gain
  repeat_rows <- list(`4821` = data.frame(
    seed = canonical_seed, baseline_logloss = canonical_baseline_loss,
    candidate_logloss = candidate_loss, gain = canonical_baseline_loss - candidate_loss
  ))

  for (seed in additional_seeds) {
    seed_name <- as.character(seed)
    cat(sprintf("\n######## repeated CV seed %d ########\n", seed))
    repeat_v14 <- v14_registry[[seed_name]]
    repeat_hurdle <- run_hurdle_oof(train, repeat_v14$fold_map, repeat_v14$candidate_prediction[, 4L])
    repeat_blend <- crossfit_blend(
      truth, repeat_v14$candidate_prediction, repeat_hurdle$prediction, repeat_hurdle$row_fold
    )
    baseline_loss <- log_loss_matrix_local(truth, repeat_v14$candidate_prediction)
    seed_candidate_loss <- log_loss_matrix_local(truth, repeat_blend$prediction)
    case_gain <- respondent_gain(truth, repeat_v14$candidate_prediction, repeat_blend$prediction, train$Case)
    case_gain_matrix[, seed_name] <- case_gain
    repeat_rows[[seed_name]] <- data.frame(
      seed = seed, baseline_logloss = baseline_loss,
      candidate_logloss = seed_candidate_loss, gain = baseline_loss - seed_candidate_loss
    )
    saveRDS(
      list(
        experiment_id = experiment_id, seed = seed,
        baseline_prediction = repeat_v14$candidate_prediction,
        hurdle_prediction = repeat_hurdle$prediction,
        candidate_prediction = repeat_blend$prediction,
        fold_map = repeat_v14$fold_map, weights = repeat_blend$weights,
        respondent_gain = case_gain
      ),
      file.path(output_dir, sprintf("repeat_result_%d.rds", seed))
    )
  }

  repeated_by_seed <- do.call(rbind, repeat_rows)
  average_case_gain <- rowMeans(case_gain_matrix)
  pooled <- summarize_case_gain(average_case_gain, bootstrap_replicates, canonical_seed)
  repeated_summary <- cbind(
    data.frame(experiment = experiment_id, stage = "repeated_cv"),
    pooled$summary,
    data.frame(
      positive_repeats = sum(repeated_by_seed$gain > 0),
      n_repeats = nrow(repeated_by_seed)
    )
  )

  ## Test-like-respondent diagnostic, same adversarial-validation classifier
  ## pattern as codex_two_head_ensemble_v2.R's fit_test_propensity().
  test <- read.csv(file.path("csv files", "test.csv"))
  domain_covariates <- c(
    "segmentind", "yearind", "milesind", "milesa", "nightind", "nighta",
    "pparkind", "genderind", "ageind", "agea", "educind", "regionind",
    "Urbind", "incomeind", "incomea"
  )
  train_resp <- train[!duplicated(train$Case), c("Case", domain_covariates), drop = FALSE]
  test_resp <- test[!duplicated(test$Case), c("Case", domain_covariates), drop = FALSE]
  train_domain <- train_resp[, domain_covariates, drop = FALSE]
  test_domain <- test_resp[, domain_covariates, drop = FALSE]
  train_domain$is_test <- 0L
  test_domain$is_test <- 1L
  domain <- rbind(train_domain, test_domain)
  classifier <- glm(is_test ~ ., data = domain, family = binomial())
  propensity <- predict(classifier, newdata = train_resp[, domain_covariates, drop = FALSE], type = "response")
  propensity <- pmin(pmax(as.numeric(propensity), 1e-8), 1 - 1e-8)
  names(propensity) <- as.character(train_resp$Case)
  propensity <- propensity[names(average_case_gain)]
  ordering <- order(propensity, decreasing = TRUE)
  top_30_index <- ordering[seq_len(ceiling(0.30 * length(ordering)))]
  top_30_summary <- summarize_case_gain(average_case_gain[top_30_index], bootstrap_replicates, seed = 16601L)$summary

  repeated_summary$test_like_30_point_gain <- top_30_summary$point_gain[[1L]]
  repeated_summary$test_like_30_lower_95 <- top_30_summary$lower_95[[1L]]
  repeated_summary$promote <-
    repeated_summary$point_gain > 0 &
    repeated_summary$lower_95 > 0 &
    repeated_summary$positive_repeats >= 5L &
    repeated_summary$test_like_30_point_gain >= 0 &
    repeated_summary$test_like_30_lower_95 >= -0.001

  write.csv(repeated_by_seed, file.path(output_dir, "repeated_cv_by_seed.csv"), row.names = FALSE)
  write.csv(repeated_summary, file.path(output_dir, "repeated_cv_summary.csv"), row.names = FALSE)
  saveRDS(
    list(
      experiment_id = experiment_id, repeated_by_seed = repeated_by_seed,
      repeated_summary = repeated_summary, case_gain_matrix = case_gain_matrix,
      average_case_gain = average_case_gain, pooled_bootstrap = pooled$bootstrap,
      test_propensity = propensity
    ),
    file.path(output_dir, "repeated_cv_result.rds")
  )

  verdict <- if (isTRUE(repeated_summary$promote)) {
    "PROMOTE hurdle model for a separate full-data build audit. Do not submit until the candidate is reproduced and checked."
  } else {
    "REJECT hurdle model; retain the exact set-context v14 submission."
  }
  writeLines(verdict, file.path(output_dir, "verdict.txt"))
  cat("\nRepeated-CV result:\n")
  print(repeated_summary, digits = 9)
  cat("\n", verdict, "\n", sep = "")
  invisible(list(canonical = canonical_summary, repeated = repeated_summary, verdict = verdict))
}

run_experiment()
