## Data-driven interaction discovery from multiclass xgboost SHAP interaction
## values, followed by conditional-logit screening/CV. Discovery uses only the
## canonical screen-training respondents, so the held-out screen labels do not
## nominate their own candidate interactions.

suppressPackageStartupMessages({
  library(xgboost)
  library(mlogit)
  library(dfidx)
})
source("R/codex_shift_common.R")

stage <- Sys.getenv("CODEX_STAGE", "discover")
stopifnot(stage %in% c("discover", "screen", "cv"))

output_dir <- "data_processed/codex_final_round"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

top_n <- as.integer(Sys.getenv("CODEX_TOP_PAIRS", "8"))
stopifnot(top_n >= 5L, top_n <= 10L)

raw_to_canonical <- c(
  yearind = "year",
  milesind = "miles",
  milesa = "miles",
  nightind = "night",
  nighta = "night",
  genderind = "gender",
  ageind = "age",
  agea = "age",
  educind = "educ",
  Urbind = "urb",
  incomeind = "income",
  incomea = "income"
)

canonical_source <- c(
  year = "yearind",
  miles = "milesa",
  night = "nighta",
  gender = "genderind",
  age = "agea",
  educ = "educind",
  urb = "Urbind",
  income = "incomea"
)

pair_key <- function(a, b) {
  paste(sort(c(a, b)), collapse = "__")
}

# These continuous-product pairs were already explicitly tested in the prior
# triple-product round and must not be rediscovered post hoc.
already_tested <- c(
  pair_key("income", "age"),
  pair_key("income", "miles"),
  pair_key("income", "night")
)

xgb_feature_names <- function() {
  attr_wide <- paste0(
    rep(attrs, each = 4),
    rep(1:4, times = length(attrs))
  )
  c(attr_wide, paste0("Price", 1:4), xgb_covariates)
}

xgb_feature_matrix_named <- function(df) {
  cols <- xgb_feature_names()
  x <- as.matrix(df[, cols, drop = FALSE])
  storage.mode(x) <- "double"
  colnames(x) <- cols
  x
}

discover_shap_pairs <- function() {
  logged <- read.csv("submissions_log.csv", stringsAsFactors = FALSE)
  stopifnot(
    "triple_price_income_miles" %in% logged$model_name,
    "xgboost_multiclass_v1" %in% logged$model_name
  )

  split_data <- readRDS("data_processed/train_val_split.rds")
  train_wide <- split_data$train_wide_tr
  train_wide$Choice <- max.col(
    train_wide[, paste0("Ch", 1:4)]
  )
  x_train <- xgb_feature_matrix_named(train_wide)
  dtrain <- xgb.DMatrix(
    x_train, label = train_wide$Choice - 1L
  )
  params <- list(
    objective = "multi:softprob",
    num_class = 4,
    eval_metric = "mlogloss",
    eta = 0.1,
    max_depth = 4,
    subsample = 0.8,
    colsample_bytree = 0.8,
    seed = 4821,
    nthread = 1
  )
  model <- xgb.train(
    params = params, data = dtrain,
    nrounds = 73, verbose = 0
  )

  # Sample whole screen-training respondents, then process their 19 tasks in
  # small batches. This preserves respondent coverage without materializing
  # the full n x class x feature x feature tensor at once.
  respondents <- sort(unique(train_wide$Case))
  set.seed(4821)
  shap_cases <- sample(
    respondents, min(100L, length(respondents)), replace = FALSE
  )
  shap_rows <- which(train_wide$Case %in% shap_cases)
  batch_size <- 25L
  p <- ncol(x_train)
  interaction_sum <- matrix(
    0, nrow = p + 1L, ncol = p + 1L
  )
  interaction_count <- 0L

  batches <- split(
    shap_rows,
    ceiling(seq_along(shap_rows) / batch_size)
  )
  for (b in seq_along(batches)) {
    idx <- batches[[b]]
    values <- predict(
      model,
      xgb.DMatrix(x_train[idx, , drop = FALSE]),
      predinteraction = TRUE,
      strict_shape = TRUE
    )
    stopifnot(
      identical(
        dim(values),
        c(length(idx), 4L, p + 1L, p + 1L)
      )
    )
    interaction_sum <- interaction_sum +
      apply(abs(values), c(3, 4), sum)
    interaction_count <- interaction_count + length(idx) * 4L
    if (b %% 10L == 0L || b == length(batches)) {
      cat(sprintf(
        "SHAP batch %d/%d complete\n", b, length(batches)
      ))
      flush.console()
    }
  }
  mean_abs <- interaction_sum / interaction_count
  mean_abs <- mean_abs[seq_len(p), seq_len(p), drop = FALSE]
  dimnames(mean_abs) <- list(colnames(x_train), colnames(x_train))

  pair_index <- which(upper.tri(mean_abs), arr.ind = TRUE)
  raw <- data.frame(
    feature1 = rownames(mean_abs)[pair_index[, 1]],
    feature2 = colnames(mean_abs)[pair_index[, 2]],
    mean_abs_shap_interaction = mean_abs[pair_index],
    stringsAsFactors = FALSE
  )
  raw <- raw[
    order(raw$mean_abs_shap_interaction, decreasing = TRUE),
    ,
    drop = FALSE
  ]
  raw$raw_rank <- seq_len(nrow(raw))

  eligible <- raw[
    raw$feature1 %in% names(raw_to_canonical) &
      raw$feature2 %in% names(raw_to_canonical),
    ,
    drop = FALSE
  ]
  eligible$covariate1 <- unname(
    raw_to_canonical[eligible$feature1]
  )
  eligible$covariate2 <- unname(
    raw_to_canonical[eligible$feature2]
  )
  eligible <- eligible[
    eligible$covariate1 != eligible$covariate2,
    ,
    drop = FALSE
  ]
  eligible$pair <- mapply(
    pair_key, eligible$covariate1, eligible$covariate2,
    USE.NAMES = FALSE
  )
  eligible <- eligible[
    !(eligible$pair %in% already_tested),
    ,
    drop = FALSE
  ]

  # A concept can have both binned and fine-grained raw encodings. Keep the
  # strongest raw representation for each conceptual pair so concepts with
  # duplicate encodings do not win merely by having more columns.
  best_row <- !duplicated(
    eligible$pair[
      order(
        eligible$mean_abs_shap_interaction,
        decreasing = TRUE
      )
    ]
  )
  eligible_ordered <- eligible[
    order(
      eligible$mean_abs_shap_interaction,
      decreasing = TRUE
    ),
    ,
    drop = FALSE
  ]
  conceptual <- eligible_ordered[best_row, , drop = FALSE]
  conceptual$shap_rank <- seq_len(nrow(conceptual))
  selected <- head(conceptual, top_n)

  write_result_csv(
    raw,
    file.path(output_dir, "shap_raw_pair_ranking.csv")
  )
  write_result_csv(
    conceptual,
    file.path(output_dir, "shap_eligible_pair_ranking.csv")
  )
  write_result_csv(
    selected,
    file.path(output_dir, "shap_selected_pairs.csv")
  )
  saveRDS(
    list(
      selected = selected,
      conceptual = conceptual,
      raw = raw,
      sampled_cases = shap_cases,
      n_shap_rows = length(shap_rows)
    ),
    file.path(output_dir, "shap_discovery.rds")
  )
  print(selected)
}

shap_scaler <- function(df) {
  vars <- unname(canonical_source)
  ctr <- vapply(df[, vars, drop = FALSE], mean, numeric(1))
  scl <- vapply(df[, vars, drop = FALSE], sd, numeric(1))
  scl[scl == 0] <- 1
  list(ctr = ctr, scl = scl)
}

candidate_terms <- function(covariate1, covariate2) {
  suffix <- pair_key(covariate1, covariate2)
  c(paste0("P_", suffix), paste0("In_", suffix))
}

add_shap_pair_features <- function(df, extra_scaler,
                                   covariate1, covariate2) {
  source1 <- canonical_source[[covariate1]]
  source2 <- canonical_source[[covariate2]]
  z1 <- (
    as.numeric(df[[source1]]) - extra_scaler$ctr[[source1]]
  ) / extra_scaler$scl[[source1]]
  z2 <- (
    as.numeric(df[[source2]]) - extra_scaler$ctr[[source2]]
  ) / extra_scaler$scl[[source2]]
  product <- z1 * z2
  terms <- candidate_terms(covariate1, covariate2)
  df[[terms[[1]]]] <- df$Price_num * product
  df[[terms[[2]]]] <- df$inside * product
  df
}

shap_candidate_formula <- function(covariate1, covariate2) {
  base <- m8trpg_formula("none")
  base_rhs <- sub("\\| 0$", "", as.character(base)[3])
  extra <- paste(
    candidate_terms(covariate1, covariate2),
    collapse = " + "
  )
  fml <- as.formula(paste(
    "chosen ~", base_rhs, "+", extra, "| 0"
  ))
  environment(fml) <- environment()
  fml
}

fit_predict_shap_pair <- function(train_long, valid_long,
                                  covariate1, covariate2) {
  base_scaler <- choice_scaler(train_long)
  extra_scaler <- shap_scaler(train_long)
  tr_feat <- make_m8trpg_features(
    train_long, base_scaler$ctr, base_scaler$scl, "none"
  )
  va_feat <- make_m8trpg_features(
    valid_long, base_scaler$ctr, base_scaler$scl, "none"
  )
  tr_feat <- add_shap_pair_features(
    tr_feat, extra_scaler, covariate1, covariate2
  )
  va_feat <- add_shap_pair_features(
    va_feat, extra_scaler, covariate1, covariate2
  )
  fml <- shap_candidate_formula(covariate1, covariate2)
  model <- mlogit(
    fml,
    data = tr_feat,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen"
  )

  mdat_va <- dfidx(
    va_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen"
  )
  class(mdat_va) <- c("dfidx_mlogit", class(mdat_va))
  mf_va <- model.frame(mdat_va, fml, balanced = TRUE)
  x_va <- model.matrix(mf_va, rhs = 1:3)
  beta <- coef(model)
  stopifnot(all(names(beta) %in% colnames(x_va)))
  eta <- as.numeric(
    x_va[, names(beta), drop = FALSE] %*% beta
  )

  va_map <- unique(va_feat[, c("chid", "No")])
  va_map <- va_map[order(va_map$No), , drop = FALSE]
  eta_chid <- as.character(dfidx::idx(mf_va, 1))
  eta_alt <- as.integer(as.character(dfidx::idx(mf_va, 2)))
  eta_order <- order(match(eta_chid, va_map$chid), eta_alt)
  stopifnot(
    identical(eta_chid[eta_order], rep(va_map$chid, each = 4)),
    identical(eta_alt[eta_order], rep(1:4, times = nrow(va_map)))
  )
  pred <- softmax_margins(eta[eta_order])
  rownames(pred) <- va_map$chid
  extra_coef <- beta[candidate_terms(covariate1, covariate2)]
  stopifnot(!anyNA(pred), !anyNA(extra_coef))
  list(
    pred = pred,
    no = va_map$No,
    extra_coef = extra_coef
  )
}

respondent_bootstrap_gain <- function(truth, baseline, candidate,
                                      case, seed = 4821,
                                      replicates = 2000L) {
  row_loss <- function(pred) {
    pred <- pmin(pmax(pred / rowSums(pred), 1e-15), 1 - 1e-15)
    -rowSums(truth * log(pred))
  }
  respondent_gain <- tapply(
    row_loss(baseline) - row_loss(candidate),
    case,
    mean
  )
  set.seed(seed)
  boot <- replicate(
    replicates,
    mean(sample(
      respondent_gain, length(respondent_gain), replace = TRUE
    ))
  )
  data.frame(
    point_gain = mean(respondent_gain),
    bootstrap_mean = mean(boot),
    bootstrap_sd = sd(boot),
    lower_95 = unname(quantile(boot, 0.025)),
    upper_95 = unname(quantile(boot, 0.975)),
    win_rate = mean(boot > 0)
  )
}

if (stage == "discover") {
  discover_shap_pairs()
}

if (stage == "screen") {
  selected <- read.csv(
    file.path(output_dir, "shap_selected_pairs.csv"),
    stringsAsFactors = FALSE
  )
  split <- readRDS("data_processed/train_val_split.rds")
  tr <- split$train_long_tr
  va <- split$train_long_val
  tr$chid <- paste(tr$Case, tr$Task, sep = "_")
  va$chid <- paste(va$Case, va$Task, sep = "_")
  tr$d2 <- as.integer(tr$alt == 2L)
  tr$d3 <- as.integer(tr$alt == 3L)
  va$d2 <- as.integer(va$alt == 2L)
  va$d3 <- as.integer(va$alt == 3L)
  va_wide <- split$train_wide_val
  va_wide <- va_wide[order(va_wide$No), , drop = FALSE]
  truth <- as.matrix(va_wide[, paste0("Ch", 1:4)])

  baseline_fit <- fit_predict_m8trpg(tr, va)
  baseline <- baseline_fit$pred[
    match(va_wide$No, baseline_fit$no), , drop = FALSE
  ]
  baseline_loss <- log_loss_matrix(truth, baseline)
  stopifnot(abs(baseline_loss - 1.15968144721113) < 1e-8)

  result_rows <- list()
  coefficient_rows <- list()
  prediction_list <- list()
  for (i in seq_len(nrow(selected))) {
    covariate1 <- selected$covariate1[[i]]
    covariate2 <- selected$covariate2[[i]]
    pair <- selected$pair[[i]]
    fitted <- fit_predict_shap_pair(
      tr, va, covariate1, covariate2
    )
    idx <- match(va_wide$No, fitted$no)
    pred <- fitted$pred[idx, , drop = FALSE]
    loss <- log_loss_matrix(truth, pred)
    result_rows[[pair]] <- data.frame(
      pair = pair,
      covariate1 = covariate1,
      covariate2 = covariate2,
      shap_rank = selected$shap_rank[[i]],
      shap_score = selected$mean_abs_shap_interaction[[i]],
      baseline_logloss = baseline_loss,
      candidate_logloss = loss,
      gain = baseline_loss - loss
    )
    coefficient_rows[[pair]] <- data.frame(
      pair = pair,
      term = names(fitted$extra_coef),
      coefficient = unname(fitted$extra_coef)
    )
    prediction_list[[pair]] <- pred
    cat(sprintf(
      "%s: val %.6f; gain %+.6f\n",
      pair, loss, baseline_loss - loss
    ))
  }
  result <- do.call(rbind, result_rows)
  write_result_csv(
    result,
    file.path(output_dir, "shap_interactions_screen.csv")
  )
  write_result_csv(
    do.call(rbind, coefficient_rows),
    file.path(output_dir, "shap_interactions_screen_coefficients.csv")
  )
  saveRDS(
    list(
      result = result,
      predictions = prediction_list,
      baseline = baseline,
      validation_no = va_wide$No
    ),
    file.path(output_dir, "shap_interactions_screen.rds")
  )
  print(result, digits = 7)
}

if (stage == "cv") {
  screen <- read.csv(
    file.path(output_dir, "shap_interactions_screen.csv"),
    stringsAsFactors = FALSE
  )
  requested <- Sys.getenv("CODEX_CANDIDATES", "")
  if (nzchar(requested)) {
    candidates <- strsplit(requested, ",", fixed = TRUE)[[1]]
  } else {
    candidates <- screen$pair[screen$gain > 0]
  }
  if (length(candidates) == 0L) {
    cat("No SHAP-nominated pair passed the screen; CV skipped.\n")
    quit(save = "no", status = 0)
  }
  stopifnot(all(candidates %in% screen$pair))

  train <- read.csv("csv files/train.csv")
  truth <- as.matrix(train[, paste0("Ch", 1:4)])
  train_long <- reshape_choice_long(train)
  saved <- readRDS("data_processed/oof_ensemble_v10.rds")
  fold_map <- saved$fold_of_case
  baseline <- saved$oof_mlogit
  xgb <- saved$oof_xgb
  v11 <- 0.8 * baseline + 0.2 * xgb
  stopifnot(
    abs(log_loss_matrix(truth, baseline) - 1.1470212110518) < 1e-8,
    abs(log_loss_matrix(truth, v11) - 1.145094) < 5e-6
  )

  candidate_oof <- lapply(
    candidates,
    function(x) matrix(NA_real_, nrow(train), 4)
  )
  names(candidate_oof) <- candidates
  fold_rows <- list()
  coefficient_rows <- list()

  for (pair in candidates) {
    row <- screen[screen$pair == pair, , drop = FALSE]
    covariate1 <- row$covariate1[[1]]
    covariate2 <- row$covariate2[[1]]
    for (fold in 1:5) {
      val_cases <- as.integer(names(fold_map)[fold_map == fold])
      tr <- train_long[
        !(train_long$Case %in% val_cases), , drop = FALSE
      ]
      va <- train_long[
        train_long$Case %in% val_cases, , drop = FALSE
      ]
      fitted <- fit_predict_shap_pair(
        tr, va, covariate1, covariate2
      )
      idx <- match(fitted$no, train$No)
      stopifnot(!anyNA(idx))
      candidate_oof[[pair]][idx, ] <- fitted$pred
      base_fold <- log_loss_matrix(truth[idx, ], baseline[idx, ])
      candidate_fold <- log_loss_matrix(
        truth[idx, ], fitted$pred
      )
      fold_rows[[length(fold_rows) + 1L]] <- data.frame(
        pair = pair,
        fold = fold,
        baseline_logloss = base_fold,
        candidate_logloss = candidate_fold,
        gain = base_fold - candidate_fold
      )
      coefficient_rows[[length(coefficient_rows) + 1L]] <-
        data.frame(
          pair = pair,
          fold = fold,
          term = names(fitted$extra_coef),
          coefficient = unname(fitted$extra_coef)
        )
      cat(sprintf(
        "%s fold %d: gain %+.6f\n",
        pair, fold, base_fold - candidate_fold
      ))
    }
    stopifnot(!anyNA(candidate_oof[[pair]]))
  }

  summary_rows <- list()
  bootstrap_rows <- list()
  for (pair in candidates) {
    pred <- candidate_oof[[pair]]
    fixed_blend <- 0.8 * pred + 0.2 * xgb
    summary_rows[[pair]] <- data.frame(
      pair = pair,
      baseline_mlogit_logloss = log_loss_matrix(truth, baseline),
      candidate_mlogit_logloss = log_loss_matrix(truth, pred),
      mlogit_gain =
        log_loss_matrix(truth, baseline) -
        log_loss_matrix(truth, pred),
      v11_logloss = log_loss_matrix(truth, v11),
      fixed_080_blend_logloss =
        log_loss_matrix(truth, fixed_blend),
      fixed_080_blend_gain =
        log_loss_matrix(truth, v11) -
        log_loss_matrix(truth, fixed_blend)
    )
    for (comparison in c("mlogit", "fixed_080_blend")) {
      if (comparison == "mlogit") {
        base_pred <- baseline
        candidate_pred <- pred
      } else {
        base_pred <- v11
        candidate_pred <- fixed_blend
      }
      bootstrap_rows[[length(bootstrap_rows) + 1L]] <- cbind(
        data.frame(pair = pair, comparison = comparison),
        respondent_bootstrap_gain(
          truth, base_pred, candidate_pred, train$Case
        )
      )
    }
  }

  summary <- do.call(rbind, summary_rows)
  bootstrap <- do.call(rbind, bootstrap_rows)
  write_result_csv(
    do.call(rbind, fold_rows),
    file.path(output_dir, "shap_interactions_cv_folds.csv")
  )
  write_result_csv(
    do.call(rbind, coefficient_rows),
    file.path(output_dir, "shap_interactions_cv_coefficients.csv")
  )
  write_result_csv(
    summary,
    file.path(output_dir, "shap_interactions_cv.csv")
  )
  write_result_csv(
    bootstrap,
    file.path(output_dir, "shap_interactions_bootstrap.csv")
  )
  saveRDS(
    list(
      candidates = candidates,
      oof = candidate_oof,
      summary = summary,
      bootstrap = bootstrap
    ),
    file.path(output_dir, "shap_interactions_oof.rds")
  )
  print(summary, digits = 7)
  print(bootstrap, digits = 7)
}
