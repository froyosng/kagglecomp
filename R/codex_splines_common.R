# Shared machinery for the smooth-spline covariate-interaction experiment.
#
# Hypothesis under test: replace the LINEAR Price x z(covariate) and
# inside x z(covariate) interaction terms in the current-best m8trpg
# conditional logit with a natural cubic spline (splines::ns()) basis for
# the same covariate, for age, mileage, and income (the three continuous
# covariates most central to this project's confirmed heterogeneity
# findings). This is deliberately NOT a re-test of the already-rejected
# binned/categorical covariate treatment (2026-07-26 cleaning_log entry,
# "Binned covariate x Price interactions"): that attempt used discrete
# dummy levels (nightind, milesind) and blew up from sparse-cell
# quasi-separation (nightind levels 9-10, ~6 respondents each, a
# coefficient of -0.815). A natural cubic spline is smooth (C2-continuous),
# places knots at quantiles of the training respondents' covariate values
# (so each spline segment sees comparable data mass, unlike arbitrary bin
# edges), and is constrained to extrapolate LINEARLY beyond the boundary
# knots -- a materially different, better-behaved functional form, not a
# re-run of the rejected one.
#
# Reuses this project's own established m8trpg machinery unmodified
# (R/codex_modeling_common.R, R/codex_shift_common.R) so the comparison is
# a clean like-for-like substitution: same attributes, same Price-as-factor
# terms, same segment/task/region/ppark interactions, same price-gap
# context terms -- only the functional form of ONE covariate's pair of
# interaction terms changes at a time.

suppressPackageStartupMessages({
  library(mlogit)
  library(dfidx)
  library(splines)
})
source("R/codex_modeling_common.R")
source("R/codex_shift_common.R")

## ---- Candidate family -----------------------------------------------------

# Map from the covariate name used in m8trpg's own term names (P_age/In_age
# etc.) to the actual continuous source column in the raw data. Deliberately
# excludes night: the task scope is age, mileage, and income only (the three
# covariates the pre-registration explicitly names), night is left untouched
# exactly as in every other term of m8trpg.
spline_covariate_source <- c(age = "agea", miles = "milesa", income = "incomea")

spline_df_grid <- c(3L, 4L)

spline_candidate_names <- as.vector(outer(
  names(spline_covariate_source), spline_df_grid,
  FUN = function(cov, d) paste0(cov, "_df", d)
))

spline_candidate_spec <- function(candidate) {
  parts <- regmatches(candidate, regexec("^(age|miles|income)_df([0-9]+)$", candidate))[[1]]
  stopifnot(length(parts) == 3L)
  list(covariate = parts[2], df = as.integer(parts[3]))
}

## ---- Fold-safe spline basis: fit on training respondents only ------------

# Fits splines::ns() on the UNIQUE per-respondent covariate values in the
# training fold only (never on validation/test), exactly mirroring how
# choice_scaler() computes mean/sd from the training fold only elsewhere in
# this project. predict.ns() on the returned object then applies the frozen
# knots to any new data (validation, test), including linear extrapolation
# beyond the training boundary knots -- the natural-spline property that
# specifically avoids the runaway-coefficient risk that broke the binned
# version at sparse tails.
fit_ns_basis <- function(train_long, covariate, spline_df) {
  source_col <- spline_covariate_source[[covariate]]
  stopifnot(!is.null(source_col))
  resp <- train_long[!duplicated(train_long$Case), c("Case", source_col)]
  x <- as.numeric(resp[[source_col]])
  stopifnot(!anyNA(x), length(x) >= spline_df + 2L)
  ns(x, df = spline_df)
}

# Adds P_<covariate>_ns<j> / In_<covariate>_ns<j> columns (j = 1..spline_df)
# to a data frame that has already been through make_m8trpg_features() (so
# Price_num/inside already exist). Must be called with the SAME ns_obj
# (fitted on that fold's training respondents) for both the training and
# validation/target data of that fold.
add_spline_terms <- function(df, covariate, ns_obj) {
  source_col <- spline_covariate_source[[covariate]]
  x <- as.numeric(df[[source_col]])
  basis <- predict(ns_obj, x)
  stopifnot(!anyNA(basis))
  for (j in seq_len(ncol(basis))) {
    df[[paste0("P_", covariate, "_ns", j)]] <- df$Price_num * basis[, j]
    df[[paste0("In_", covariate, "_ns", j)]] <- df$inside * basis[, j]
  }
  df
}

spline_extra_terms <- function(covariate, spline_df) {
  c(
    paste0("P_", covariate, "_ns", seq_len(spline_df)),
    paste0("In_", covariate, "_ns", seq_len(spline_df))
  )
}

# Base m8trpg formula with the linear P_<covariate>/In_<covariate> terms
# removed and the spline-interaction terms substituted in their place. Every
# other term (attributes, Price-as-factor levels, the OTHER covariates'
# linear interactions, segment/task/region/ppark, price-gap context) is
# untouched -- a genuine one-term-family substitution, not a re-specified
# model.
spline_formula <- function(covariate, spline_df) {
  base <- m8trpg_formula("none")
  base_rhs <- sub("\\| 0$", "", as.character(base)[3])
  base_terms <- trimws(strsplit(base_rhs, "\\+")[[1]])
  drop_terms <- c(paste0("P_", covariate), paste0("In_", covariate))
  stopifnot(all(drop_terms %in% base_terms))
  kept_terms <- setdiff(base_terms, drop_terms)
  extra_terms <- spline_extra_terms(covariate, spline_df)
  rhs <- paste(c(kept_terms, extra_terms), collapse = " + ")
  out <- as.formula(paste("chosen ~", rhs, "| 0"))
  environment(out) <- environment()
  out
}

# Fit the candidate mlogit on train_long, predict on valid_long. Mirrors
# fit_predict_m8trpg()/fit_predict_candidate() in R/codex_shift_common.R and
# R/codex_triple_common.R exactly (same dfidx + model.matrix prediction
# path, since predict.mlogit() cannot be trusted on already-dfidx newdata --
# a bug this project documented and worked around previously).
fit_predict_spline <- function(train_long, valid_long, candidate) {
  spec <- spline_candidate_spec(candidate)
  covariate <- spec$covariate
  spline_df <- spec$df

  scaler <- choice_scaler(train_long)
  tr_feat <- make_m8trpg_features(train_long, scaler$ctr, scaler$scl, "none")
  va_feat <- make_m8trpg_features(valid_long, scaler$ctr, scaler$scl, "none")

  ns_obj <- fit_ns_basis(train_long, covariate, spline_df)
  tr_feat <- add_spline_terms(tr_feat, covariate, ns_obj)
  va_feat <- add_spline_terms(va_feat, covariate, ns_obj)

  fml <- spline_formula(covariate, spline_df)
  environment(fml) <- environment()
  model <- mlogit(
    fml,
    data = tr_feat,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen"
  )

  mdat_va <- dfidx(va_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
  class(mdat_va) <- c("dfidx_mlogit", class(mdat_va))
  mf_va <- model.frame(mdat_va, fml, balanced = TRUE)
  x_va <- model.matrix(mf_va, rhs = 1:3)
  beta <- coef(model)
  if (!all(names(beta) %in% colnames(x_va))) {
    cat("Missing validation columns:\n")
    print(setdiff(names(beta), colnames(x_va)))
    stop("held-out model matrix does not align with fitted coefficients")
  }
  eta <- as.numeric(x_va[, names(beta), drop = FALSE] %*% beta)

  va_map <- unique(va_feat[, c("chid", "No")])
  va_map <- va_map[order(va_map$No), , drop = FALSE]
  eta_chid <- as.character(dfidx::idx(mf_va, 1))
  eta_alt <- as.integer(as.character(dfidx::idx(mf_va, 2)))
  eta_order <- order(match(eta_chid, va_map$chid), eta_alt)
  stopifnot(
    identical(eta_chid[eta_order], rep(va_map$chid, each = 4)),
    identical(eta_alt[eta_order], rep(1:4, times = nrow(va_map)))
  )
  raw_pred <- softmax_margins(eta[eta_order])
  rownames(raw_pred) <- va_map$chid
  pred <- raw_pred[match(va_map$chid, rownames(raw_pred)), , drop = FALSE]
  stopifnot(!anyNA(pred))

  extra_terms <- spline_extra_terms(covariate, spline_df)
  extra_coef <- beta[extra_terms]
  stopifnot(!anyNA(extra_coef))

  list(pred = pred, no = va_map$No, model = model, extra_coef = extra_coef, ns_obj = ns_obj)
}

## ---- Respondent-clustered paired bootstrap (100,000 replicates) ----------

# Vectorized to make 100,000 replicates practical: draws a n_case x n_boot
# matrix of respondent indices (with replacement) in chunks and averages.
# Matches the method already used in R/codex_repeat_cv_common.R and
# R/codex_price_history_only.R (same chunked resampling, same seed
# convention) but generalized to accept any number of pooled-repeat columns.
spline_bootstrap_gain <- function(case_gain_matrix, n_boot = 100000L, seed = 4821L) {
  case_gain_matrix <- as.matrix(case_gain_matrix)
  stopifnot(nrow(case_gain_matrix) == 1135L, !anyNA(case_gain_matrix))
  case_gain <- rowMeans(case_gain_matrix)
  set.seed(seed)
  n_case <- length(case_gain)
  bootstrap <- numeric(n_boot)
  for (start in seq.int(1L, n_boot, by = 1000L)) {
    stop_at <- min(n_boot, start + 999L)
    n_this <- stop_at - start + 1L
    sampled <- matrix(sample.int(n_case, n_case * n_this, replace = TRUE), nrow = n_case)
    bootstrap[start:stop_at] <- colMeans(matrix(case_gain[sampled], nrow = n_case))
  }
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
      n_boot = n_boot
    ),
    case_gain = case_gain,
    bootstrap = bootstrap
  )
}

# Per-respondent mean log-loss gain (baseline - candidate; positive =
# candidate better), the input to spline_bootstrap_gain().
row_loss <- function(truth, pred, eps = 1e-15) {
  pred <- pmin(pmax(pred / rowSums(pred), eps), 1 - eps)
  -rowSums(as.matrix(truth) * log(pred))
}

case_gain_vector <- function(truth, baseline_pred, candidate_pred, case) {
  gain <- row_loss(truth, baseline_pred) - row_loss(truth, candidate_pred)
  unname(tapply(gain, case, mean))
}
