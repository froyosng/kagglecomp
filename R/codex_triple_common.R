suppressPackageStartupMessages({
  library(mlogit)
  library(dfidx)
})
source("R/codex_shift_common.R")

triple_candidate_specs <- list(
  triple_income_age = list(
    family = "triple_joint",
    pairs = list(c("income", "age")),
    segment_covariates = character()
  ),
  triple_income_miles = list(
    family = "triple_joint",
    pairs = list(c("income", "miles")),
    segment_covariates = character()
  ),
  triple_income_night = list(
    family = "triple_joint",
    pairs = list(c("income", "night")),
    segment_covariates = character()
  ),
  triple_all = list(
    family = "triple_joint",
    pairs = list(
      c("income", "age"),
      c("income", "miles"),
      c("income", "night")
    ),
    segment_covariates = character()
  ),
  triple_price_income_age = list(
    family = "triple_atomic",
    pairs = list(),
    price_only_pairs = list(c("income", "age")),
    segment_covariates = character()
  ),
  triple_inside_income_age = list(
    family = "triple_atomic",
    pairs = list(),
    inside_only_pairs = list(c("income", "age")),
    segment_covariates = character()
  ),
  triple_price_income_miles = list(
    family = "triple_atomic",
    pairs = list(),
    price_only_pairs = list(c("income", "miles")),
    segment_covariates = character()
  ),
  triple_inside_income_miles = list(
    family = "triple_atomic",
    pairs = list(),
    inside_only_pairs = list(c("income", "miles")),
    segment_covariates = character()
  ),
  triple_price_income_night = list(
    family = "triple_atomic",
    pairs = list(),
    price_only_pairs = list(c("income", "night")),
    segment_covariates = character()
  ),
  triple_inside_income_night = list(
    family = "triple_atomic",
    pairs = list(),
    inside_only_pairs = list(c("income", "night")),
    segment_covariates = character()
  ),
  segment_income = list(
    family = "segment",
    pairs = list(),
    segment_covariates = "income"
  ),
  segment_age = list(
    family = "segment",
    pairs = list(),
    segment_covariates = "age"
  ),
  segment_miles = list(
    family = "segment",
    pairs = list(),
    segment_covariates = "miles"
  ),
  segment_night = list(
    family = "segment",
    pairs = list(),
    segment_covariates = "night"
  ),
  segment_all = list(
    family = "segment",
    pairs = list(),
    segment_covariates = c("income", "age", "miles", "night")
  )
)

continuous_source <- c(
  income = "incomea",
  age = "agea",
  miles = "milesa",
  night = "nighta"
)

candidate_extra_terms <- function(candidate) {
  spec <- triple_candidate_specs[[candidate]]
  stopifnot(!is.null(spec))
  out <- character()
  for (pair in spec$pairs) {
    suffix <- paste(pair, collapse = "_")
    out <- c(out, paste0("P_", suffix), paste0("In_", suffix))
  }
  if (!is.null(spec$price_only_pairs)) {
    for (pair in spec$price_only_pairs) {
      out <- c(out, paste0("P_", paste(pair, collapse = "_")))
    }
  }
  if (!is.null(spec$inside_only_pairs)) {
    for (pair in spec$inside_only_pairs) {
      out <- c(out, paste0("In_", paste(pair, collapse = "_")))
    }
  }
  for (covariate in spec$segment_covariates) {
    out <- c(out, paste0("P_", covariate, "_seg", 2:6))
  }
  out
}

add_candidate_features <- function(df, ctr, scl, candidate) {
  spec <- triple_candidate_specs[[candidate]]
  stopifnot(!is.null(spec))

  z <- lapply(names(continuous_source), function(covariate) {
    source_name <- continuous_source[[covariate]]
    (as.numeric(df[[source_name]]) - ctr[[source_name]]) /
      scl[[source_name]]
  })
  names(z) <- names(continuous_source)

  for (pair in spec$pairs) {
    suffix <- paste(pair, collapse = "_")
    product <- z[[pair[1]]] * z[[pair[2]]]
    df[[paste0("P_", suffix)]] <- df$Price_num * product
    df[[paste0("In_", suffix)]] <- df$inside * product
  }
  if (!is.null(spec$price_only_pairs)) {
    for (pair in spec$price_only_pairs) {
      suffix <- paste(pair, collapse = "_")
      product <- z[[pair[1]]] * z[[pair[2]]]
      df[[paste0("P_", suffix)]] <- df$Price_num * product
    }
  }
  if (!is.null(spec$inside_only_pairs)) {
    for (pair in spec$inside_only_pairs) {
      suffix <- paste(pair, collapse = "_")
      product <- z[[pair[1]]] * z[[pair[2]]]
      df[[paste0("In_", suffix)]] <- df$inside * product
    }
  }
  for (covariate in spec$segment_covariates) {
    for (segment in 2:6) {
      df[[paste0("P_", covariate, "_seg", segment)]] <-
        df$Price_num * z[[covariate]] *
        as.integer(df$segmentind == segment)
    }
  }
  df
}

candidate_formula <- function(candidate) {
  base <- m8trpg_formula("none")
  base_rhs <- sub("\\| 0$", "", as.character(base)[3])
  extra <- paste(candidate_extra_terms(candidate), collapse = " + ")
  out <- as.formula(paste("chosen ~", base_rhs, "+", extra, "| 0"))
  environment(out) <- environment()
  out
}

fit_predict_candidate <- function(train_long, valid_long, candidate) {
  scaler <- choice_scaler(train_long)
  tr_feat <- make_m8trpg_features(
    train_long, scaler$ctr, scaler$scl, "none"
  )
  va_feat <- make_m8trpg_features(
    valid_long, scaler$ctr, scaler$scl, "none"
  )
  tr_feat <- add_candidate_features(
    tr_feat, scaler$ctr, scaler$scl, candidate
  )
  va_feat <- add_candidate_features(
    va_feat, scaler$ctr, scaler$scl, candidate
  )

  fml <- candidate_formula(candidate)
  model <- mlogit(
    fml,
    data = tr_feat,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen"
  )

  # predict.mlogit replays idx/choice conversion and can fail on already-dfidx
  # new data. Construct the held-out design directly, then apply softmax.
  mdat_va <- dfidx(
    va_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen"
  )
  class(mdat_va) <- c("dfidx_mlogit", class(mdat_va))
  mf_va <- model.frame(mdat_va, fml, balanced = TRUE)
  x_va <- model.matrix(mf_va, rhs = 1:3)
  beta <- coef(model)
  stopifnot(all(names(beta) %in% colnames(x_va)))
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

  pred <- softmax_margins(eta[eta_order])
  rownames(pred) <- va_map$chid
  extra_coef <- beta[candidate_extra_terms(candidate)]
  stopifnot(!anyNA(pred), !anyNA(extra_coef))
  list(
    pred = pred,
    no = va_map$No,
    model = model,
    extra_coef = extra_coef
  )
}

prepare_saved_long <- function(df) {
  df$chid <- paste(df$Case, df$Task, sep = "_")
  df$d2 <- as.integer(df$alt == 2L)
  df$d3 <- as.integer(df$alt == 3L)
  df
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
