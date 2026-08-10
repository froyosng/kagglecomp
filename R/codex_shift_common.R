suppressPackageStartupMessages({
  library(mlogit)
  library(dfidx)
})
source("R/codex_modeling_common.R")

shift_scaler_vars <- c(
  "incomea", "agea", "milesa", "nighta",
  "genderind", "Urbind", "educind"
)

shift_resp_vars <- c(
  "segmentind", "yearind", "milesind", "milesa", "nightind", "nighta",
  "pparkind", "genderind", "ageind", "agea", "educind",
  "regionind", "Urbind", "incomeind", "incomea"
)

reshape_choice_long <- function(wide) {
  varying_pattern <- paste0(
    "^(", paste(c(attrs, "Price", "Ch"), collapse = "|"), ")([1-4])$"
  )
  fixed_names <- names(wide)[!grepl(varying_pattern, names(wide))]
  parts <- lapply(1:4, function(a) {
    out <- wide[, fixed_names, drop = FALSE]
    for (v in c(attrs, "Price", "Ch")) out[[v]] <- wide[[paste0(v, a)]]
    out$alt <- a
    out$chosen <- as.integer(out$Ch == 1)
    out
  })
  out <- do.call(rbind, parts)
  out <- out[order(out$No, out$alt), , drop = FALSE]
  out$chid <- paste(out$Case, out$Task, sep = "_")
  out$d2 <- as.integer(out$alt == 2L)
  out$d3 <- as.integer(out$alt == 3L)
  rownames(out) <- NULL
  out
}

choice_scaler <- function(long_df) {
  ctr <- vapply(long_df[, shift_scaler_vars, drop = FALSE], mean, numeric(1))
  scl <- vapply(long_df[, shift_scaler_vars, drop = FALSE], sd, numeric(1))
  scl[scl == 0] <- 1
  list(ctr = ctr, scl = scl)
}

add_attribute_rank_features <- function(df, scope = c("none", "inside", "all")) {
  scope <- match.arg(scope)
  if (scope == "none") return(df)

  df <- df[order(df$No, df$alt), , drop = FALSE]
  for (a in attrs) {
    values <- matrix(as.numeric(df[[a]]), ncol = 4, byrow = TRUE)
    if (scope == "inside") {
      ref <- values[, 1:3, drop = FALSE]
      lo <- apply(ref, 1, min)
      hi <- apply(ref, 1, max)
      inside <- df$alt != 4L
      df[[paste0(a, "_is_min")]] <-
        as.integer(inside & df[[a]] == rep(lo, each = 4))
      df[[paste0(a, "_is_max")]] <-
        as.integer(inside & df[[a]] == rep(hi, each = 4))
    } else {
      lo <- apply(values, 1, min)
      hi <- apply(values, 1, max)
      df[[paste0(a, "_is_min")]] <-
        as.integer(df[[a]] == rep(lo, each = 4))
      df[[paste0(a, "_is_max")]] <-
        as.integer(df[[a]] == rep(hi, each = 4))
    }
  }
  df
}

make_m8trpg_features <- function(df, ctr, scl,
                                 attr_rank_scope = "none") {
  df <- as.data.frame(df)
  df <- df[order(df$No, df$alt), , drop = FALSE]
  z <- sweep(
    sweep(as.matrix(df[, shift_scaler_vars]), 2, ctr, "-"),
    2, scl, "/"
  )
  df$inside <- as.integer(df$alt != 4L)
  df$Price_num <- as.numeric(df$Price)
  for (k in 2:12) {
    df[[paste0("Pr_lvl", k)]] <- as.integer(df$Price_num == k)
  }
  df$Task_c <- (as.numeric(df$Task) - 10) / 9

  df$P_income <- df$Price_num * z[, "incomea"]
  df$P_age <- df$Price_num * z[, "agea"]
  df$P_miles <- df$Price_num * z[, "milesa"]
  df$P_night <- df$Price_num * z[, "nighta"]
  df$In_income <- df$inside * z[, "incomea"]
  df$In_age <- df$inside * z[, "agea"]
  df$In_miles <- df$inside * z[, "milesa"]
  df$In_night <- df$inside * z[, "nighta"]
  df$In_gender <- df$inside * z[, "genderind"]
  df$In_urb <- df$inside * z[, "Urbind"]
  df$In_educ <- df$inside * z[, "educind"]

  for (s in 2:6) {
    df[[paste0("P_seg", s)]] <-
      df$Price_num * as.integer(df$segmentind == s)
    df[[paste0("In_seg", s)]] <-
      df$inside * as.integer(df$segmentind == s)
  }
  df$P_task <- df$Price_num * df$Task_c
  df$In_task <- df$inside * df$Task_c

  for (s in 2:5) {
    df[[paste0("P_region", s)]] <-
      df$Price_num * as.integer(df$regionind == s)
    df[[paste0("In_region", s)]] <-
      df$inside * as.integer(df$regionind == s)
    df[[paste0("P_ppark", s)]] <-
      df$Price_num * as.integer(df$pparkind == s)
    df[[paste0("In_ppark", s)]] <-
      df$inside * as.integer(df$pparkind == s)
  }

  price_values <- matrix(df$Price_num, ncol = 4, byrow = TRUE)
  inside_prices <- price_values[, 1:3, drop = FALSE]
  price_min <- apply(inside_prices, 1, min)
  price_max <- apply(inside_prices, 1, max)
  df$price_min <- rep(price_min, each = 4)
  df$price_max <- rep(price_max, each = 4)
  df$is_cheapest <- as.integer(
    df$inside == 1L & df$Price_num == df$price_min
  )
  df$is_dearest <- as.integer(
    df$inside == 1L & df$Price_num == df$price_max
  )
  df$price_gap_min <- ifelse(
    df$inside == 1L, df$Price_num - df$price_min, 0
  )
  df$price_gap_max <- ifelse(
    df$inside == 1L, df$price_max - df$Price_num, 0
  )

  add_attribute_rank_features(df, attr_rank_scope)
}

m8trpg_formula <- function(attr_rank_scope = "none") {
  attr_terms <- paste0("factor(", attrs, ")")
  price_terms <- paste0("Pr_lvl", 2:12)
  int_terms <- c(
    "P_income", "P_age", "P_miles", "P_night",
    "In_income", "In_age", "In_miles", "In_night",
    "In_gender", "In_urb", "In_educ",
    paste0("P_seg", 2:6), paste0("In_seg", 2:6),
    "P_task", "In_task",
    paste0("P_region", 2:5), paste0("In_region", 2:5),
    paste0("P_ppark", 2:5), paste0("In_ppark", 2:5),
    "is_cheapest", "is_dearest", "price_gap_min", "price_gap_max"
  )
  if (attr_rank_scope != "none") {
    int_terms <- c(
      int_terms,
      paste0(attrs, "_is_min"),
      paste0(attrs, "_is_max")
    )
  }
  as.formula(paste(
    "chosen ~",
    paste(c(attr_terms, price_terms, "d2", "d3", int_terms),
          collapse = " + "),
    "| 0"
  ))
}

fit_predict_m8trpg <- function(train_long, valid_long,
                               case_weights = NULL,
                               attr_rank_scope = "none") {
  scaler <- choice_scaler(train_long)
  tr_feat <- make_m8trpg_features(
    train_long, scaler$ctr, scaler$scl, attr_rank_scope
  )
  va_feat <- make_m8trpg_features(
    valid_long, scaler$ctr, scaler$scl, attr_rank_scope
  )

  if (is.null(case_weights)) {
    tr_feat$case_weight <- 1
  } else {
    tr_feat$case_weight <- unname(
      case_weights[as.character(tr_feat$Case)]
    )
    stopifnot(!anyNA(tr_feat$case_weight))
  }

  fml <- m8trpg_formula(attr_rank_scope)
  environment(fml) <- environment()
  model <- mlogit(
    fml,
    data = tr_feat,
    idx = list(c("chid", "Case"), "alt"),
    choice = "chosen",
    weights = case_weight
  )
  # predict.mlogit replays the original idx/choice call and errors when its
  # internally converted newdata is already dfidx. Build the held-out design
  # matrix directly and apply the conditional-logit softmax instead.
  mdat_va <- dfidx(
    va_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen"
  )
  class(mdat_va) <- c("dfidx_mlogit", class(mdat_va))
  mf_va <- model.frame(mdat_va, fml, balanced = TRUE)
  x_va <- model.matrix(mf_va, rhs = 1:3)
  beta <- coef(model)
  if (!all(names(beta) %in% colnames(x_va))) {
    cat("Missing validation columns:\n")
    print(setdiff(names(beta), colnames(x_va)))
    cat("Extra validation columns (first 30):\n")
    print(head(setdiff(colnames(x_va), names(beta)), 30))
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
  list(pred = pred, no = va_map$No, model = model)
}

distinct_respondents <- function(wide) {
  wide[!duplicated(wide$Case), c("Case", shift_resp_vars), drop = FALSE]
}

fit_density_ratio <- function(source_fit_wide, target_wide) {
  source <- distinct_respondents(source_fit_wide)
  target <- distinct_respondents(target_wide)
  source_x <- source[, shift_resp_vars, drop = FALSE]
  target_x <- target[, shift_resp_vars, drop = FALSE]
  source_x$is_test <- 0L
  target_x$is_test <- 1L
  domain <- rbind(source_x, target_x)
  classifier <- glm(is_test ~ ., data = domain, family = binomial())
  list(
    classifier = classifier,
    prior_ratio = nrow(source) / nrow(target),
    source_cases = source$Case
  )
}

predict_density_ratio <- function(ratio_fit, wide) {
  respondents <- distinct_respondents(wide)
  propensity <- predict(
    ratio_fit$classifier,
    newdata = respondents[, shift_resp_vars, drop = FALSE],
    type = "response"
  )
  propensity <- pmin(pmax(propensity, 1e-6), 1 - 1e-6)
  ratio <- propensity / (1 - propensity) * ratio_fit$prior_ratio
  names(ratio) <- respondents$Case
  ratio
}

stabilize_ratio <- function(ratio, cap = 20, power = 1) {
  out <- pmin(as.numeric(ratio), cap)^power
  names(out) <- names(ratio)
  out / mean(out)
}

weighted_log_loss <- function(actual, pred, weights, eps = 1e-15) {
  pred <- pred / rowSums(pred)
  pred <- pmin(pmax(pred, eps), 1 - eps)
  row_loss <- -rowSums(as.matrix(actual) * log(pred))
  weighted.mean(row_loss, weights)
}

effective_sample_size <- function(weights) {
  sum(weights)^2 / sum(weights^2)
}
