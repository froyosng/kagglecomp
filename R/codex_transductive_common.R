# Shared helpers for the codex_transductive track: quantile-mapping moment match
# and the m8trpg feature/formula builder, kept close to R/cv_ensemble_v10.R /
# R/submit_ensemble_v11.R so the only deliberate difference is the treatment under
# test.

attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
           "PP","KA","SC","TS","NV","MA","LB","AF","HU")
xgb_covariates <- c("segmentind","yearind","milesind","milesa","nightind","nighta",
                     "pparkind","genderind","ageind","agea","educind",
                     "regionind","Urbind","incomeind","incomea")

log_loss_matrix <- function(truth, pred, eps = 1e-15) {
  truth <- as.matrix(truth)
  pred <- as.matrix(pred)
  pred <- pred / rowSums(pred)
  pred <- pmin(pmax(pred, eps), 1 - eps)
  -mean(rowSums(truth * log(pred)))
}

row_log_loss <- function(truth, pred, eps = 1e-15) {
  truth <- as.matrix(truth)
  pred <- as.matrix(pred)
  pred <- pred / rowSums(pred)
  pred <- pmin(pmax(pred, eps), 1 - eps)
  -rowSums(truth * log(pred))
}

# Quantile-mapping moment match: recode x_new (drawn from `source_values`'
# distribution) onto the distributional shape of `target_values`, preserving each
# value's rank/percentile within its own source distribution. Uses linear
# interpolation between order statistics (plotting position (i-0.5)/n) for the
# source CDF, and type-7 (R default) quantile interpolation for the target
# quantile function. Extrapolates by clamping to the boundary percentile
# (rule = 2) rather than exploding outside the observed source range.
quantile_match <- function(x_new, source_values, target_values) {
  stopifnot(length(source_values) >= 2L, length(target_values) >= 2L)
  sorted_source <- sort(source_values)
  n <- length(sorted_source)
  p_grid <- ((seq_len(n)) - 0.5) / n
  p <- approx(x = sorted_source, y = p_grid, xout = x_new, rule = 2, ties = mean)$y
  p <- pmin(pmax(p, 1e-6), 1 - 1e-6)
  as.numeric(quantile(target_values, probs = p, type = 7, names = FALSE))
}

# Build the m8trpg feature set given already-long-format df and a covariate
# recoding function `recode_fn(varname, x_train_side) -> matched values`, used only
# for the four continuous Price/inside interaction covariates. `recode_fn` receives
# the raw variable name and the raw numeric vector, and must return a same-length
# numeric vector of matched RAW values (matching-then-standardizing is handled
# internally below, using `std_ctr`/`std_scl`, so recode_fn need not standardize).
# When `recode_fn` is NULL, this reproduces the current-best m8trpg exactly (raw
# value standardized with std_ctr/std_scl, no recoding) -- used to sanity check
# against the saved OOF before trusting any modified version.
make_features_m8trpg <- function(df, std_ctr, std_scl, recode_fn = NULL) {
  df <- as.data.frame(df)
  scaler_vars <- c("incomea","agea","milesa","nighta","genderind","Urbind","educind")

  raw <- df[scaler_vars]
  if (!is.null(recode_fn)) {
    for (v in c("incomea","agea","milesa","nighta")) {
      raw[[v]] <- recode_fn(v, raw[[v]])
    }
  }
  z <- sweep(sweep(raw, 2, std_ctr, "-"), 2, std_scl, "/")

  df$inside <- as.integer(df$alt != 4)
  df$Price_num <- as.numeric(df$Price)
  for (k in 2:12) df[[paste0("Pr_lvl", k)]] <- as.integer(df$Price_num == k)
  df$Task_c <- (as.numeric(df$Task) - 10) / 9

  df$P_income <- df$Price_num * z$incomea
  df$P_age <- df$Price_num * z$agea
  df$P_miles <- df$Price_num * z$milesa
  df$P_night <- df$Price_num * z$nighta
  df$In_income <- df$inside * z$incomea
  df$In_age <- df$inside * z$agea
  df$In_miles <- df$inside * z$milesa
  df$In_night <- df$inside * z$nighta
  df$In_gender <- df$inside * z$genderind
  df$In_urb <- df$inside * z$Urbind
  df$In_educ <- df$inside * z$educind

  for (s in 2:6) {
    df[[paste0("P_seg", s)]] <- df$Price_num * (df$segmentind == s)
    df[[paste0("In_seg", s)]] <- df$inside * (df$segmentind == s)
  }

  df$P_task <- df$Price_num * df$Task_c
  df$In_task <- df$inside * df$Task_c

  for (s in 2:5) {
    df[[paste0("P_region", s)]] <- df$Price_num * (df$regionind == s)
    df[[paste0("In_region", s)]] <- df$inside * (df$regionind == s)
    df[[paste0("P_ppark", s)]] <- df$Price_num * (df$pparkind == s)
    df[[paste0("In_ppark", s)]] <- df$inside * (df$pparkind == s)
  }

  task_stats <- aggregate(Price_num ~ chid, data = df[df$inside == 1, , drop = FALSE],
                           FUN = function(x) c(min(x), max(x)))
  price_min_map <- setNames(task_stats$Price_num[, 1], task_stats$chid)
  price_max_map <- setNames(task_stats$Price_num[, 2], task_stats$chid)
  df$price_min <- unname(price_min_map[df$chid])
  df$price_max <- unname(price_max_map[df$chid])
  df$is_cheapest <- as.integer(df$inside == 1 & df$Price_num == df$price_min)
  df$is_dearest <- as.integer(df$inside == 1 & df$Price_num == df$price_max)
  df$price_gap_min <- ifelse(df$inside == 1, df$Price_num - df$price_min, 0)
  df$price_gap_max <- ifelse(df$inside == 1, df$price_max - df$Price_num, 0)
  df
}

m8trpg_formula <- function() {
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
  as.formula(paste("chosen ~", paste(c(attr_terms, price_terms, "d2", "d3", int_terms),
                                      collapse = " + "), "| 0"))
}

pat_long <- function() {
  paste0("^(", paste(c(attrs, "Price", "Ch"), collapse = "|"), ")([1-4])$")
}

to_long <- function(df) {
  df <- as.data.frame(df)
  pat <- pat_long()
  nm <- names(df)
  m <- regmatches(nm, regexec(pat, nm))
  value_cols <- nm[lengths(m) > 0]
  base_names <- unique(vapply(m[lengths(m) > 0], function(x) x[2], character(1)))
  id_cols <- setdiff(nm, value_cols)

  out <- do.call(rbind, lapply(1:4, function(a) {
    part <- df[, id_cols, drop = FALSE]
    for (b in base_names) {
      part[[b]] <- df[[paste0(b, a)]]
    }
    part$alt <- a
    part
  }))
  out <- out[order(out$No, out$alt), , drop = FALSE]
  out$chosen <- as.integer(out$Ch == 1)
  out$chid <- paste(out$Case, out$Task, sep = "_")
  out$d2 <- as.integer(out$alt == 2)
  out$d3 <- as.integer(out$alt == 3)
  out
}

bootstrap_case_means <- function(gain_by_case, n_boot = 100000L, seed = 4821L,
                                  chunk_size = 2000L) {
  set.seed(seed)
  n_case <- length(gain_by_case)
  result <- numeric(n_boot)
  starts <- seq.int(1L, n_boot, by = chunk_size)
  for (start in starts) {
    stop_at <- min(n_boot, start + chunk_size - 1L)
    n_this <- stop_at - start + 1L
    sampled <- matrix(sample(gain_by_case, n_case * n_this, replace = TRUE),
                       nrow = n_case, ncol = n_this)
    result[start:stop_at] <- colMeans(sampled)
  }
  result
}
