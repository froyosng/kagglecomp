## Shared functions for the neighbor-pooled (kernel-smoothed) questionnaire-version
## opt-out correction. See codex_version_pool_preregister.md for the full spec this
## code implements. This script defines functions only; it does not run anything on
## source() (no side effects besides making functions available).
##
## Design note: this deliberately does NOT re-fit the frozen v11+MLP architecture.
## It reuses the already-audited nested inner-OOF / outer-refit prediction matrices
## produced by the rejected codex-version-shrinkage experiment
## (data_processed/codex_version_shrinkage/outer{f}_{inner{g},final}_base_fit.rds),
## copied read-only into this worktree. Only the correction estimator itself (the
## similarity graph + kernel-pooled Newton step) is new.

attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
           "PP","KA","SC","TS","NV","MA","LB","AF","HU")
attrs_plus_price <- c(attrs, "Price")
n_version_global <- 299L

log_loss_matrix <- function(actual, pred, eps = 1e-15) {
  actual <- as.matrix(actual)
  pred <- as.matrix(pred)
  pred <- pred / rowSums(pred)
  pred <- pmin(pmax(pred, eps), 1 - eps)
  -mean(rowSums(actual * log(pred)))
}

normalize_probability <- function(prediction) {
  prediction <- as.matrix(prediction)
  prediction <- prediction / rowSums(prediction)
  stopifnot(ncol(prediction) == 4L, all(is.finite(prediction)), all(prediction > 0))
  prediction
}

## ---- version map + design-feature similarity (Section 3 of the preregister) ----

load_version_map <- function() {
  artifact <- readRDS("data_processed/questionnaire_fingerprints.rds")
  stopifnot(
    identical(as.integer(artifact$n_unique), n_version_global),
    length(artifact$case_ids) == length(artifact$fingerprints),
    !anyDuplicated(artifact$case_ids),
    identical(as.integer(artifact$case_ids), seq_len(1398L))
  )
  version_id <- match(artifact$fingerprints, unique(artifact$fingerprints))
  stopifnot(length(unique(version_id)) == n_version_global)
  data.frame(Case = as.integer(artifact$case_ids), version_id = as.integer(version_id))
}

## Build the 299 x 20 design-only feature matrix: for each version, the mean level
## of each of the 20 design columns (19 attributes + Price) across that version's
## 19 tasks x 3 real alternatives (57 cells), using ONE representative respondent
## per version (every respondent sharing a version has, by the fingerprint's own
## construction, an identical design). Never touches Ch1..Ch4.
build_version_design_matrix <- function(version_map) {
  attr_cols_alt123 <- unlist(lapply(attrs_plus_price, function(a) paste0(a, 1:3)))
  train <- read.csv("csv files/train.csv")
  test <- read.csv("csv files/test.csv")
  keep <- c("Case", "Task", attr_cols_alt123)
  all_small <- rbind(train[, keep], test[, keep])

  representative_case <- vapply(seq_len(n_version_global), function(v) {
    min(version_map$Case[version_map$version_id == v])
  }, integer(1))

  design_matrix <- t(vapply(seq_len(n_version_global), function(v) {
    cid <- representative_case[v]
    rows <- all_small[all_small$Case == cid, attr_cols_alt123, drop = FALSE]
    stopifnot(nrow(rows) == 19L)
    vapply(attrs_plus_price, function(a) {
      mean(as.numeric(as.matrix(rows[, paste0(a, 1:3), drop = FALSE])))
    }, numeric(1))
  }, numeric(length(attrs_plus_price))))
  rownames(design_matrix) <- as.character(seq_len(n_version_global))
  colnames(design_matrix) <- attrs_plus_price
  design_matrix
}

standardize_columns <- function(design_matrix) {
  scale(design_matrix)
}

## k-nearest-neighbor Gaussian kernel weight matrix. W[v, v] = 1 always;
## k = 0 returns the identity (no pooling at all -- exact reproduction of the
## rejected per-version-only Newton estimator).
build_weight_matrix <- function(design_std, k) {
  n <- nrow(design_std)
  if (k == 0L) return(diag(n))
  d <- as.matrix(dist(design_std, method = "euclidean"))
  W <- matrix(0, n, n)
  for (v in seq_len(n)) {
    dv <- d[v, ]
    dv[v] <- Inf
    nn <- order(dv)[seq_len(k)]
    bw <- max(dv[nn])
    if (!is.finite(bw) || bw <= 0) bw <- 1e-8
    W[v, nn] <- exp(-0.5 * (dv[nn] / bw)^2)
  }
  diag(W) <- 1
  W
}

## ---- pooled Newton statistics (Section 4) ----

version_group_sums <- function(version_id_of_row, values, n_version = n_version_global) {
  out <- numeric(n_version)
  agg <- tapply(values, version_id_of_row, sum)
  idx <- as.integer(names(agg))
  out[idx] <- as.numeric(agg)
  out
}

## df must have columns: Case, version_id, g, h (one row per task).
## Returns one delta per DISTINCT Case in df (leave-that-respondent's-own-
## contribution-out, per Section 4's exclusion rule). Uses the identity
## W[v,v] == 1 to simplify: G_excluding_i = (W %*% total_g)[v(i)] - g_i.
loo_delta_by_respondent <- function(df, W, lambda, n_version = n_version_global) {
  total_g <- version_group_sums(df$version_id, df$g, n_version)
  total_h <- version_group_sums(df$version_id, df$h, n_version)
  G <- as.vector(W %*% total_g)
  H <- as.vector(W %*% total_h)

  resp_g <- tapply(df$g, df$Case, sum)
  resp_h <- tapply(df$h, df$Case, sum)
  resp_case <- as.integer(names(resp_g))
  resp_version <- df$version_id[match(resp_case, df$Case)]

  G_excl <- G[resp_version] - as.numeric(resp_g)
  H_excl <- H[resp_version] - as.numeric(resp_h)
  delta <- if (is.infinite(lambda)) {
    rep(0, length(resp_case))
  } else {
    -G_excl / (H_excl + lambda)
  }
  data.frame(Case = resp_case, delta = delta)
}

## Final (non-leave-one-out) per-version delta table, for applying to a fully
## disjoint outer-holdout fold. No exclusion needed: none of the holdout
## respondents contributed to any g/h statistic here.
final_delta_table <- function(df, W, lambda, n_version = n_version_global) {
  total_g <- version_group_sums(df$version_id, df$g, n_version)
  total_h <- version_group_sums(df$version_id, df$h, n_version)
  G <- as.vector(W %*% total_g)
  H <- as.vector(W %*% total_h)
  delta <- if (is.infinite(lambda)) rep(0, n_version) else -G / (H + lambda)
  data.frame(version_id = seq_len(n_version), delta = delta)
}

apply_optout_delta_vector <- function(prediction, delta) {
  prediction <- normalize_probability(prediction)
  stopifnot(length(delta) == nrow(prediction), !anyNA(delta))
  multiplier <- exp(delta)
  denom <- rowSums(prediction[, 1:3, drop = FALSE]) + prediction[, 4] * multiplier
  adjusted <- prediction
  adjusted[, 1:3] <- prediction[, 1:3, drop = FALSE] / denom
  adjusted[, 4] <- prediction[, 4] * multiplier / denom
  normalize_probability(adjusted)
}

## ---- baseline artifacts ----

truth_wide <- function(wide) {
  wide <- wide[order(wide$No), , drop = FALSE]
  truth <- as.matrix(wide[, paste0("Ch", 1:4), drop = FALSE])
  storage.mode(truth) <- "double"
  stopifnot(all(rowSums(truth) == 1))
  truth
}

## Exact primary baseline: 0.85*(0.8*mlogit_OOF + 0.2*xgb_OOF) + 0.15*mlp_OOF,
## asserted CV log loss 1.143686618134879 ("ensemble_v11+MLP", current best).
current_fixed_oof <- function() {
  train <- read.csv("csv files/train.csv")
  train <- train[order(train$No), , drop = FALSE]
  ensemble <- readRDS("data_processed/oof_ensemble_v10.rds")
  mlp <- readRDS("data_processed/codex_behavioral_round/mlp_oof.rds")
  mlp_prediction <- mlp$oof[["h08_d0.100"]]
  stopifnot(
    identical(dim(ensemble$oof_mlogit), c(nrow(train), 4L)),
    identical(dim(mlp_prediction), c(nrow(train), 4L)),
    max(abs(ensemble$oof_truth - truth_wide(train))) < 1e-12
  )
  v11 <- 0.8 * ensemble$oof_mlogit + 0.2 * ensemble$oof_xgb
  current <- 0.85 * v11 + 0.15 * mlp_prediction
  current <- normalize_probability(current)
  baseline_loss <- log_loss_matrix(truth_wide(train), current)
  stopifnot(abs(baseline_loss - 1.143686618134879) < 1e-10)
  list(
    pred = current,
    no = as.integer(train$No),
    case = as.integer(train$Case),
    fold_of_case = ensemble$fold_of_case,
    truth = truth_wide(train),
    baseline_loss = baseline_loss
  )
}

## ---- respondent-clustered paired bootstrap (Section 6) ----

case_mean_gain <- function(truth, baseline, candidate, case) {
  baseline_row <- -rowSums(truth * log(pmin(pmax(normalize_probability(baseline), 1e-15), 1)))
  candidate_row <- -rowSums(truth * log(pmin(pmax(normalize_probability(candidate), 1e-15), 1)))
  tapply(baseline_row - candidate_row, case, mean)
}

bootstrap_case_gain <- function(truth, baseline, candidate, case,
                                 replicates = 100000L, seed = 4821L) {
  gain <- case_mean_gain(truth, baseline, candidate, case)
  set.seed(seed)
  bootstrap <- replicate(as.integer(replicates), mean(sample(gain, length(gain), replace = TRUE)))
  data.frame(
    point_gain = mean(gain),
    bootstrap_mean = mean(bootstrap),
    bootstrap_sd = sd(bootstrap),
    lower_95 = unname(quantile(bootstrap, 0.025)),
    upper_95 = unname(quantile(bootstrap, 0.975)),
    lower_99 = unname(quantile(bootstrap, 0.005)),
    upper_99 = unname(quantile(bootstrap, 0.995)),
    win_rate = mean(bootstrap > 0),
    replicates = as.integer(replicates)
  )
}

## ---- loading cached nested base-model artifacts ----

load_cached_fit <- function(path) {
  obj <- readRDS(path)
  stopifnot(all(c("source_pred","target_pred","source_no","target_no",
                  "source_case","target_case") %in% names(obj)))
  obj
}

## Build the per-row (Case, version_id, y4, p4, g, h) frame needed for the
## Newton statistics, for a cached fit's TARGET portion.
newton_rows_from_fit <- function(fit, truth_lookup, version_map) {
  no <- as.integer(fit$target_no)
  case <- as.integer(fit$target_case)
  pred <- normalize_probability(fit$target_pred)
  y4 <- truth_lookup$truth[match(no, truth_lookup$no), 4]
  stopifnot(!anyNA(y4))
  version_id <- version_map$version_id[match(case, version_map$Case)]
  stopifnot(!anyNA(version_id))
  p4 <- pred[, 4]
  data.frame(
    No = no, Case = case, version_id = version_id,
    y4 = y4, p4 = p4, g = p4 - y4, h = p4 * (1 - p4)
  )
}
