## Shared machinery for the pre-registered Bayesian hierarchical mixed logit
## experiment (codex_bayes_mixed_preregister.md). Implements:
##  - the 23-column continuous+ASC design matrix (19 attrs + Price + ASC2/3/4)
##  - per-fold bayesm::rhierMnlRwMixture fitting (2 chains)
##  - POPULATION-LEVEL posterior-predictive scoring for held-out respondents
##    (the specific correctness requirement of this track)
##  - baseline (ensemble_v11 + MLP) reconstruction from raw cached OOF
##  - respondent-clustered paired bootstrap

suppressPackageStartupMessages({
  library(bayesm)
})

attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
           "PP","KA","SC","TS","NV","MA","LB","AF","HU")
bayes_vars <- c(attrs, "Price")   # 20 continuous columns
bayes_nvar <- length(bayes_vars) + 3L  # + ASC2, ASC3, ASC4 = 23

log_loss_matrix <- function(actual, pred, eps = 1e-15) {
  actual <- as.matrix(actual)
  pred <- as.matrix(pred)
  pred <- pred / rowSums(pred)
  pred <- pmin(pmax(pred, eps), 1 - eps)
  -mean(rowSums(actual * log(pred)))
}

## ---- Design matrix / lgtdata construction --------------------------------

bayes_scaler <- function(wide) {
  long_vals <- do.call(cbind, lapply(bayes_vars, function(v) {
    unlist(wide[, paste0(v, 1:4)])
  }))
  ctr <- colMeans(long_vals)
  scl <- apply(long_vals, 2, sd)
  scl[scl == 0] <- 1
  names(ctr) <- bayes_vars
  names(scl) <- bayes_vars
  list(ctr = ctr, scl = scl)
}

## One respondent's design matrix: 76 x 23 (19 tasks x 4 alts, ordered by
## Task then alt 1..4), columns = z(attrs), z(Price), ASC2, ASC3, ASC4.
respondent_design <- function(resp_wide_row_block, ctr, scl) {
  resp <- resp_wide_row_block[order(resp_wide_row_block$Task), , drop = FALSE]
  ntask <- nrow(resp)
  X <- matrix(0, nrow = ntask * 4L, ncol = bayes_nvar)
  colnames(X) <- c(paste0("z_", bayes_vars), "ASC2", "ASC3", "ASC4")
  for (t in seq_len(ntask)) {
    row <- resp[t, ]
    for (a in 1:4) {
      r <- (t - 1L) * 4L + a
      raw <- as.numeric(row[paste0(bayes_vars, a)])
      X[r, seq_along(bayes_vars)] <- (raw - ctr) / scl
      if (a == 2L) X[r, "ASC2"] <- 1
      if (a == 3L) X[r, "ASC3"] <- 1
      if (a == 4L) X[r, "ASC4"] <- 1
    }
  }
  X
}

## lgtdata for a set of respondents (bayesm's per-unit list format).
build_lgtdata <- function(wide, cases, ctr, scl) {
  wide <- wide[wide$Case %in% cases, , drop = FALSE]
  by_case <- split(wide, wide$Case)
  # preserve the requested case order
  by_case <- by_case[as.character(cases)]
  lapply(by_case, function(resp) {
    resp <- resp[order(resp$Task), , drop = FALSE]
    y <- vapply(seq_len(nrow(resp)), function(t) {
      row <- resp[t, ]
      which(c(row$Ch1, row$Ch2, row$Ch3, row$Ch4) == 1L)
    }, integer(1))
    list(y = y, X = respondent_design(resp, ctr, scl))
  })
}

## ---- Fitting --------------------------------------------------------------

bayes_prior <- function(nvar = bayes_nvar) {
  list(
    ncomp = 1,
    nu = nvar + 3,
    V = (nvar + 3) * diag(nvar),
    Amu = 0.5,
    mubar = rep(0, nvar),
    Ad = 0.01 * diag(nvar),
    deltabar = rep(0, nvar),
    a = 5
  )
}

## Fit one chain; returns the pooled (post-burn-in) list of ncomp=1 mixture
## components (each a list(mu=, rooti=)) plus diagnostics. Only ever exposes
## the population-level nmix$compdraw -- callers must never reach into
## out$betadraw for held-out prediction (enforced by not even returning it).
fit_bayes_chain <- function(lgtdata, seed, R = 8000L, keep = 8L,
                            burn_frac = 0.25, nprint = 0) {
  set.seed(seed)
  Data1 <- list(p = 4, lgtdata = lgtdata)
  Prior1 <- bayes_prior()
  Mcmc1 <- list(R = R, keep = keep, nprint = nprint)
  out <- rhierMnlRwMixture(Data = Data1, Prior = Prior1, Mcmc = Mcmc1)
  n_kept <- length(out$nmix$compdraw)
  burn <- floor(n_kept * burn_frac)
  keep_idx <- (burn + 1L):n_kept
  list(
    compdraw = out$nmix$compdraw[keep_idx],
    Deltadraw = out$Deltadraw[keep_idx, , drop = FALSE],
    loglike = out$loglike[keep_idx],
    n_kept = length(keep_idx),
    seed = seed
  )
}

## Fit 2 chains for one fold's training respondents; returns the pooled list
## of post-burn-in components plus a between-chain convergence summary.
fit_bayes_fold <- function(lgtdata, fold_seed_offset,
                           R = 8000L, keep = 8L) {
  chain1 <- fit_bayes_chain(lgtdata, seed = 4821L + fold_seed_offset,
                             R = R, keep = keep)
  chain2 <- fit_bayes_chain(lgtdata, seed = 9001L + fold_seed_offset,
                             R = R, keep = keep)

  chain_summary <- function(chain) {
    mus <- t(vapply(chain$compdraw, function(d) d[[1]]$mu, numeric(bayes_nvar)))
    sigmas <- vapply(chain$compdraw, function(d) {
      rooti <- d[[1]]$rooti
      root <- backsolve(rooti, diag(bayes_nvar))
      Sigma <- t(root) %*% root
      diag(Sigma)
    }, numeric(bayes_nvar))
    list(mu_mean = colMeans(mus), sigma_diag_mean = rowMeans(sigmas))
  }
  s1 <- chain_summary(chain1)
  s2 <- chain_summary(chain2)
  price_idx <- which(colnames(NULL) %in% NA)  # placeholder, overwritten below
  var_names <- c(paste0("z_", bayes_vars), "ASC2", "ASC3", "ASC4")
  price_i <- match("z_Price", var_names)
  asc4_i <- match("ASC4", var_names)
  convergence <- data.frame(
    term = c("mu_Price", "mu_ASC4", "sigma_Price", "sigma_ASC4"),
    chain1 = c(s1$mu_mean[price_i], s1$mu_mean[asc4_i],
               s1$sigma_diag_mean[price_i], s1$sigma_diag_mean[asc4_i]),
    chain2 = c(s2$mu_mean[price_i], s2$mu_mean[asc4_i],
               s2$sigma_diag_mean[price_i], s2$sigma_diag_mean[asc4_i])
  )
  convergence$abs_diff <- abs(convergence$chain1 - convergence$chain2)

  list(
    compdraw = c(chain1$compdraw, chain2$compdraw),
    convergence = convergence,
    var_names = var_names,
    loglike1 = chain1$loglike,
    loglike2 = chain2$loglike
  )
}

## ---- Population-level posterior-predictive scoring (THE key mechanism) ---

## Predict held-out (NEW / never-in-lgtdata) respondents' 19-task choice
## probabilities using ONLY the population-level draws (compdraw list of
## list(mu, rooti)), never any respondent-specific fitted value. For every
## pooled draw, every held-out respondent gets an INDEPENDENT fresh draw from
## N(mu_r, Sigma_r) via bayesm's own rmixture() utility.
population_predict <- function(compdraw, wide_heldout, cases, ctr, scl,
                               seed = 4821L) {
  set.seed(seed)
  n_case <- length(cases)
  by_case <- split(wide_heldout[wide_heldout$Case %in% cases, , drop = FALSE],
                    wide_heldout$Case[wide_heldout$Case %in% cases])
  by_case <- by_case[as.character(cases)]

  X_list <- lapply(by_case, function(resp) {
    respondent_design(resp[order(resp$Task), , drop = FALSE], ctr, scl)
  })
  n_rows_each <- vapply(X_list, nrow, integer(1))
  stopifnot(all(n_rows_each == 76L))
  X_stack <- do.call(rbind, X_list)
  resp_row_index <- rep(seq_len(n_case), each = 76L)

  n_draws <- length(compdraw)
  n_task_rows <- nrow(X_stack) / 4L
  prob_sum <- matrix(0, n_task_rows, 4L)
  for (r in seq_len(n_draws)) {
    # One FRESH population draw per held-out respondent for this posterior
    # sample -- never a respondent's own fitted value (there isn't one; these
    # respondents were never in lgtdata), never shared across respondents.
    beta_new <- rmixture(n_case, 1, list(compdraw[[r]][[1]]))$x
    beta_expanded <- beta_new[resp_row_index, , drop = FALSE]
    eta <- rowSums(X_stack * beta_expanded)
    z <- matrix(eta, ncol = 4L, byrow = TRUE)
    z <- z - apply(z, 1, max)
    ez <- exp(z)
    prob <- ez / rowSums(ez)
    prob_sum <- prob_sum + prob
  }
  prediction <- prob_sum / n_draws
  list(prediction = prediction, cases = cases, n_draws = n_draws)
}

## ---- Baseline (ensemble_v11 + MLP) reconstruction -------------------------

## `row_fold` must be supplied by the caller: an integer vector, one entry per
## row of `oof_ensemble_v10.rds`'s truth/OOF matrices (train.csv row order),
## giving each row's canonical fold (built from `fold_of_case` keyed by Case).
reconstruct_current_best <- function(row_fold) {
  saved <- readRDS("data_processed/oof_ensemble_v10.rds")
  mlp <- readRDS("data_processed/codex_behavioral_round/mlp_oof.rds")
  truth <- saved$oof_truth
  fold_of_case <- saved$fold_of_case
  mlogit_oof <- saved$oof_mlogit
  xgb_oof <- saved$oof_xgb
  stopifnot(length(row_fold) == nrow(truth))
  stopifnot(abs(log_loss_matrix(truth, mlogit_oof) - 1.147021) < 1e-5)
  stopifnot(abs(log_loss_matrix(truth, xgb_oof) - 1.178668) < 1e-5)
  v11 <- 0.8 * mlogit_oof + 0.2 * xgb_oof
  stopifnot(abs(log_loss_matrix(truth, v11) - 1.145094) < 1e-5)

  mlp_pred <- mlp$oof[["h08_d0.100"]]
  stopifnot(!is.null(mlp_pred))

  weight_grid <- seq(0, 0.30, by = 0.01)
  crossfit <- matrix(NA_real_, nrow(truth), 4L)
  per_fold_weight <- numeric(5L)
  for (fold in 1:5) {
    train_rows <- row_fold != fold
    valid_rows <- row_fold == fold
    losses <- vapply(weight_grid, function(w) {
      log_loss_matrix(truth[train_rows, ],
                       (1 - w) * v11[train_rows, ] + w * mlp_pred[train_rows, ])
    }, numeric(1))
    best_w <- weight_grid[[which.min(losses)]]
    per_fold_weight[fold] <- best_w
    crossfit[valid_rows, ] <-
      (1 - best_w) * v11[valid_rows, ] + best_w * mlp_pred[valid_rows, ]
  }
  loss <- log_loss_matrix(truth, crossfit)
  stopifnot(abs(loss - 1.143789) < 5e-5)
  list(
    prediction = crossfit,
    truth = truth,
    fold_of_case = fold_of_case,
    per_fold_mlp_weight = per_fold_weight,
    logloss = loss
  )
}

## ---- Respondent-clustered paired bootstrap --------------------------------

respondent_bootstrap_gain <- function(truth, baseline, candidate, case,
                                      seed = 4821L, n_boot = 100000L) {
  row_loss <- function(pred) {
    pred <- pmin(pmax(pred / rowSums(pred), 1e-15), 1 - 1e-15)
    -rowSums(truth * log(pred))
  }
  respondent_gain <- tapply(row_loss(baseline) - row_loss(candidate), case, mean)
  n_case <- length(respondent_gain)
  set.seed(seed)
  boot <- numeric(n_boot)
  for (start in seq.int(1L, n_boot, by = 2000L)) {
    stop_at <- min(n_boot, start + 1999L)
    n_this <- stop_at - start + 1L
    sampled <- matrix(sample.int(n_case, n_case * n_this, replace = TRUE),
                       nrow = n_case)
    boot[start:stop_at] <- colMeans(matrix(respondent_gain[sampled], nrow = n_case))
  }
  list(
    summary = data.frame(
      point_gain = mean(respondent_gain),
      bootstrap_mean = mean(boot),
      bootstrap_sd = sd(boot),
      lower_95 = unname(quantile(boot, 0.025)),
      upper_95 = unname(quantile(boot, 0.975)),
      win_rate = mean(boot > 0),
      n_boot = n_boot
    ),
    respondent_gain = respondent_gain,
    bootstrap = boot
  )
}
