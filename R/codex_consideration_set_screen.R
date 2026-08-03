# Screen: a conjunctive price-screening consideration-set mixture on top of
# v14's own frozen predictions.
#
# Pre-registration: codex_consideration_set_preregister.md
#
# Run:
#   source("R/codex_consideration_set_screen.R")

options(stringsAsFactors = FALSE)

v14_dir <- file.path("data_processed", "codex_set_context_network")
output_dir <- file.path("data_processed", "codex_consideration_set")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

tau_fixed <- 1.0

sigmoid <- function(x) 1 / (1 + exp(-x))

logsumexp2 <- function(x, y) {
  hi <- pmax(x, y)
  hi + log1p(exp(-abs(x - y)))
}

validate_probability <- function(prediction) {
  prediction <- pmax(as.matrix(prediction), 1e-15)
  prediction / rowSums(prediction)
}

log_loss_matrix <- function(truth, prediction) {
  prediction <- validate_probability(prediction)
  -mean(rowSums(truth * log(pmax(prediction, 1e-15))))
}

## ---- data ----

v14 <- readRDS(file.path(v14_dir, "canonical_result.rds"))
pred_all <- validate_probability(v14$candidate_prediction)

train <- read.csv(file.path("csv files", "train.csv"))
train <- train[order(train$No), , drop = FALSE]
rownames(train) <- NULL
truth_all <- as.matrix(train[, paste0("Ch", 1:4), drop = FALSE])
stopifnot(nrow(train) == 21565L, all(rowSums(truth_all) == 1L))

split <- readRDS("data_processed/train_val_split.rds")
fitting_cases <- unique(split$train_wide_tr$Case)
validation_cases <- split$val_cases
stopifnot(length(intersect(fitting_cases, validation_cases)) == 0L)

fit_rows <- which(train$Case %in% fitting_cases)
val_rows <- which(train$Case %in% validation_cases)
stopifnot(length(fit_rows) == 908L * 19L, length(val_rows) == 227L * 19L)

income_scaler <- list(
  centre = mean(train$incomea[fit_rows]), scale = sd(train$incomea[fit_rows])
)
z_income_row <- (train$incomea - income_scaler$centre) / income_scaler$scale

price_matrix_all <- as.matrix(train[, paste0("Price", 1:3), drop = FALSE])
truth_idx_all <- max.col(truth_all, ties.method = "first")

## per-respondent (panel) precomputed quantities, training set only
fit_case_id <- train$Case[fit_rows]
fit_case_levels <- sort(unique(fit_case_id))
fit_income_by_case <- z_income_row[fit_rows][match(fit_case_levels, fit_case_id)]

logP0_row_fit <- log(pmax(pred_all[cbind(fit_rows, truth_idx_all[fit_rows])], 1e-15))
logP0_n_fit <- unname(tapply(logP0_row_fit, fit_case_id, sum)[as.character(fit_case_levels)])
stopifnot(!anyNA(logP0_n_fit))

screened_branch_logprob_at_truth <- function(theta, rows, case_id_rows) {
  a <- theta[["a"]]; b <- theta[["b"]]
  c_n <- a + b * z_income_row[rows]
  gate <- sigmoid((c_n - price_matrix_all[rows, , drop = FALSE]) / tau_fixed)
  numerator_inside <- gate * pred_all[rows, 1:3, drop = FALSE]
  numerator_optout <- pred_all[rows, 4]
  denom <- numerator_optout + rowSums(numerator_inside)
  ps_full <- cbind(numerator_inside, numerator_optout) / denom
  ps_full <- validate_probability(ps_full)
  log(pmax(ps_full[cbind(seq_along(rows), truth_idx_all[rows])], 1e-15))
}

panel_nll <- function(theta_vec) {
  theta <- c(a = theta_vec[[1]], b = theta_vec[[2]], alpha0 = theta_vec[[3]], alpha1 = theta_vec[[4]])
  logps_row <- screened_branch_logprob_at_truth(theta, fit_rows, fit_case_id)
  logps_n <- unname(tapply(logps_row, fit_case_id, sum)[as.character(fit_case_levels)])
  pi_n <- sigmoid(theta[["alpha0"]] + theta[["alpha1"]] * fit_income_by_case)
  mixture_loglik_n <- logsumexp2(log1p(-pi_n) + logP0_n_fit, log(pi_n) + logps_n)
  -mean(mixture_loglik_n)
}

## ---- correctness check: pi_n = 0 must exactly reproduce v14's own loss ----
baseline_val_loss <- log_loss_matrix(truth_all[val_rows, , drop = FALSE], pred_all[val_rows, , drop = FALSE])
baseline_fit_loss <- log_loss_matrix(truth_all[fit_rows, , drop = FALSE], pred_all[fit_rows, , drop = FALSE])
cat(sprintf("v14 baseline: fit-set logloss %.6f, validation logloss %.6f\n", baseline_fit_loss, baseline_val_loss))

check_theta <- c(a = 8, b = 0, alpha0 = -30, alpha1 = 0) # pi_n effectively 0 everywhere
check_nll <- panel_nll(check_theta)
check_target <- -mean(tapply(logP0_row_fit, fit_case_id, mean))
cat(sprintf(
  "Correctness check (pi=0): panel NLL %.6f vs. mean-per-task v14 NLL %.6f (informational; panel objective is a respondent-sum, not directly the row-mean log loss)\n",
  check_nll, -mean(logP0_row_fit)
))
stopifnot(abs(check_nll - (-mean(logP0_n_fit))) < 1e-8)
cat("Correctness check passed: pi=0 reproduces the frozen v14 panel likelihood exactly.\n\n")

## ---- optimize: multiple random restarts, Nelder-Mead (robust for a small,
## potentially non-convex mixture likelihood) ----
set.seed(4821)
n_restarts <- 12L
starts <- lapply(seq_len(n_restarts), function(i) {
  c(
    a = rnorm(1, 8, 2), b = rnorm(1, 0, 1),
    alpha0 = rnorm(1, -2, 1), alpha1 = rnorm(1, 0, 1)
  )
})
starts[[1]] <- c(a = 8, b = 0, alpha0 = -3, alpha1 = 0)

fits <- lapply(starts, function(start) {
  optim(par = start, fn = panel_nll, method = "Nelder-Mead", control = list(maxit = 2000, reltol = 1e-10))
})
values <- vapply(fits, function(f) f$value, numeric(1))
best <- fits[[which.min(values)]]
cat("Restart objective values:\n")
print(sort(values))
cat(sprintf("\nBest fit: a=%.4f b=%.4f alpha0=%.4f alpha1=%.4f, NLL=%.6f, convergence=%d\n",
            best$par[[1]], best$par[[2]], best$par[[3]], best$par[[4]], best$value, best$convergence))

theta_hat <- c(a = best$par[[1]], b = best$par[[2]], alpha0 = best$par[[3]], alpha1 = best$par[[4]])

## ---- held-out prediction: prior-weighted mixture using pi_n(z_n) only,
## never a posterior conditioned on the respondent's own held-out choices ----
predict_mixture <- function(theta, rows) {
  a <- theta[["a"]]; b <- theta[["b"]]
  c_n <- a + b * z_income_row[rows]
  gate <- sigmoid((c_n - price_matrix_all[rows, , drop = FALSE]) / tau_fixed)
  numerator_inside <- gate * pred_all[rows, 1:3, drop = FALSE]
  numerator_optout <- pred_all[rows, 4]
  denom <- numerator_optout + rowSums(numerator_inside)
  ps_full <- validate_probability(cbind(numerator_inside, numerator_optout) / denom)
  pi_row <- sigmoid(theta[["alpha0"]] + theta[["alpha1"]] * z_income_row[rows])
  p0_full <- pred_all[rows, , drop = FALSE]
  validate_probability((1 - pi_row) * p0_full + pi_row * ps_full)
}

val_prediction <- predict_mixture(theta_hat, val_rows)
val_loss <- log_loss_matrix(truth_all[val_rows, , drop = FALSE], val_prediction)
cat(sprintf("\nScreen: consideration-set mixture validation logloss %.6f vs. v14 baseline %.6f\n", val_loss, baseline_val_loss))
cat(sprintf("Screen gain (v14 - mixture): %.6f\n", baseline_val_loss - val_loss))

mean_pi_fit <- mean(sigmoid(theta_hat[["alpha0"]] + theta_hat[["alpha1"]] * fit_income_by_case))
cat(sprintf("Mean fitted screening-type probability (training respondents): %.4f\n", mean_pi_fit))
cat(sprintf("Fitted threshold at mean income (z=0): %.4f (price levels 1-12)\n", theta_hat[["a"]]))

saveRDS(
  list(theta_hat = theta_hat, tau_fixed = tau_fixed, restarts = values,
       baseline_val_loss = baseline_val_loss, val_loss = val_loss,
       mean_pi_fit = mean_pi_fit),
  file.path(output_dir, "screen_result.rds")
)
write.csv(
  data.frame(
    a = theta_hat[["a"]], b = theta_hat[["b"]], alpha0 = theta_hat[["alpha0"]], alpha1 = theta_hat[["alpha1"]],
    tau = tau_fixed, baseline_val_loss = baseline_val_loss, val_loss = val_loss,
    screen_gain = baseline_val_loss - val_loss, mean_pi_fit = mean_pi_fit
  ),
  file.path(output_dir, "screen_summary.csv"),
  row.names = FALSE
)
cat("\nScreen complete. Results in", output_dir, "\n")
