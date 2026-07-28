## Nested respondent-grouped CV for a questionnaire-version opt-out utility
## correction. For each outer fold, version sufficient statistics come only
## from inner-OOF predictions of the outer-training respondents.

source("R/codex_version_shrinkage_common.R")

derivative_check <- check_optout_newton_derivatives()
stopifnot(derivative_check$passed)

train <- read.csv("csv files/train.csv")
train <- train[order(train$No), , drop = FALSE]
truth <- truth_wide(train)
reference <- feature_reference(train)
version_map <- load_questionnaire_versions()
official <- current_fixed_oof(train)
official_pred <- official$pred
row_fold <- unname(
  official$fold_of_case[as.character(train$Case)]
)
stopifnot(
  !anyNA(row_fold),
  identical(sort(unique(as.integer(row_fold))), 1:5),
  all(tapply(row_fold, train$Case, function(x) length(unique(x))) == 1L)
)

lambda_key <- function(lambda) {
  if (is.infinite(lambda)) "Inf" else as.character(lambda)
}

load_or_fit <- function(path, source_wide, target_wide,
                        seed_offset, fit_label) {
  source_wide <- source_wide[order(source_wide$No), , drop = FALSE]
  target_wide <- target_wide[order(target_wide$No), , drop = FALSE]
  if (file.exists(path)) {
    fit <- readRDS(path)
    stopifnot(
      identical(fit$source_no, as.integer(source_wide$No)),
      identical(fit$target_no, as.integer(target_wide$No)),
      identical(fit$seed_offset, as.integer(seed_offset))
    )
    cat(sprintf("Loaded %s from cache.\n", fit_label))
    flush.console()
    return(fit)
  }
  cat(sprintf(
    "Fitting %s: %d source respondents, %d target respondents.\n",
    fit_label,
    length(unique(source_wide$Case)),
    length(unique(target_wide$Case))
  ))
  flush.console()
  fit <- fit_submitted_architecture(
    source_wide, target_wide, reference,
    seed_offset = seed_offset
  )
  saveRDS(fit, path)
  cat(sprintf(
    "Finished %s in %.1fs; target loss %.9f.\n",
    fit_label, fit$elapsed_seconds, fit$target_loss
  ))
  flush.console()
  fit
}

fresh_baseline_oof <- matrix(NA_real_, nrow(train), 4L)
fresh_corrected_oof <- matrix(NA_real_, nrow(train), 4L)
official_corrected_oof <- matrix(NA_real_, nrow(train), 4L)
fresh_peer_robust_oof <- matrix(NA_real_, nrow(train), 4L)
official_peer_robust_oof <- matrix(NA_real_, nrow(train), 4L)
lambda_curve_rows <- list()
outer_rows <- list()
delta_rows <- list()
heldout_peer_rows <- list()

for (outer_fold in 1:5) {
  cat(sprintf("\n===== OUTER FOLD %d =====\n", outer_fold))
  flush.console()
  outer_train_rows <- row_fold != outer_fold
  outer_valid_rows <- row_fold == outer_fold
  outer_train <- train[outer_train_rows, , drop = FALSE]
  outer_valid <- train[outer_valid_rows, , drop = FALSE]
  inner_oof <- matrix(
    NA_real_, nrow(outer_train), 4L,
    dimnames = list(as.character(outer_train$No), NULL)
  )

  for (inner_fold in setdiff(1:5, outer_fold)) {
    source_rows <- !row_fold %in% c(outer_fold, inner_fold)
    target_rows <- row_fold == inner_fold
    source_wide <- train[source_rows, , drop = FALSE]
    target_wide <- train[target_rows, , drop = FALSE]
    fit_label <- sprintf(
      "outer%d_inner%d", outer_fold, inner_fold
    )
    fit <- load_or_fit(
      file.path(
        version_output_dir, paste0(fit_label, "_base_fit.rds")
      ),
      source_wide, target_wide,
      seed_offset = outer_fold * 10000L + inner_fold * 100L,
      fit_label = fit_label
    )
    inner_oof[as.character(target_wide$No), ] <- fit$target_pred
  }
  stopifnot(!anyNA(inner_oof))

  lambda_evaluation <- evaluate_newton_lambda_grid(
    outer_train, inner_oof, version_map,
    lambda_grid = version_lambda_grid
  )
  lambda_curve <- lambda_evaluation$curve
  selected_index <- which.min(lambda_curve$corrected_loss)
  selected_lambda <- lambda_curve$lambda[[selected_index]]
  selected_key <- lambda_key(selected_lambda)
  lambda_curve$selected <- seq_len(nrow(lambda_curve)) == selected_index
  lambda_curve$outer_fold <- outer_fold
  lambda_curve_rows[[as.character(outer_fold)]] <- lambda_curve

  cat("Inner-OOF leave-respondent-out lambda curve:\n")
  print(lambda_curve)
  cat(sprintf("Selected lambda %s.\n", selected_key))
  flush.console()

  # Final version deltas use only outer-training respondents' inner-OOF
  # residuals. No in-sample base prediction enters this estimate.
  delta_table <- estimate_newton_delta(
    outer_train, inner_oof, version_map, selected_lambda
  )
  delta_table$outer_fold <- outer_fold
  delta_rows[[as.character(outer_fold)]] <- delta_table

  outer_label <- sprintf("outer%d_final", outer_fold)
  outer_fit <- load_or_fit(
    file.path(
      version_output_dir, paste0(outer_label, "_base_fit.rds")
    ),
    outer_train, outer_valid,
    seed_offset = outer_fold * 100L,
    fit_label = outer_label
  )
  fresh_baseline <- outer_fit$target_pred
  official_baseline <- official_pred[outer_valid_rows, , drop = FALSE]
  fresh_correction <- apply_optout_delta(
    fresh_baseline, outer_valid$Case,
    version_map, delta_table
  )
  official_correction <- apply_optout_delta(
    official_baseline, outer_valid$Case,
    version_map, delta_table
  )

  version <- case_version(outer_valid$Case, version_map)
  peer_count <- delta_table$n_respondent[
    match(version, delta_table$version_id)
  ]
  peer_count[is.na(peer_count)] <- 0L
  dominance <- delta_table$max_respondent_gradient_share[
    match(version, delta_table$version_id)
  ]
  dominance[is.na(dominance)] <- 0
  robust_delta <- fresh_correction$delta
  robust_delta[peer_count <= 1L] <- 0
  fresh_robust <- apply_optout_delta_vector(
    fresh_baseline, robust_delta
  )$pred
  official_robust <- apply_optout_delta_vector(
    official_baseline, robust_delta
  )$pred

  fresh_baseline_oof[outer_valid_rows, ] <- fresh_baseline
  fresh_corrected_oof[outer_valid_rows, ] <- fresh_correction$pred
  official_corrected_oof[outer_valid_rows, ] <-
    official_correction$pred
  fresh_peer_robust_oof[outer_valid_rows, ] <- fresh_robust
  official_peer_robust_oof[outer_valid_rows, ] <- official_robust

  heldout_case <- unique(data.frame(
    Case = outer_valid$Case,
    version_id = version,
    peer_count = peer_count,
    delta = fresh_correction$delta,
    max_respondent_gradient_share = dominance
  ))
  heldout_case$outer_fold <- outer_fold
  heldout_peer_rows[[as.character(outer_fold)]] <- heldout_case

  outer_truth <- truth[outer_valid_rows, , drop = FALSE]
  fresh_loss <- log_loss_matrix(outer_truth, fresh_baseline)
  fresh_corrected_loss <- log_loss_matrix(
    outer_truth, fresh_correction$pred
  )
  official_loss <- log_loss_matrix(outer_truth, official_baseline)
  official_corrected_loss <- log_loss_matrix(
    outer_truth, official_correction$pred
  )
  outer_rows[[as.character(outer_fold)]] <- data.frame(
    outer_fold = outer_fold,
    selected_lambda = selected_lambda,
    inner_baseline_loss = lambda_curve$baseline_loss[[1]],
    inner_selected_loss =
      lambda_curve$corrected_loss[[selected_index]],
    inner_gain = lambda_curve$gain[[selected_index]],
    fresh_baseline_loss = fresh_loss,
    fresh_corrected_loss = fresh_corrected_loss,
    fresh_gain = fresh_loss - fresh_corrected_loss,
    official_baseline_loss = official_loss,
    official_corrected_loss = official_corrected_loss,
    official_gain = official_loss - official_corrected_loss,
    fresh_peer_robust_loss = log_loss_matrix(
      outer_truth, fresh_robust
    ),
    official_peer_robust_loss = log_loss_matrix(
      outer_truth, official_robust
    ),
    validation_respondents = length(unique(outer_valid$Case)),
    zero_peer_respondents = sum(heldout_case$peer_count == 0L),
    one_peer_respondents = sum(heldout_case$peer_count == 1L),
    mean_peers = mean(heldout_case$peer_count),
    mean_abs_delta = mean(abs(heldout_case$delta)),
    max_abs_delta = max(abs(heldout_case$delta)),
    max_gradient_dominance = max(
      heldout_case$max_respondent_gradient_share
    ),
    outer_fit_elapsed_seconds = outer_fit$elapsed_seconds
  )
  cat("Outer-fold result:\n")
  print(outer_rows[[as.character(outer_fold)]])
  flush.console()

  saveRDS(
    list(
      completed_outer_folds = outer_fold,
      fresh_baseline_oof = fresh_baseline_oof,
      fresh_corrected_oof = fresh_corrected_oof,
      official_baseline_oof = official_pred,
      official_corrected_oof = official_corrected_oof,
      fresh_peer_robust_oof = fresh_peer_robust_oof,
      official_peer_robust_oof = official_peer_robust_oof,
      lambda_curves = do.call(rbind, lambda_curve_rows),
      outer_results = do.call(rbind, outer_rows),
      delta_tables = do.call(rbind, delta_rows),
      heldout_peers = do.call(rbind, heldout_peer_rows),
      truth = truth,
      case = train$Case,
      no = train$No,
      row_fold = row_fold
    ),
    file.path(version_output_dir, "newton_cv_partial.rds")
  )
}

stopifnot(
  !anyNA(fresh_baseline_oof),
  !anyNA(fresh_corrected_oof),
  !anyNA(official_corrected_oof),
  !anyNA(fresh_peer_robust_oof),
  !anyNA(official_peer_robust_oof)
)
lambda_curves <- do.call(rbind, lambda_curve_rows)
outer_results <- do.call(rbind, outer_rows)
delta_tables <- do.call(rbind, delta_rows)
heldout_peers <- do.call(rbind, heldout_peer_rows)

overall <- data.frame(
  model = c("fresh_refit", "official_fixed15"),
  baseline_loss = c(
    log_loss_matrix(truth, fresh_baseline_oof),
    log_loss_matrix(truth, official_pred)
  ),
  corrected_loss = c(
    log_loss_matrix(truth, fresh_corrected_oof),
    log_loss_matrix(truth, official_corrected_oof)
  ),
  gain = c(
    log_loss_matrix(truth, fresh_baseline_oof) -
      log_loss_matrix(truth, fresh_corrected_oof),
    log_loss_matrix(truth, official_pred) -
      log_loss_matrix(truth, official_corrected_oof)
  ),
  peer_robust_loss = c(
    log_loss_matrix(truth, fresh_peer_robust_oof),
    log_loss_matrix(truth, official_peer_robust_oof)
  ),
  peer_robust_gain = c(
    log_loss_matrix(truth, fresh_baseline_oof) -
      log_loss_matrix(truth, fresh_peer_robust_oof),
    log_loss_matrix(truth, official_pred) -
      log_loss_matrix(truth, official_peer_robust_oof)
  )
)
bootstrap <- rbind(
  cbind(
    model = "fresh_refit",
    bootstrap_case_gain(
      truth, fresh_baseline_oof, fresh_corrected_oof,
      train$Case, replicates = 100000L, seed = 4821L
    )
  ),
  cbind(
    model = "official_fixed15",
    bootstrap_case_gain(
      truth, official_pred, official_corrected_oof,
      train$Case, replicates = 100000L, seed = 4821L
    )
  ),
  cbind(
    model = "fresh_refit_peer_robust",
    bootstrap_case_gain(
      truth, fresh_baseline_oof, fresh_peer_robust_oof,
      train$Case, replicates = 100000L, seed = 4821L
    )
  ),
  cbind(
    model = "official_fixed15_peer_robust",
    bootstrap_case_gain(
      truth, official_pred, official_peer_robust_oof,
      train$Case, replicates = 100000L, seed = 4821L
    )
  )
)

write.csv(
  lambda_curves,
  file.path(version_output_dir, "newton_lambda_curves.csv"),
  row.names = FALSE
)
write.csv(
  outer_results,
  file.path(version_output_dir, "newton_outer_results.csv"),
  row.names = FALSE
)
write.csv(
  delta_tables,
  file.path(version_output_dir, "newton_version_deltas.csv"),
  row.names = FALSE
)
write.csv(
  heldout_peers,
  file.path(version_output_dir, "newton_heldout_peers.csv"),
  row.names = FALSE
)
write.csv(
  overall,
  file.path(version_output_dir, "newton_overall.csv"),
  row.names = FALSE
)
write.csv(
  bootstrap,
  file.path(version_output_dir, "newton_bootstrap.csv"),
  row.names = FALSE
)
saveRDS(
  list(
    derivative_check = derivative_check,
    fresh_baseline_oof = fresh_baseline_oof,
    fresh_corrected_oof = fresh_corrected_oof,
    official_baseline_oof = official_pred,
    official_corrected_oof = official_corrected_oof,
    fresh_peer_robust_oof = fresh_peer_robust_oof,
    official_peer_robust_oof = official_peer_robust_oof,
    lambda_curves = lambda_curves,
    outer_results = outer_results,
    delta_tables = delta_tables,
    heldout_peers = heldout_peers,
    overall = overall,
    bootstrap = bootstrap,
    truth = truth,
    case = train$Case,
    no = train$No,
    row_fold = row_fold,
    lambda_grid = version_lambda_grid
  ),
  file.path(version_output_dir, "newton_cv_result.rds")
)

cat("\n===== FINAL NEWTON VERSION-CV RESULT =====\n")
print(derivative_check)
print(overall)
cat("\nRespondent-clustered bootstrap:\n")
print(bootstrap)
