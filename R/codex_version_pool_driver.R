## Nested-CV driver for the neighbor-pooled version correction. Parametrized by
## (k_grid, lambda_grid) so the identical code path serves both the k=0 smoke
## test (Section 7 of the preregister) and the full run (Section 5).

source("R/codex_version_pool_common.R")

run_version_pool_cv <- function(k_grid, lambda_grid, out_dir = "data_processed/codex_version_pool",
                                 tag = "run") {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  message(sprintf("[%s] k_grid = %s", tag, paste(k_grid, collapse = ",")))
  message(sprintf("[%s] lambda_grid = %s", tag, paste(lambda_grid, collapse = ",")))

  version_map <- load_version_map()
  design_raw <- build_version_design_matrix(version_map)
  design_std <- standardize_columns(design_raw)

  W_list <- lapply(k_grid, build_weight_matrix, design_std = design_std)
  names(W_list) <- as.character(k_grid)

  baseline <- current_fixed_oof()
  truth_lookup <- list(no = baseline$no, truth = baseline$truth)
  fold_of_case <- baseline$fold_of_case

  fold_rows <- vector("list", 5L)
  fold_summary <- vector("list", 5L)
  grid_records <- vector("list", 5L)

  for (f in 1:5) {
    holdout_case <- which(fold_of_case == f)
    inner_ids <- setdiff(1:5, f)

    inner_frames <- lapply(inner_ids, function(ii) {
      path <- sprintf("data_processed/codex_version_shrinkage/outer%d_inner%d_base_fit.rds", f, ii)
      fit <- load_cached_fit(path)
      list(
        rows = newton_rows_from_fit(fit, truth_lookup, version_map),
        pred = normalize_probability(fit$target_pred)
      )
    })
    inner_oof_df <- do.call(rbind, lapply(inner_frames, `[[`, "rows"))
    inner_pred <- do.call(rbind, lapply(inner_frames, `[[`, "pred"))
    stopifnot(nrow(inner_oof_df) == nrow(inner_pred))
    inner_truth <- truth_lookup$truth[match(inner_oof_df$No, truth_lookup$no), , drop = FALSE]
    stopifnot(!anyNA(inner_truth))

    baseline_inner_loss <- log_loss_matrix(inner_truth, inner_pred)

    grid <- expand.grid(k = k_grid, lambda = lambda_grid)
    grid$inner_loss <- NA_real_
    for (row_i in seq_len(nrow(grid))) {
      k_val <- grid$k[row_i]
      lambda_val <- grid$lambda[row_i]
      W <- W_list[[as.character(k_val)]]
      delta_df <- loo_delta_by_respondent(inner_oof_df, W, lambda_val)
      delta_row <- delta_df$delta[match(inner_oof_df$Case, delta_df$Case)]
      stopifnot(!anyNA(delta_row))
      corrected <- apply_optout_delta_vector(inner_pred, delta_row)
      grid$inner_loss[row_i] <- log_loss_matrix(inner_truth, corrected)
    }
    grid$outer_fold <- f
    grid$baseline_inner_loss <- baseline_inner_loss
    grid_records[[f]] <- grid

    best_i <- which.min(grid$inner_loss)
    k_star <- grid$k[best_i]
    lambda_star <- grid$lambda[best_i]

    W_star <- W_list[[as.character(k_star)]]
    final_tbl <- final_delta_table(inner_oof_df, W_star, lambda_star)

    ## Primary baseline: exact fixed submitted OOF, this fold's holdout rows.
    primary_rows <- which(baseline$case %in% holdout_case)
    primary_pred <- baseline$pred[primary_rows, , drop = FALSE]
    primary_no <- baseline$no[primary_rows]
    primary_case <- baseline$case[primary_rows]
    primary_version <- version_map$version_id[match(primary_case, version_map$Case)]
    primary_delta <- final_tbl$delta[match(primary_version, final_tbl$version_id)]
    stopifnot(!anyNA(primary_delta))
    primary_corrected <- apply_optout_delta_vector(primary_pred, primary_delta)
    primary_truth <- truth_lookup$truth[match(primary_no, truth_lookup$no), , drop = FALSE]
    primary_baseline_loss <- log_loss_matrix(primary_truth, primary_pred)
    primary_corrected_loss <- log_loss_matrix(primary_truth, primary_corrected)

    ## Secondary baseline: freshly-refit final_base_fit target predictions.
    final_fit <- load_cached_fit(sprintf(
      "data_processed/codex_version_shrinkage/outer%d_final_base_fit.rds", f
    ))
    secondary_no <- as.integer(final_fit$target_no)
    secondary_case <- as.integer(final_fit$target_case)
    secondary_pred <- normalize_probability(final_fit$target_pred)
    secondary_version <- version_map$version_id[match(secondary_case, version_map$Case)]
    secondary_delta <- final_tbl$delta[match(secondary_version, final_tbl$version_id)]
    stopifnot(!anyNA(secondary_delta))
    secondary_corrected <- apply_optout_delta_vector(secondary_pred, secondary_delta)
    secondary_truth <- truth_lookup$truth[match(secondary_no, truth_lookup$no), , drop = FALSE]
    secondary_baseline_loss <- log_loss_matrix(secondary_truth, secondary_pred)
    secondary_corrected_loss <- log_loss_matrix(secondary_truth, secondary_corrected)

    fold_rows[[f]] <- list(
      primary_no = primary_no, primary_corrected = primary_corrected,
      secondary_no = secondary_no, secondary_corrected = secondary_corrected,
      secondary_pred = secondary_pred
    )
    fold_summary[[f]] <- data.frame(
      outer_fold = f, k_star = k_star, lambda_star = lambda_star,
      n_holdout_respondents = length(holdout_case),
      inner_selected_loss = grid$inner_loss[best_i],
      baseline_inner_loss = baseline_inner_loss,
      primary_baseline_loss = primary_baseline_loss,
      primary_corrected_loss = primary_corrected_loss,
      primary_gain = primary_baseline_loss - primary_corrected_loss,
      secondary_baseline_loss = secondary_baseline_loss,
      secondary_corrected_loss = secondary_corrected_loss,
      secondary_gain = secondary_baseline_loss - secondary_corrected_loss
    )
    message(sprintf(
      "[%s] outer fold %d: k*=%s lambda*=%s primary_gain=%.6f secondary_gain=%.6f",
      tag, f, k_star, ifelse(is.infinite(lambda_star), "Inf", as.character(lambda_star)),
      fold_summary[[f]]$primary_gain, fold_summary[[f]]$secondary_gain
    ))
  }

  fold_summary_df <- do.call(rbind, fold_summary)
  grid_all <- do.call(rbind, grid_records)

  ## Pool primary-corrected predictions across folds, in baseline$no order.
  primary_no_all <- unlist(lapply(fold_rows, `[[`, "primary_no"))
  primary_corrected_all <- do.call(rbind, lapply(fold_rows, `[[`, "primary_corrected"))
  ord <- match(baseline$no, primary_no_all)
  stopifnot(!anyNA(ord), length(ord) == length(baseline$no))
  primary_corrected_full <- primary_corrected_all[ord, , drop = FALSE]

  overall_primary_baseline_loss <- log_loss_matrix(baseline$truth, baseline$pred)
  overall_primary_corrected_loss <- log_loss_matrix(baseline$truth, primary_corrected_full)

  bootstrap_primary <- bootstrap_case_gain(
    baseline$truth, baseline$pred, primary_corrected_full, baseline$case
  )

  ## Secondary: pool baseline + corrected across folds (own row order/truth).
  secondary_no_all <- unlist(lapply(fold_rows, `[[`, "secondary_no"))
  secondary_pred_all <- do.call(rbind, lapply(fold_rows, `[[`, "secondary_pred"))
  secondary_corrected_all <- do.call(rbind, lapply(fold_rows, `[[`, "secondary_corrected"))
  ## Recover Case straight from the cached fits, in the same stacking order
  ## used to build secondary_no_all/secondary_pred_all/secondary_corrected_all.
  secondary_case_all <- unlist(lapply(seq_len(5), function(f) {
    final_fit <- load_cached_fit(sprintf(
      "data_processed/codex_version_shrinkage/outer%d_final_base_fit.rds", f
    ))
    as.integer(final_fit$target_case)
  }))
  secondary_truth_all <- truth_lookup$truth[match(secondary_no_all, truth_lookup$no), , drop = FALSE]
  overall_secondary_baseline_loss <- log_loss_matrix(secondary_truth_all, secondary_pred_all)
  overall_secondary_corrected_loss <- log_loss_matrix(secondary_truth_all, secondary_corrected_all)
  bootstrap_secondary <- bootstrap_case_gain(
    secondary_truth_all, secondary_pred_all, secondary_corrected_all, secondary_case_all
  )

  overall <- data.frame(
    tag = tag,
    primary_baseline_loss = overall_primary_baseline_loss,
    primary_corrected_loss = overall_primary_corrected_loss,
    primary_gain = overall_primary_baseline_loss - overall_primary_corrected_loss,
    primary_boot_lower95 = bootstrap_primary$lower_95,
    primary_boot_upper95 = bootstrap_primary$upper_95,
    primary_boot_lower99 = bootstrap_primary$lower_99,
    primary_boot_upper99 = bootstrap_primary$upper_99,
    primary_win_rate = bootstrap_primary$win_rate,
    secondary_baseline_loss = overall_secondary_baseline_loss,
    secondary_corrected_loss = overall_secondary_corrected_loss,
    secondary_gain = overall_secondary_baseline_loss - overall_secondary_corrected_loss,
    secondary_boot_lower95 = bootstrap_secondary$lower_95,
    secondary_boot_upper95 = bootstrap_secondary$upper_95
  )

  write.csv(fold_summary_df, file.path(out_dir, sprintf("%s_fold_summary.csv", tag)), row.names = FALSE)
  write.csv(grid_all, file.path(out_dir, sprintf("%s_grid_all.csv", tag)), row.names = FALSE)
  write.csv(overall, file.path(out_dir, sprintf("%s_overall.csv", tag)), row.names = FALSE)
  write.csv(bootstrap_primary, file.path(out_dir, sprintf("%s_bootstrap_primary.csv", tag)), row.names = FALSE)
  write.csv(bootstrap_secondary, file.path(out_dir, sprintf("%s_bootstrap_secondary.csv", tag)), row.names = FALSE)
  saveRDS(
    list(
      fold_summary = fold_summary_df, grid_all = grid_all, overall = overall,
      bootstrap_primary = bootstrap_primary, bootstrap_secondary = bootstrap_secondary,
      primary_corrected_full = primary_corrected_full, baseline_no = baseline$no,
      baseline_case = baseline$case
    ),
    file.path(out_dir, sprintf("%s_result.rds", tag))
  )

  list(
    fold_summary = fold_summary_df, grid_all = grid_all, overall = overall,
    bootstrap_primary = bootstrap_primary, bootstrap_secondary = bootstrap_secondary
  )
}
