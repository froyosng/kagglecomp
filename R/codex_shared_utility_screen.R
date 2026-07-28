source("R/codex_shared_utility_common.R")

dir.create(
  shared_utility_output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

gradient_check <- check_group_softmax_gradient()
stopifnot(gradient_check$passed)
write_result_csv(
  gradient_check,
  file.path(shared_utility_output_dir, "gradient_check.csv")
)
cat(sprintf(
  "Finite-difference gradient check passed; max error %.3e\n",
  gradient_check$max_absolute_error
))

# A one-iteration clogitboost fit on the project's full 44-column long-format
# matrix spent ~195 seconds before smooth.spline stopped at the first binary
# feature ("need at least four unique 'x' values"). Binary alternative flags
# and categorical attribute levels are essential here, so silently deleting
# all such columns would not be a faithful lower-risk implementation. Re-run
# the package's small built-in fit/predict smoke test, record the observed
# project-data gate, and proceed to the custom-softmax fallback.
clogitboost_installed <- requireNamespace(
  "clogitboost", quietly = TRUE
)
clogitboost_smoke <- FALSE
if (clogitboost_installed) {
  smoke_result <- try({
    smoke_environment <- new.env(parent = emptyenv())
    data(
      "travel",
      package = "clogitboost",
      envir = smoke_environment
    )
    travel <- smoke_environment$travel
    smoke_x <- as.matrix(
      travel[, c("TTME", "INVC", "INVT", "GC")]
    )
    smoke_fit <- clogitboost::clogitboost(
      travel$MODE,
      smoke_x,
      travel$Group,
      iter = 2L,
      rho = 0.01
    )
    smoke_prediction <- predict(
      smoke_fit,
      smoke_x,
      travel$Group
    )
    stopifnot(
      length(smoke_prediction$prob) == nrow(travel),
      all(is.finite(smoke_prediction$prob))
    )
    TRUE
  }, silent = TRUE)
  clogitboost_smoke <- isTRUE(smoke_result)
}
clogitboost_gate <- data.frame(
  installed = clogitboost_installed,
  package_version = if (clogitboost_installed) {
    as.character(packageVersion("clogitboost"))
  } else {
    NA_character_
  },
  built_in_smoke_test = clogitboost_smoke,
  full_feature_fit_attempted = TRUE,
  full_feature_fit_succeeded = FALSE,
  elapsed_seconds_before_error = 194,
  error = "smooth.spline: need at least four unique 'x' values",
  decision = paste(
    "fallback to custom exact group-softmax objective;",
    "binary/categorical features are structurally necessary"
  )
)
write_result_csv(
  clogitboost_gate,
  file.path(shared_utility_output_dir, "clogitboost_gate.csv")
)

split_data <- readRDS("data_processed/train_val_split.rds")
train_long <- ensure_choice_long(split_data$train_long_tr)
valid_long <- ensure_choice_long(split_data$train_long_val)
screen_cache <- readRDS(
  "data_processed/codex_behavioral_round/mlp_screen.rds"
)
valid_no <- screen_cache$validation_no
truth <- screen_cache$truth
v11 <- screen_cache$v11_screen
stopifnot(
  identical(valid_no, sort(unique(valid_long$No))),
  abs(log_loss_matrix(truth, v11) - 1.160568492729377) < 1e-8
)

cat("Fitting one m8trpg base model for residual offsets...\n")
base_fit <- fit_m8trpg_model(train_long)
base_train_result <- predict_m8trpg_margin(base_fit, train_long)
base_valid_result <- predict_m8trpg_margin(base_fit, valid_long)
base_valid <- aligned_prediction(base_valid_result, valid_no)
stopifnot(
  abs(log_loss_matrix(truth, base_valid) -
    1.15968144721113) < 1e-8
)
xgb_valid <- (v11 - 0.8 * base_valid) / 0.2
xgb_valid <- pmax(xgb_valid, 1e-15)
xgb_valid <- xgb_valid / rowSums(xgb_valid)
stopifnot(
  min(xgb_valid) > 0,
  max(abs(rowSums(xgb_valid) - 1)) < 1e-8
)

specifications <- list(
  shared_d2 = list(
    family = "shared",
    params = shared_xgb_params(
      max_depth = 2L, eta = 0.05,
      min_child_weight = 20, lambda = 10
    ),
    checkpoints = c(50L, 100L, 200L, 300L)
  ),
  shared_d3 = list(
    family = "shared",
    params = shared_xgb_params(
      max_depth = 3L, eta = 0.05,
      min_child_weight = 20, lambda = 10
    ),
    checkpoints = c(50L, 100L, 200L)
  ),
  residual_d1 = list(
    family = "residual",
    params = shared_xgb_params(
      max_depth = 1L, eta = 0.03,
      min_child_weight = 50, lambda = 30
    ),
    checkpoints = c(10L, 25L, 50L, 100L, 200L)
  ),
  residual_d2 = list(
    family = "residual",
    params = shared_xgb_params(
      max_depth = 2L, eta = 0.03,
      min_child_weight = 50, lambda = 30
    ),
    checkpoints = c(10L, 25L, 50L, 100L, 200L)
  ),
  residual_d2_strong = list(
    family = "residual",
    params = shared_xgb_params(
      max_depth = 2L, eta = 0.02,
      min_child_weight = 100, lambda = 100
    ),
    checkpoints = c(25L, 50L, 100L, 200L)
  )
)

result_rows <- list()
predictions <- list()
for (name in names(specifications)) {
  specification <- specifications[[name]]
  is_residual <- identical(specification$family, "residual")
  train_offset <- if (is_residual) {
    base_train_result$margin
  } else {
    NULL
  }
  valid_offset <- if (is_residual) {
    base_valid_result$margin
  } else {
    NULL
  }
  max_rounds <- max(specification$checkpoints)
  cat(sprintf(
    "Fitting %s (%s), maximum %d rounds...\n",
    name, specification$family, max_rounds
  ))
  model <- fit_group_softmax_xgb(
    train_long,
    params = specification$params,
    nrounds = max_rounds,
    base_margin = train_offset
  )
  for (rounds in specification$checkpoints) {
    predicted <- predict_group_softmax_xgb(
      model,
      valid_long,
      base_margin = valid_offset,
      nrounds = rounds
    )$pred
    component_loss <- log_loss_matrix(truth, predicted)
    if (is_residual) {
      primary_candidate <- 0.8 * predicted + 0.2 * xgb_valid
      primary_label <- "replace_m8trpg_in_v11"
    } else {
      primary_candidate <- predicted
      primary_label <- "shared_component"
    }
    primary_loss <- log_loss_matrix(truth, primary_candidate)
    weights <- seq(0, 0.40, by = 0.01)
    blend_losses <- vapply(weights, function(weight) {
      log_loss_matrix(
        truth,
        (1 - weight) * v11 + weight * primary_candidate
      )
    }, numeric(1))
    best <- which.min(blend_losses)
    key <- paste(name, rounds, sep = "_r")
    result_rows[[key]] <- data.frame(
      config = name,
      family = specification$family,
      rounds = rounds,
      component_logloss = component_loss,
      primary_comparison = primary_label,
      primary_logloss = primary_loss,
      primary_gain_vs_v11 =
        log_loss_matrix(truth, v11) - primary_loss,
      diagnostic_best_weight = weights[best],
      diagnostic_blend_logloss = blend_losses[best],
      diagnostic_blend_gain =
        log_loss_matrix(truth, v11) - blend_losses[best]
    )
    predictions[[key]] <- list(
      component = predicted,
      primary_candidate = primary_candidate
    )
    cat(sprintf(
      "%s r=%d: component %.6f; primary %.6f (%+.6f); ",
      name, rounds, component_loss, primary_loss,
      log_loss_matrix(truth, v11) - primary_loss
    ))
    cat(sprintf(
      "diagnostic blend w=%.2f, %.6f (%+.6f)\n",
      weights[best], blend_losses[best],
      log_loss_matrix(truth, v11) - blend_losses[best]
    ))
    flush.console()
  }
}

results <- do.call(rbind, result_rows)
results <- results[order(results$primary_logloss), , drop = FALSE]
write_result_csv(
  results,
  file.path(shared_utility_output_dir, "shared_utility_screen.csv")
)
saveRDS(
  list(
    results = results,
    predictions = predictions,
    truth = truth,
    valid_no = valid_no,
    base_m8trpg = base_valid,
    xgb = xgb_valid,
    v11 = v11
  ),
  file.path(shared_utility_output_dir, "shared_utility_screen.rds")
)
print(results, digits = 9)
