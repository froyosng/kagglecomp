# Triple-free v14 diversity recombination.
#
# Runner v2 reconstructs canonical seed 4821 from its source OOF caches;
# the original repeated-CV runner never wrote repeat_seed_4821.rds.
#
# Pre-registration:
#   codex_safe_diversity_recombination_preregister.md
#
# Smoke test:
#   Sys.setenv(SAFE_DIVERSITY_SMOKE = "1")
#   source("R/codex_safe_diversity_recombination_v2.R")
#   Sys.unsetenv("SAFE_DIVERSITY_SMOKE")
#
# Full repeated-OOF evaluation:
#   Sys.unsetenv("SAFE_DIVERSITY_SMOKE")
#   source("R/codex_safe_diversity_recombination_v2.R")

options(stringsAsFactors = FALSE)

experiment_id <- "safe_v14_diversity_recombination_v1"
output_dir <- file.path(
  "data_processed",
  "codex_safe_diversity_recombination"
)
smoke_mode <- identical(
  Sys.getenv("SAFE_DIVERSITY_SMOKE", "0"),
  "1"
)

canonical_seed <- 4821L
additional_seeds <- c(
  1907L, 2719L, 6151L, 8293L, 104729L
)
all_seeds <- c(canonical_seed, additional_seeds)
component_names <- c(
  "v14", "rank_ndcg", "retuned_xgb",
  "cox", "deep_mlp"
)
anchor_weights <- c(
  v14 = 0.80,
  rank_ndcg = 0.06,
  retuned_xgb = 0.03,
  cox = 0.07,
  deep_mlp = 0.04
)
ridge_penalty <- 0.001
expected_v14_repeated_gain <- 0.0011636195
expected_v14_repeated_lower <- 0.0000105775
expected_v14_canonical_loss <- 1.1435331472708
matrix_tolerance <- 1e-10
metric_tolerance <- 5e-8
bootstrap_replicates <- if (smoke_mode) {
  200L
} else {
  as.integer(Sys.getenv(
    "SAFE_DIVERSITY_N_BOOT", "100000"
  ))
}

v14_dir <- file.path(
  "data_processed", "codex_set_context_network"
)
repeat_dir <- file.path(
  "data_processed", "codex_repeat_cv"
)
canonical_repeat_paths <- c(
  base = file.path(
    "data_processed", "oof_ensemble_v10.rds"
  ),
  rank_ndcg = file.path(
    "data_processed", "codex", "rank_oof.rds"
  ),
  retuned_xgb = file.path(
    "data_processed", "codex", "xgb_retune_oof.rds"
  ),
  cox = file.path(
    "data_processed", "codex", "cox_oof.rds"
  ),
  shallow_mlp = file.path(
    "data_processed", "codex_behavioral_round",
    "mlp_oof.rds"
  ),
  triple_mlogit = file.path(
    "data_processed", "codex_triples",
    "triple_oof.rds"
  ),
  deep_mlp = file.path(
    "data_processed", "codex_deep_stack",
    "torch_deep_oof.rds"
  )
)
domain_covariates <- c(
  "segmentind", "yearind", "milesind", "milesa",
  "nightind", "nighta", "pparkind", "genderind",
  "ageind", "agea", "educind", "regionind",
  "Urbind", "incomeind", "incomea"
)

stopifnot(
  identical(names(anchor_weights), component_names),
  all(anchor_weights > 0),
  abs(sum(anchor_weights) - 1) < 1e-12,
  ridge_penalty >= 0,
  bootstrap_replicates > 0L
)

stop_with_context <- function(message) {
  stop(
    paste0("[", experiment_id, "] ", message),
    call. = FALSE
  )
}

validate_probability <- function(
    prediction,
    n_rows = NULL) {
  prediction <- as.matrix(prediction)
  storage.mode(prediction) <- "double"
  if (!is.null(n_rows)) {
    if (!identical(
      dim(prediction),
      c(as.integer(n_rows), 4L)
    )) {
      stop_with_context(
        "A probability matrix has the wrong dimensions."
      )
    }
  } else if (ncol(prediction) != 4L) {
    stop_with_context(
      "A probability matrix does not have four columns."
    )
  }
  if (
    anyNA(prediction) ||
    !all(is.finite(prediction)) ||
    min(prediction) < -1e-12
  ) {
    stop_with_context(
      "A probability matrix contains invalid values."
    )
  }
  prediction <- pmax(prediction, 1e-15)
  prediction <- prediction / rowSums(prediction)
  if (
    max(abs(rowSums(prediction) - 1)) >= 1e-10
  ) {
    stop_with_context(
      "A probability matrix is not normalized."
    )
  }
  prediction
}

log_loss_matrix <- function(truth, prediction) {
  prediction <- validate_probability(
    prediction, nrow(truth)
  )
  -mean(rowSums(
    as.matrix(truth) *
      log(pmax(prediction, 1e-15))
  ))
}

row_log_loss <- function(truth, prediction) {
  prediction <- validate_probability(
    prediction, nrow(truth)
  )
  -rowSums(
    as.matrix(truth) *
      log(pmax(prediction, 1e-15))
  )
}

balanced_fold_map <- function(cases, seed) {
  cases <- sort(unique(as.integer(cases)))
  set.seed(as.integer(seed))
  assignment <- sample(
    rep(1:5, length.out = length(cases))
  )
  names(assignment) <- as.character(cases)
  assignment
}

validate_fold_map <- function(
    train,
    fold_map,
    seed) {
  if (
    is.null(names(fold_map)) ||
    length(fold_map) !=
      length(unique(train$Case)) ||
    anyNA(
      fold_map[
        as.character(unique(train$Case))
      ]
    ) ||
    !identical(
      sort(unique(as.integer(fold_map))),
      1:5
    )
  ) {
    stop_with_context(
      sprintf(
        "The fold map for seed %d is invalid.",
        seed
      )
    )
  }
  if (seed != canonical_seed) {
    expected <- balanced_fold_map(
      train$Case, seed
    )
    if (!identical(
      as.integer(fold_map[names(expected)]),
      as.integer(expected)
    )) {
      stop_with_context(
        sprintf(
          "The fold map for seed %d is not the ",
          "established balanced assignment.",
          seed
        )
      )
    }
  }
  fold_map
}

reconstruct_v14 <- function(
    baseline,
    set_context,
    row_fold,
    weights) {
  required <- c("fold", "set_context_weight")
  if (
    !is.data.frame(weights) ||
    !all(required %in% names(weights)) ||
    !all(1:5 %in% weights$fold)
  ) {
    stop_with_context(
      "The v14 fold-weight table is invalid."
    )
  }
  fold_weight <- weights$set_context_weight[
    match(1:5, weights$fold)
  ]
  if (
    anyNA(fold_weight) ||
    any(!is.finite(fold_weight)) ||
    any(fold_weight < 0) ||
    any(fold_weight > 1)
  ) {
    stop_with_context(
      "The v14 set-context weights are invalid."
    )
  }
  row_weight <- fold_weight[row_fold]
  validate_probability(
    (1 - row_weight) * baseline +
      row_weight * set_context,
    nrow(baseline)
  )
}

validate_v14_summary <- function() {
  path <- file.path(
    v14_dir, "repeated_cv_summary.csv"
  )
  if (!file.exists(path)) {
    stop_with_context(
      paste("Missing v14 summary:", path)
    )
  }
  summary <- read.csv(path)
  if (
    nrow(summary) != 1L ||
    !identical(
      summary$experiment[[1L]],
      "set_context_utility_network_v1"
    ) ||
    !identical(
      summary$stage[[1L]], "repeated_cv"
    ) ||
    !isTRUE(as.logical(summary$promote[[1L]])) ||
    as.integer(
      summary$positive_repeats[[1L]]
    ) != 6L ||
    as.integer(summary$n_repeats[[1L]]) != 6L ||
    abs(
      summary$point_gain[[1L]] -
        expected_v14_repeated_gain
    ) > metric_tolerance ||
    abs(
      summary$lower_95[[1L]] -
        expected_v14_repeated_lower
    ) > metric_tolerance
  ) {
    stop_with_context(
      "The promoted v14 summary no longer matches."
    )
  }
  summary
}

load_v14_artifact <- function(
    train,
    truth,
    seed) {
  path <- if (seed == canonical_seed) {
    file.path(v14_dir, "canonical_result.rds")
  } else {
    file.path(
      v14_dir,
      sprintf("repeat_result_%d.rds", seed)
    )
  }
  if (!file.exists(path)) {
    stop_with_context(
      paste("Missing v14 artifact:", path)
    )
  }
  object <- readRDS(path)
  required <- c(
    "experiment_id", "fold_map",
    "baseline_prediction",
    "set_context_prediction",
    "candidate_prediction", "weights"
  )
  if (
    !all(required %in% names(object)) ||
    !identical(
      object$experiment_id,
      "set_context_utility_network_v1"
    )
  ) {
    stop_with_context(
      sprintf(
        "The v14 artifact for seed %d has ",
        "an invalid schema.",
        seed
      )
    )
  }
  fold_map <- validate_fold_map(
    train, object$fold_map, seed
  )
  row_fold <- unname(
    fold_map[as.character(train$Case)]
  )
  baseline <- validate_probability(
    object$baseline_prediction, nrow(train)
  )
  set_context <- validate_probability(
    object$set_context_prediction, nrow(train)
  )
  candidate <- validate_probability(
    object$candidate_prediction, nrow(train)
  )
  reconstructed <- reconstruct_v14(
    baseline,
    set_context,
    row_fold,
    object$weights
  )
  if (
    max(abs(candidate - reconstructed)) >
      matrix_tolerance
  ) {
    stop_with_context(
      sprintf(
        "The v14 candidate for seed %d does ",
        "not reconstruct exactly.",
        seed
      )
    )
  }
  loss <- log_loss_matrix(truth, candidate)
  if (
    seed == canonical_seed &&
    abs(loss - expected_v14_canonical_loss) >
      metric_tolerance
  ) {
    stop_with_context(
      "The canonical v14 loss no longer matches."
    )
  }
  list(
    path = path,
    fold_map = fold_map,
    row_fold = row_fold,
    baseline = baseline,
    set_context = set_context,
    candidate = candidate,
    weights = object$weights,
    loss = loss
  )
}

load_repeat_artifact <- function(
    train,
    truth,
    seed,
    v14) {
  if (seed == canonical_seed) {
    base <- readRDS(
      canonical_repeat_paths[["base"]]
    )
    object <- list(
      repeat_seed = canonical_seed,
      fold_map = base$fold_of_case,
      components = list(
        mlogit = base$oof_mlogit,
        original_xgb = base$oof_xgb,
        rank_ndcg = readRDS(
          canonical_repeat_paths[["rank_ndcg"]]
        )[[1L]]$pred,
        retuned_xgb = readRDS(
          canonical_repeat_paths[["retuned_xgb"]]
        )[[1L]]$pred,
        cox = readRDS(
          canonical_repeat_paths[["cox"]]
        )$oof_min,
        shallow_mlp = readRDS(
          canonical_repeat_paths[["shallow_mlp"]]
        )$oof[["h08_d0.100"]],
        triple_mlogit = readRDS(
          canonical_repeat_paths[["triple_mlogit"]]
        )$oof[["triple_price_income_miles"]],
        deep_mlp = readRDS(
          canonical_repeat_paths[["deep_mlp"]]
        )$deep_oof
      )
    )
    path <- paste(
      unname(canonical_repeat_paths),
      collapse = ";"
    )
  } else {
    path <- file.path(
      repeat_dir,
      sprintf("repeat_seed_%d.rds", seed)
    )
    if (!file.exists(path)) {
      stop_with_context(
        paste(
          "Missing repeated-component artifact:",
          path
        )
      )
    }
    object <- readRDS(path)
  }
  required_components <- c(
    "mlogit", "original_xgb",
    "rank_ndcg", "retuned_xgb", "cox",
    "shallow_mlp", "triple_mlogit",
    "deep_mlp"
  )
  if (
    !identical(
      as.integer(object$repeat_seed),
      as.integer(seed)
    ) ||
    is.null(object$fold_map) ||
    is.null(object$components) ||
    !identical(
      names(object$components),
      required_components
    )
  ) {
    stop_with_context(
      sprintf(
        "The repeated-component artifact for ",
        "seed %d has an invalid schema.",
        seed
      )
    )
  }
  fold_map <- validate_fold_map(
    train, object$fold_map, seed
  )
  if (!identical(
    as.integer(
      fold_map[names(v14$fold_map)]
    ),
    as.integer(v14$fold_map)
  )) {
    stop_with_context(
      sprintf(
        "The repeated and v14 fold maps differ ",
        "for seed %d.",
        seed
      )
    )
  }
  checked <- lapply(
    object$components,
    validate_probability,
    n_rows = nrow(train)
  )
  names(checked) <- names(object$components)
  reconstructed_v12 <- validate_probability(
    0.85 * (
      0.80 * checked$mlogit +
        0.20 * checked$original_xgb
    ) +
      0.15 * checked$shallow_mlp,
    nrow(train)
  )
  if (
    max(abs(
      reconstructed_v12 - v14$baseline
    )) > matrix_tolerance
  ) {
    stop_with_context(
      sprintf(
        "The repeated components do not ",
        "reconstruct v14's baseline for seed %d.",
        seed
      )
    )
  }
  components <- list(
    v14 = v14$candidate,
    rank_ndcg = checked$rank_ndcg,
    retuned_xgb = checked$retuned_xgb,
    cox = checked$cox,
    deep_mlp = checked$deep_mlp
  )
  if (!identical(
    names(components), component_names
  )) {
    stop_with_context(
      "The frozen component order changed."
    )
  }
  list(
    path = path,
    fold_map = fold_map,
    row_fold = unname(
      fold_map[as.character(train$Case)]
    ),
    components = components
  )
}

softmax_weights <- function(theta) {
  eta <- c(as.numeric(theta), 0)
  eta <- eta - max(eta)
  output <- exp(eta)
  output / sum(output)
}

theta_from_weights <- function(weights) {
  weights <- pmax(as.numeric(weights), 1e-12)
  log(
    weights[-length(weights)] /
      weights[[length(weights)]]
  )
}

project_to_simplex <- function(value) {
  value <- as.numeric(value)
  if (
    length(value) == 0L ||
    !all(is.finite(value))
  ) {
    stop_with_context(
      "Cannot project invalid simplex weights."
    )
  }
  ordered <- sort(value, decreasing = TRUE)
  threshold_candidates <- (
    cumsum(ordered) - 1
  ) / seq_along(ordered)
  active <- which(
    ordered - threshold_candidates > 0
  )
  if (length(active) == 0L) {
    stop_with_context(
      "Simplex projection found no active weight."
    )
  }
  threshold <- threshold_candidates[[max(active)]]
  output <- pmax(value - threshold, 0)
  output / sum(output)
}

fit_convex_pool <- function(
    observed_probability,
    anchor = anchor_weights,
    penalty = ridge_penalty) {
  observed_probability <- as.matrix(
    observed_probability
  )
  if (
    ncol(observed_probability) !=
      length(anchor) ||
    nrow(observed_probability) == 0L ||
    anyNA(observed_probability) ||
    !all(is.finite(observed_probability)) ||
    !all(observed_probability > 0) ||
    penalty < 0
  ) {
    stop_with_context(
      "The convex-pool fitting input is invalid."
    )
  }

  objective_weights <- function(weights) {
    chosen_probability <- as.numeric(
      observed_probability %*% weights
    )
    -mean(log(pmax(
      chosen_probability, 1e-15
    ))) +
      penalty * sum((weights - anchor)^2)
  }
  gradient_weights <- function(weights) {
    chosen_probability <- as.numeric(
      observed_probability %*% weights
    )
    -colMeans(
      observed_probability /
        pmax(chosen_probability, 1e-15)
    ) +
      2 * penalty * (weights - anchor)
  }
  objective <- function(theta) {
    objective_weights(softmax_weights(theta))
  }
  gradient <- function(theta) {
    weights <- softmax_weights(theta)
    weight_gradient <- gradient_weights(weights)
    centered <- weight_gradient -
      sum(weights * weight_gradient)
    weights[-length(weights)] *
      centered[-length(weights)]
  }
  stationarity_residual <- function(weights) {
    projected <- project_to_simplex(
      weights - gradient_weights(weights)
    )
    max(abs(weights - projected))
  }
  make_candidate <- function(
      weights,
      solver,
      convergence,
      message) {
    list(
      weights = as.numeric(weights),
      objective = objective_weights(weights),
      stationarity =
        stationarity_residual(weights),
      solver = solver,
      convergence = as.integer(convergence),
      message = if (is.null(message)) {
        ""
      } else {
        as.character(message)
      }
    )
  }
  valid_candidate <- function(candidate) {
    !is.null(candidate) &&
      all(is.finite(candidate$weights)) &&
      all(candidate$weights >= -1e-10) &&
      abs(sum(candidate$weights) - 1) < 1e-9 &&
      is.finite(candidate$objective) &&
      is.finite(candidate$stationarity) &&
      candidate$stationarity <= 1e-6
  }
  projected_gradient_candidate <- function(
      start,
      solver) {
    weights <- project_to_simplex(start)
    value <- objective_weights(weights)
    step_size <- 1
    convergence <- 1L
    iterations <- 0L
    for (iteration in seq_len(10000L)) {
      iterations <- iteration
      weight_gradient <- gradient_weights(weights)
      if (
        stationarity_residual(weights) <= 1e-8
      ) {
        convergence <- 0L
        break
      }
      local_step <- step_size
      accepted <- FALSE
      for (backtrack in seq_len(80L)) {
        proposed <- project_to_simplex(
          weights - local_step * weight_gradient
        )
        change <- proposed - weights
        proposed_value <-
          objective_weights(proposed)
        quadratic_bound <- value +
          sum(weight_gradient * change) +
          sum(change^2) / (2 * local_step)
        if (
          is.finite(proposed_value) &&
          proposed_value <=
            quadratic_bound + 1e-14
        ) {
          accepted <- TRUE
          break
        }
        local_step <- local_step / 2
      }
      if (!accepted) {
        break
      }
      old_weights <- weights
      old_gradient <- weight_gradient
      weights <- proposed
      value <- proposed_value
      new_gradient <- gradient_weights(weights)
      displacement <- weights - old_weights
      gradient_change <- new_gradient - old_gradient
      curvature <- sum(
        displacement * gradient_change
      )
      if (
        curvature > 1e-20 &&
        sum(displacement^2) > 0
      ) {
        step_size <- min(
          1e4,
          max(
            1e-8,
            sum(displacement^2) / curvature
          )
        )
      } else {
        step_size <- min(1e4, 2 * local_step)
      }
    }
    make_candidate(
      weights,
      solver,
      convergence,
      sprintf(
        "projected-gradient iterations=%d",
        iterations
      )
    )
  }

  diagnostics <- character(0)
  candidates <- list()
  starts <- list(
    anchor,
    rep(1 / length(anchor), length(anchor))
  )
  for (start_index in seq_along(starts)) {
    fitted <- tryCatch(
      stats::optim(
        par = theta_from_weights(
          starts[[start_index]]
        ),
        fn = objective,
        gr = gradient,
        method = "BFGS",
        control = list(
          maxit = 3000L,
          reltol = 1e-11
        )
      ),
      error = function(error) error
    )
    if (inherits(fitted, "error")) {
      diagnostics <- c(
        diagnostics,
        sprintf(
          "BFGS start %d error: %s",
          start_index,
          conditionMessage(fitted)
        )
      )
    } else {
      candidate <- make_candidate(
        softmax_weights(fitted$par),
        sprintf("BFGS_start_%d", start_index),
        fitted$convergence,
        fitted$message
      )
      candidates[[length(candidates) + 1L]] <-
        candidate
      diagnostics <- c(
        diagnostics,
        sprintf(
          "%s code=%d KKT=%.3e objective=%.12f",
          candidate$solver,
          candidate$convergence,
          candidate$stationarity,
          candidate$objective
        )
      )
    }
  }

  valid <- vapply(
    candidates, valid_candidate, logical(1)
  )
  if (!any(valid)) {
    n_components <- length(anchor)
    epsilon <- 1e-10
    free_objective <- function(free_weights) {
      weights <- c(
        free_weights,
        1 - sum(free_weights)
      )
      objective_weights(weights)
    }
    free_gradient <- function(free_weights) {
      weights <- c(
        free_weights,
        1 - sum(free_weights)
      )
      weight_gradient <- gradient_weights(weights)
      weight_gradient[-n_components] -
        weight_gradient[[n_components]]
    }
    constraint_matrix <- rbind(
      diag(n_components - 1L),
      rep(-1, n_components - 1L)
    )
    constraint_offset <- c(
      rep(epsilon, n_components - 1L),
      -1 + epsilon
    )
    for (start_index in seq_along(starts)) {
      fitted <- tryCatch(
        stats::constrOptim(
          theta = starts[[start_index]][
            -n_components
          ],
          f = free_objective,
          grad = free_gradient,
          ui = constraint_matrix,
          ci = constraint_offset,
          method = "BFGS",
          control = list(
            maxit = 3000L,
            reltol = 1e-11
          ),
          outer.iterations = 100L,
          outer.eps = 1e-10
        ),
        error = function(error) error
      )
      if (inherits(fitted, "error")) {
        diagnostics <- c(
          diagnostics,
          sprintf(
            "constrOptim start %d error: %s",
            start_index,
            conditionMessage(fitted)
          )
        )
      } else {
        weights <- c(
          fitted$par,
          1 - sum(fitted$par)
        )
        candidate <- make_candidate(
          weights,
          sprintf(
            "constrOptim_start_%d",
            start_index
          ),
          fitted$convergence,
          fitted$message
        )
        candidates[[length(candidates) + 1L]] <-
          candidate
        diagnostics <- c(
          diagnostics,
          sprintf(
            "%s code=%d KKT=%.3e objective=%.12f",
            candidate$solver,
            candidate$convergence,
            candidate$stationarity,
            candidate$objective
          )
        )
      }
    }
    valid <- vapply(
      candidates, valid_candidate, logical(1)
    )
  }

  if (!any(valid)) {
    candidate_values <- vapply(
      candidates,
      function(candidate) candidate$objective,
      numeric(1)
    )
    projected_starts <- c(
      lapply(
        candidates[order(candidate_values)],
        function(candidate) candidate$weights
      ),
      starts
    )
    for (start_index in seq_along(
      projected_starts
    )) {
      candidate <- projected_gradient_candidate(
        projected_starts[[start_index]],
        sprintf(
          "projected_gradient_start_%d",
          start_index
        )
      )
      candidates[[length(candidates) + 1L]] <-
        candidate
      diagnostics <- c(
        diagnostics,
        sprintf(
          "%s code=%d KKT=%.3e objective=%.12f",
          candidate$solver,
          candidate$convergence,
          candidate$stationarity,
          candidate$objective
        )
      )
      if (valid_candidate(candidate)) {
        break
      }
    }
    valid <- vapply(
      candidates, valid_candidate, logical(1)
    )
  }

  if (!any(valid)) {
    stop_with_context(paste0(
      "Convex-pool optimizer failed the simplex ",
      "stationarity check:\n  ",
      paste(diagnostics, collapse = "\n  ")
    ))
  }
  valid_candidates <- candidates[valid]
  values <- vapply(
    valid_candidates,
    function(candidate) candidate$objective,
    numeric(1)
  )
  best <- valid_candidates[[which.min(values)]]
  weights <- best$weights
  names(weights) <- names(anchor)
  list(
    weights = weights,
    objective = best$objective,
    convergence = best$convergence,
    message = best$message,
    solver = best$solver,
    stationarity = best$stationarity
  )
}

chosen_component_probability <- function(
    truth,
    components,
    rows) {
  selected_truth <- as.matrix(
    truth[rows, , drop = FALSE]
  )
  output <- matrix(
    NA_real_,
    nrow = length(rows),
    ncol = length(component_names),
    dimnames = list(NULL, component_names)
  )
  for (index in seq_along(component_names)) {
    name <- component_names[[index]]
    output[, index] <- rowSums(
      selected_truth *
        components[[name]][
          rows, , drop = FALSE
        ]
    )
  }
  if (
    anyNA(output) ||
    !all(is.finite(output)) ||
    !all(output > 0)
  ) {
    stop_with_context(
      "Chosen component probabilities are invalid."
    )
  }
  output
}

blend_component_rows <- function(
    components,
    weights,
    rows) {
  output <- matrix(
    0,
    nrow = length(rows),
    ncol = 4L
  )
  for (name in component_names) {
    output <- output +
      weights[[name]] *
        components[[name]][
          rows, , drop = FALSE
        ]
  }
  validate_probability(output, length(rows))
}

crossfit_safe_pool <- function(
    truth,
    components,
    row_fold,
    seed,
    only_fold = NULL) {
  folds <- if (is.null(only_fold)) {
    1:5
  } else {
    as.integer(only_fold)
  }
  candidate <- matrix(
    NA_real_, nrow(truth), 4L
  )
  weight_rows <- list()
  for (fold in folds) {
    fitting_rows <- which(row_fold != fold)
    validation_rows <- which(row_fold == fold)
    observed <- chosen_component_probability(
      truth, components, fitting_rows
    )
    fitted <- fit_convex_pool(observed)
    candidate[validation_rows, ] <-
      blend_component_rows(
        components,
        fitted$weights,
        validation_rows
      )
    weight_rows[[as.character(fold)]] <-
      data.frame(
        seed = seed,
        outer_fold = fold,
        component = component_names,
        weight = as.numeric(fitted$weights),
        objective = fitted$objective,
        solver = fitted$solver,
        convergence = fitted$convergence,
        stationarity = fitted$stationarity
      )
  }
  if (is.null(only_fold)) {
    prediction <- validate_probability(
      candidate, nrow(truth)
    )
  } else {
    prediction <- validate_probability(
      candidate[
        row_fold == only_fold, ,
        drop = FALSE
      ],
      sum(row_fold == only_fold)
    )
  }
  list(
    prediction = prediction,
    weights = do.call(rbind, weight_rows)
  )
}

respondent_gain <- function(
    truth,
    baseline,
    candidate,
    case) {
  row_gain <-
    row_log_loss(truth, baseline) -
      row_log_loss(truth, candidate)
  output <- tapply(row_gain, case, mean)
  output[order(as.integer(names(output)))]
}

bootstrap_case_means <- function(
    case_gain,
    replicates,
    seed,
    chunk_size = 1000L) {
  case_gain <- as.numeric(case_gain)
  if (
    length(case_gain) <= 1L ||
    !all(is.finite(case_gain)) ||
    replicates <= 0L
  ) {
    stop_with_context(
      "The respondent bootstrap input is invalid."
    )
  }
  set.seed(as.integer(seed))
  output <- numeric(replicates)
  start <- 1L
  while (start <= replicates) {
    count <- min(
      chunk_size,
      replicates - start + 1L
    )
    index <- matrix(
      sample.int(
        length(case_gain),
        length(case_gain) * count,
        replace = TRUE
      ),
      nrow = length(case_gain),
      ncol = count
    )
    output[start:(start + count - 1L)] <-
      colMeans(matrix(
        case_gain[index],
        nrow = length(case_gain),
        ncol = count
      ))
    start <- start + count
  }
  output
}

summarize_case_gain <- function(
    case_gain,
    replicates,
    seed) {
  bootstrap <- bootstrap_case_means(
    case_gain, replicates, seed
  )
  list(
    summary = data.frame(
      point_gain = mean(case_gain),
      bootstrap_mean = mean(bootstrap),
      bootstrap_sd = sd(bootstrap),
      lower_95 = unname(
        quantile(bootstrap, 0.025)
      ),
      upper_95 = unname(
        quantile(bootstrap, 0.975)
      ),
      lower_99 = unname(
        quantile(bootstrap, 0.005)
      ),
      upper_99 = unname(
        quantile(bootstrap, 0.995)
      ),
      win_rate = mean(bootstrap > 0),
      n_boot = length(bootstrap)
    ),
    bootstrap = bootstrap
  )
}

fit_test_propensity <- function(train, test) {
  train_respondents <- train[
    !duplicated(train$Case),
    c("Case", domain_covariates),
    drop = FALSE
  ]
  test_respondents <- test[
    !duplicated(test$Case),
    c("Case", domain_covariates),
    drop = FALSE
  ]
  train_domain <- train_respondents[
    , domain_covariates, drop = FALSE
  ]
  test_domain <- test_respondents[
    , domain_covariates, drop = FALSE
  ]
  train_domain$is_test <- 0L
  test_domain$is_test <- 1L
  domain <- rbind(train_domain, test_domain)
  classifier <- stats::glm(
    is_test ~ .,
    data = domain,
    family = stats::binomial()
  )
  propensity <- stats::predict(
    classifier,
    newdata = train_respondents[
      , domain_covariates, drop = FALSE
    ],
    type = "response"
  )
  propensity <- pmin(
    pmax(as.numeric(propensity), 1e-8),
    1 - 1e-8
  )
  names(propensity) <- as.character(
    train_respondents$Case
  )
  list(
    classifier = classifier,
    propensity = propensity
  )
}

test_like_diagnostic <- function(
    average_case_gain,
    propensity,
    replicates) {
  if (!all(
    names(average_case_gain) %in%
      names(propensity)
  )) {
    stop_with_context(
      "Test-propensity cases do not align."
    )
  }
  propensity <- propensity[
    names(average_case_gain)
  ]
  ordering <- order(
    propensity, decreasing = TRUE
  )
  groups <- list(
    all = seq_along(ordering),
    top_50pct = ordering[
      seq_len(ceiling(0.50 * length(ordering)))
    ],
    top_30pct = ordering[
      seq_len(ceiling(0.30 * length(ordering)))
    ]
  )
  rows <- lapply(
    seq_along(groups),
    function(index) {
      group_name <- names(groups)[[index]]
      group_index <- groups[[index]]
      summarized <- summarize_case_gain(
        average_case_gain[group_index],
        replicates,
        seed = 17700L + index
      )$summary
      cbind(
        data.frame(
          group = group_name,
          n_respondents = length(group_index),
          mean_test_propensity =
            mean(propensity[group_index])
        ),
        summarized
      )
    }
  )
  do.call(rbind, rows)
}

write_result_csv <- function(object, path) {
  write.csv(object, path, row.names = FALSE)
}

run_smoke <- function(train, truth) {
  cat(
    "\nRunning safe-diversity smoke test.\n"
  )
  validate_v14_summary()
  v14 <- load_v14_artifact(
    train, truth, canonical_seed
  )
  repeated <- load_repeat_artifact(
    train, truth, canonical_seed, v14
  )
  result <- crossfit_safe_pool(
    truth,
    repeated$components,
    repeated$row_fold,
    canonical_seed,
    only_fold = 1L
  )
  rows <- repeated$row_fold == 1L
  baseline <- v14$candidate[
    rows, , drop = FALSE
  ]
  fold_truth <- truth[rows, , drop = FALSE]
  baseline_loss <- log_loss_matrix(
    fold_truth, baseline
  )
  candidate_loss <- log_loss_matrix(
    fold_truth, result$prediction
  )
  if (
    nrow(result$weights) !=
      length(component_names) ||
    any(result$weights$outer_fold != 1L) ||
    max(result$weights$stationarity) > 1e-6
  ) {
    stop_with_context(
      "The smoke weight fit failed validation."
    )
  }
  cat("\nSmoke fold weights:\n")
  print(result$weights, digits = 9)
  cat(sprintf(
    "\nSmoke fold 1 v14 loss: %.9f\n",
    baseline_loss
  ))
  cat(sprintf(
    "Smoke fold 1 candidate loss: %.9f\n",
    candidate_loss
  ))
  cat(
    "\nSmoke test completed successfully\n"
  )
  invisible(list(
    weights = result$weights,
    baseline_loss = baseline_loss,
    candidate_loss = candidate_loss
  ))
}

run_full <- function(train, test, truth) {
  dir.create(
    output_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )
  validate_v14_summary()
  seed_rows <- list()
  weight_rows <- list()
  artifact_rows <- list()
  case_levels <- sort(unique(train$Case))
  case_gain_matrix <- matrix(
    NA_real_,
    nrow = length(case_levels),
    ncol = length(all_seeds),
    dimnames = list(
      as.character(case_levels),
      as.character(all_seeds)
    )
  )
  seed_results <- list()

  for (seed in all_seeds) {
    cat(sprintf(
      "\nLoading and evaluating seed %d\n",
      seed
    ))
    v14 <- load_v14_artifact(
      train, truth, seed
    )
    repeated <- load_repeat_artifact(
      train, truth, seed, v14
    )
    fitted <- crossfit_safe_pool(
      truth,
      repeated$components,
      repeated$row_fold,
      seed
    )
    baseline_loss <- v14$loss
    candidate_loss <- log_loss_matrix(
      truth, fitted$prediction
    )
    gain <- baseline_loss - candidate_loss
    case_gain <- respondent_gain(
      truth,
      v14$candidate,
      fitted$prediction,
      train$Case
    )
    if (!identical(
      names(case_gain),
      as.character(case_levels)
    )) {
      stop_with_context(
        sprintf(
          "Respondent gains misalign for seed %d.",
          seed
        )
      )
    }
    case_gain_matrix[, as.character(seed)] <-
      as.numeric(case_gain)
    seed_rows[[as.character(seed)]] <-
      data.frame(
        seed = seed,
        baseline_logloss = baseline_loss,
        candidate_logloss = candidate_loss,
        gain = gain,
        mean_v14_weight = mean(
          fitted$weights$weight[
            fitted$weights$component == "v14"
          ]
        )
      )
    weight_rows[[as.character(seed)]] <-
      fitted$weights
    artifact_rows[[as.character(seed)]] <-
      data.frame(
        seed = seed,
        v14_artifact = v14$path,
        repeat_artifact = repeated$path
      )
    seed_results[[as.character(seed)]] <-
      list(
        seed = seed,
        fold_map = v14$fold_map,
        row_fold = repeated$row_fold,
        baseline = v14$candidate,
        candidate = fitted$prediction,
        weights = fitted$weights,
        case_gain = case_gain
      )
    cat(sprintf(
      "Seed %d: v14 %.9f, candidate %.9f, ",
      seed, baseline_loss, candidate_loss
    ))
    cat(sprintf("gain %+.9f\n", gain))
  }

  by_seed <- do.call(rbind, seed_rows)
  weights <- do.call(rbind, weight_rows)
  artifacts <- do.call(rbind, artifact_rows)
  rownames(by_seed) <- NULL
  rownames(weights) <- NULL
  rownames(artifacts) <- NULL

  average_case_gain <- rowMeans(case_gain_matrix)
  pooled <- summarize_case_gain(
    average_case_gain,
    bootstrap_replicates,
    seed = 17601L
  )
  mean_v14_weight <- mean(
    weights$weight[
      weights$component == "v14"
    ]
  )
  repeated_summary <- cbind(
    data.frame(
      experiment = experiment_id,
      stage = "repeated_cv",
      positive_repeats = sum(by_seed$gain > 0),
      n_repeats = nrow(by_seed),
      mean_v14_weight = mean_v14_weight
    ),
    pooled$summary
  )

  propensity <- fit_test_propensity(
    train, test
  )
  test_like <- test_like_diagnostic(
    average_case_gain,
    propensity$propensity,
    bootstrap_replicates
  )
  top_30 <- test_like[
    test_like$group == "top_30pct",
    ,
    drop = FALSE
  ]

  promote <- (
    repeated_summary$point_gain[[1L]] > 0 &&
      repeated_summary$lower_95[[1L]] > 0 &&
      repeated_summary$positive_repeats[[1L]] >=
        5L &&
      top_30$point_gain[[1L]] >= 0 &&
      top_30$lower_95[[1L]] >= -0.001 &&
      mean_v14_weight >= 0.50 &&
      max(weights$stationarity) <= 1e-6
  )
  repeated_summary$test_like_30_point_gain <-
    top_30$point_gain[[1L]]
  repeated_summary$test_like_30_lower_95 <-
    top_30$lower_95[[1L]]
  repeated_summary$promote <- promote

  verdict <- if (promote) {
    paste0(
      "PROMOTE: the triple-free v14 diversity ",
      "recombination passes every pre-registered ",
      "repeated-OOF and transfer-safety gate. ",
      "Proceed only to a separate full-data build audit."
    )
  } else {
    paste0(
      "REJECT: the triple-free v14 diversity ",
      "recombination fails at least one ",
      "pre-registered gate. Retain exact v14 and ",
      "do not build or submit this pool."
    )
  }

  write_result_csv(
    by_seed,
    file.path(output_dir, "repeated_cv_by_seed.csv")
  )
  write_result_csv(
    repeated_summary,
    file.path(output_dir, "repeated_cv_summary.csv")
  )
  write_result_csv(
    weights,
    file.path(
      output_dir,
      "repeated_cv_fold_weights.csv"
    )
  )
  write_result_csv(
    test_like,
    file.path(
      output_dir,
      "test_like_respondent_diagnostic.csv"
    )
  )
  write_result_csv(
    artifacts,
    file.path(output_dir, "artifact_registry.csv")
  )
  writeLines(
    verdict,
    file.path(output_dir, "verdict.txt")
  )
  saveRDS(
    list(
      experiment_id = experiment_id,
      preregistration =
        "codex_safe_diversity_recombination_preregister.md",
      components = component_names,
      anchor_weights = anchor_weights,
      ridge_penalty = ridge_penalty,
      by_seed = by_seed,
      repeated_summary = repeated_summary,
      fold_weights = weights,
      case_gain_matrix = case_gain_matrix,
      average_case_gain = average_case_gain,
      pooled_bootstrap = pooled$bootstrap,
      test_propensity = propensity$propensity,
      test_like = test_like,
      artifact_registry = artifacts,
      seed_results = seed_results,
      verdict = verdict
    ),
    file.path(output_dir, "result.rds")
  )

  cat("\nRepeated-CV by seed:\n")
  print(by_seed, digits = 9)
  cat("\nRepeated-CV summary:\n")
  print(repeated_summary, digits = 9)
  cat("\nMean component weights:\n")
  mean_weights <- aggregate(
    weight ~ component,
    data = weights,
    FUN = mean
  )
  print(mean_weights, digits = 9)
  cat("\nTest-like respondent diagnostic:\n")
  print(test_like, digits = 9)
  cat("\n", verdict, "\n", sep = "")
  invisible(list(
    by_seed = by_seed,
    repeated_summary = repeated_summary,
    fold_weights = weights,
    test_like = test_like,
    verdict = verdict
  ))
}

required_paths <- c(
  file.path("csv files", "train.csv"),
  file.path("csv files", "test.csv"),
  file.path(v14_dir, "canonical_result.rds"),
  file.path(v14_dir, "repeated_cv_summary.csv"),
  unname(canonical_repeat_paths),
  file.path(
    repeat_dir,
    sprintf(
      "repeat_seed_%d.rds",
      additional_seeds
    )
  )
)
missing_paths <- required_paths[
  !file.exists(required_paths)
]
if (length(missing_paths) > 0L) {
  stop_with_context(paste0(
    "Missing required project file(s):\n  ",
    paste(missing_paths, collapse = "\n  "),
    "\nRun from the repository root after the ",
    "completed repeated-CV and v14 experiments."
  ))
}

train <- read.csv(
  file.path("csv files", "train.csv")
)
test <- read.csv(
  file.path("csv files", "test.csv")
)
truth <- as.matrix(
  train[, paste0("Ch", 1:4), drop = FALSE]
)
storage.mode(truth) <- "double"

if (
  nrow(train) != 21565L ||
  nrow(test) != 4997L ||
  length(unique(train$Case)) != 1135L ||
  length(unique(test$Case)) != 263L ||
  any(table(train$Case) != 19L) ||
  any(table(test$Case) != 19L) ||
  length(intersect(
    unique(train$Case),
    unique(test$Case)
  )) != 0L ||
  any(rowSums(truth) != 1L) ||
  !all(domain_covariates %in% names(train)) ||
  !all(domain_covariates %in% names(test))
) {
  stop_with_context(
    "The raw competition data fail structural checks."
  )
}

if (smoke_mode) {
  run_smoke(train, truth)
} else {
  run_full(train, test, truth)
}
