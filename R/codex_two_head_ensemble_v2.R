# Two-head ensemble: separately pool opt-out probability and conditional
# bundle-choice probability using four verified cross-fitted components.
#
# Pre-registration:
#   codex_two_head_ensemble_preregister.md
#
# Smoke test:
#   Sys.setenv(TWO_HEAD_SMOKE = "1")
#   source("R/codex_two_head_ensemble_v2.R")
#   Sys.unsetenv("TWO_HEAD_SMOKE")
#
# Full run:
#   Sys.unsetenv("TWO_HEAD_SMOKE")
#   Sys.setenv(TWO_HEAD_REPEATED = "auto")
#   source("R/codex_two_head_ensemble_v2.R")

options(stringsAsFactors = FALSE)

experiment_id <- "two_head_optout_bundle_ensemble_v1"
output_dir <- file.path(
  "data_processed", "codex_two_head_ensemble"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

smoke_mode <- identical(
  Sys.getenv("TWO_HEAD_SMOKE", "0"), "1"
)
repeated_mode <- tolower(
  Sys.getenv("TWO_HEAD_REPEATED", "auto")
)
stopifnot(repeated_mode %in% c("auto", "never", "always"))

canonical_seed <- 4821L
additional_seeds <- c(
  1907L, 2719L, 6151L, 8293L, 104729L
)
all_outer_seeds <- c(canonical_seed, additional_seeds)
component_names <- c(
  "mlogit", "original_xgb",
  "shallow_mlp", "set_context"
)

v14_dir <- file.path(
  "data_processed", "codex_set_context_network"
)
repeat_dir <- file.path(
  "data_processed", "codex_repeat_cv"
)
expected_v12_canonical_loss <- 1.143686618134879
expected_v14_repeated_gain <- 0.0011636195
expected_v14_repeated_lower <- 0.0000105775
metric_tolerance <- 5e-8
matrix_tolerance <- 1e-10

# The deployed v14 build used the mean set-context weight 0.111.
# Expanding v12's 0.68/0.17/0.15 component weights gives this anchor.
anchor_weights <- c(
  mlogit = 0.60452,
  original_xgb = 0.15113,
  shallow_mlp = 0.13335,
  set_context = 0.11100
)
penalty_grid <- c(1.00, 0.10, 0.01, 0.00)

bootstrap_replicates <- if (smoke_mode) {
  200L
} else {
  as.integer(Sys.getenv("TWO_HEAD_N_BOOT", "100000"))
}
stopifnot(
  bootstrap_replicates > 0L,
  identical(names(anchor_weights), component_names),
  all(anchor_weights > 0),
  abs(sum(anchor_weights) - 1) < 1e-12
)

domain_covariates <- c(
  "segmentind", "yearind", "milesind", "milesa",
  "nightind", "nighta", "pparkind", "genderind",
  "ageind", "agea", "educind", "regionind",
  "Urbind", "incomeind", "incomea"
)

input_files <- c(
  file.path("csv files", "train.csv"),
  file.path("csv files", "test.csv"),
  file.path("data_processed", "oof_ensemble_v10.rds"),
  file.path(v14_dir, "canonical_result.rds"),
  file.path(v14_dir, "repeated_cv_summary.csv")
)
missing_files <- input_files[!file.exists(input_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing required project file(s):\n  ",
    paste(missing_files, collapse = "\n  "),
    "\nRun this script from the kagglecomp repository root after ",
    "the completed and promoted v14 set-context experiment."
  )
}

validate_probability <- function(prediction, n_rows = NULL) {
  prediction <- as.matrix(prediction)
  if (!is.null(n_rows)) {
    stopifnot(identical(
      dim(prediction),
      c(as.integer(n_rows), 4L)
    ))
  } else {
    stopifnot(ncol(prediction) == 4L)
  }
  stopifnot(
    !anyNA(prediction),
    all(is.finite(prediction)),
    min(prediction) >= -1e-12
  )
  prediction <- pmax(prediction, 1e-15)
  prediction <- prediction / rowSums(prediction)
  stopifnot(
    max(abs(rowSums(prediction) - 1)) < 1e-10
  )
  prediction
}

log_loss_matrix_local <- function(truth, prediction) {
  prediction <- validate_probability(
    prediction, nrow(truth)
  )
  -mean(rowSums(
    as.matrix(truth) *
      log(pmax(prediction, 1e-15))
  ))
}

row_log_loss_local <- function(truth, prediction) {
  prediction <- validate_probability(
    prediction, nrow(truth)
  )
  -rowSums(
    as.matrix(truth) *
      log(pmax(prediction, 1e-15))
  )
}

balanced_fold_map <- function(cases, folds, seed) {
  cases <- sort(unique(as.integer(cases)))
  set.seed(as.integer(seed))
  assignment <- sample(
    rep(seq_len(folds), length.out = length(cases))
  )
  names(assignment) <- as.character(cases)
  stopifnot(
    length(assignment) == length(cases),
    all(table(assignment) %in% c(
      floor(length(cases) / folds),
      ceiling(length(cases) / folds)
    ))
  )
  assignment
}

validate_fold_map <- function(train, fold_map, seed) {
  stopifnot(
    !is.null(names(fold_map)),
    length(fold_map) == length(unique(train$Case)),
    !anyNA(
      fold_map[as.character(unique(train$Case))]
    ),
    identical(
      sort(unique(as.integer(fold_map))),
      1:5
    )
  )
  if (seed != canonical_seed) {
    expected <- balanced_fold_map(
      unique(train$Case), 5L, seed
    )
    stopifnot(identical(
      as.integer(fold_map[names(expected)]),
      as.integer(expected)
    ))
  }
  fold_map
}

numeric_signature <- function(x) {
  x <- as.numeric(as.matrix(x))
  index <- ((seq_along(x) - 1L) %% 997L) + 1L
  c(
    length = length(x),
    sum = sum(x),
    sum_square = sum(x * x),
    indexed_sum = sum(x * index)
  )
}

same_signature <- function(a, b, tolerance = 1e-10) {
  identical(names(a), names(b)) &&
    length(a) == length(b) &&
    max(
      abs(as.numeric(a) - as.numeric(b))
    ) <= tolerance
}

check_component_list <- function(components, n_rows) {
  stopifnot(identical(
    names(components), component_names
  ))
  for (name in component_names) {
    components[[name]] <- validate_probability(
      components[[name]], n_rows
    )
  }
  components
}

reconstruct_v14 <- function(
    baseline,
    set_context,
    row_fold,
    saved_weights) {
  stopifnot(
    is.data.frame(saved_weights),
    all(c(
      "fold", "set_context_weight"
    ) %in% names(saved_weights)),
    all(1:5 %in% saved_weights$fold)
  )
  weight_by_fold <- saved_weights$set_context_weight[
    match(1:5, saved_weights$fold)
  ]
  stopifnot(
    length(weight_by_fold) == 5L,
    !anyNA(weight_by_fold),
    all(is.finite(weight_by_fold)),
    all(weight_by_fold >= 0),
    all(weight_by_fold <= 1)
  )
  row_weight <- weight_by_fold[row_fold]
  validate_probability(
    (1 - row_weight) * baseline +
      row_weight * set_context,
    nrow(baseline)
  )
}

load_v14_summary <- function() {
  path <- file.path(
    v14_dir, "repeated_cv_summary.csv"
  )
  summary <- read.csv(path)
  stopifnot(
    nrow(summary) == 1L,
    identical(
      summary$experiment[[1L]],
      "set_context_utility_network_v1"
    ),
    identical(summary$stage[[1L]], "repeated_cv"),
    isTRUE(as.logical(summary$promote[[1L]])),
    as.integer(
      summary$positive_repeats[[1L]]
    ) == 6L,
    as.integer(summary$n_repeats[[1L]]) == 6L,
    abs(
      summary$point_gain[[1L]] -
        expected_v14_repeated_gain
    ) <= metric_tolerance,
    abs(
      summary$lower_95[[1L]] -
        expected_v14_repeated_lower
    ) <= metric_tolerance
  )
  summary
}

load_v14_artifact <- function(train, truth, seed) {
  path <- if (seed == canonical_seed) {
    file.path(v14_dir, "canonical_result.rds")
  } else {
    file.path(
      v14_dir,
      sprintf("repeat_result_%d.rds", seed)
    )
  }
  if (!file.exists(path)) {
    stop("Missing verified v14 artifact: ", path)
  }
  object <- readRDS(path)
  required <- c(
    "experiment_id", "fold_map",
    "baseline_prediction",
    "set_context_prediction",
    "candidate_prediction", "weights"
  )
  stopifnot(
    all(required %in% names(object)),
    identical(
      object$experiment_id,
      "set_context_utility_network_v1"
    )
  )
  fold_map <- validate_fold_map(
    train, object$fold_map, seed
  )
  row_fold <- unname(
    fold_map[as.character(train$Case)]
  )
  stopifnot(!anyNA(row_fold))
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
  stopifnot(
    max(abs(candidate - reconstructed)) <=
      matrix_tolerance
  )
  if (
    seed == canonical_seed &&
    !is.null(object$summary$candidate_logloss)
  ) {
    stopifnot(
      abs(
        log_loss_matrix_local(truth, candidate) -
          object$summary$candidate_logloss[[1L]]
      ) <= 1e-10
    )
  }
  list(
    path = path,
    fold_map = fold_map,
    row_fold = row_fold,
    baseline_prediction = baseline,
    set_context_prediction = set_context,
    candidate_prediction = candidate,
    weights = object$weights
  )
}

load_canonical_shallow_mlp <- function(n_rows) {
  candidate_paths <- c(
    file.path(
      "data_processed", "codex_behavioral_round",
      "mlp_oof.rds"
    ),
    file.path(
      "data_processed", "codex_componentwise_boost",
      "shallow_mlp_oof_rebuilt.rds"
    ),
    file.path(
      "data_processed", "codex_choice_set_geometry",
      "shallow_mlp_oof_rebuilt.rds"
    ),
    file.path(
      "data_processed", "codex_price_curve_shrinkage",
      "shallow_mlp_oof_rebuilt.rds"
    )
  )
  for (path in candidate_paths[file.exists(
    candidate_paths
  )]) {
    object <- readRDS(path)
    prediction <- NULL
    if (!is.null(object$oof)) {
      if (!is.null(
        object$oof[["h08_d0.100"]]
      )) {
        prediction <-
          object$oof[["h08_d0.100"]]
      } else if (length(object$oof) == 1L) {
        prediction <- object$oof[[1L]]
      }
    } else if (!is.null(object$prediction)) {
      prediction <- object$prediction
    } else if (!is.null(object$mlp_oof)) {
      prediction <- object$mlp_oof
    }
    if (
      !is.null(prediction) &&
      identical(
        dim(as.matrix(prediction)),
        c(as.integer(n_rows), 4L)
      )
    ) {
      cat(
        "Using frozen shallow-MLP OOF cache:",
        path, "\n"
      )
      return(list(
        path = path,
        prediction = validate_probability(
          prediction, n_rows
        )
      ))
    }
  }
  stop(
    paste0(
      "The frozen canonical shallow-MLP OOF cache was not found. ",
      "Run the already-verified componentwise or geometry runner ",
      "first; this script will not derive an unverified component."
    )
  )
}

load_seed_components <- function(
    train,
    truth,
    seed,
    v14) {
  if (seed == canonical_seed) {
    base_path <- file.path(
      "data_processed", "oof_ensemble_v10.rds"
    )
    base <- readRDS(base_path)
    stopifnot(
      !is.null(base$oof_mlogit),
      !is.null(base$oof_xgb),
      !is.null(base$fold_of_case)
    )
    base_map <- validate_fold_map(
      train, base$fold_of_case, seed
    )
    mlogit <- validate_probability(
      base$oof_mlogit, nrow(train)
    )
    original_xgb <- validate_probability(
      base$oof_xgb, nrow(train)
    )
    shallow_cache <- load_canonical_shallow_mlp(
      nrow(train)
    )
    shallow_mlp <- shallow_cache$prediction
    stopifnot(
      identical(
        as.integer(
          base_map[names(v14$fold_map)]
        ),
        as.integer(v14$fold_map)
      ),
      abs(
        log_loss_matrix_local(
          truth, v14$baseline_prediction
        ) - expected_v12_canonical_loss
      ) <= 1e-8,
      abs(
        log_loss_matrix_local(
          truth, shallow_mlp
        ) - 1.19054334979533
      ) <= 1e-8
    )
  } else {
    base_path <- file.path(
      repeat_dir,
      sprintf("repeat_seed_%d.rds", seed)
    )
    if (!file.exists(base_path)) {
      stop(
        "Missing verified repeated component artifact: ",
        base_path
      )
    }
    base <- readRDS(base_path)
    stopifnot(
      identical(
        as.integer(base$repeat_seed),
        as.integer(seed)
      ),
      !is.null(base$fold_map),
      !is.null(base$components),
      all(c(
        "mlogit", "original_xgb", "shallow_mlp"
      ) %in% names(base$components))
    )
    base_map <- validate_fold_map(
      train, base$fold_map, seed
    )
    stopifnot(identical(
      as.integer(
        base_map[names(v14$fold_map)]
      ),
      as.integer(v14$fold_map)
    ))
    mlogit <- validate_probability(
      base$components$mlogit, nrow(train)
    )
    original_xgb <- validate_probability(
      base$components$original_xgb, nrow(train)
    )
    shallow_mlp <- validate_probability(
      base$components$shallow_mlp, nrow(train)
    )
  }

  reconstructed_v12 <- validate_probability(
    0.85 * (
      0.80 * mlogit +
        0.20 * original_xgb
    ) + 0.15 * shallow_mlp,
    nrow(train)
  )
  stopifnot(
    max(abs(
      reconstructed_v12 -
        v14$baseline_prediction
    )) <= matrix_tolerance
  )
  components <- check_component_list(
    list(
      mlogit = mlogit,
      original_xgb = original_xgb,
      shallow_mlp = shallow_mlp,
      set_context = v14$set_context_prediction
    ),
    nrow(train)
  )
  list(
    source = base_path,
    components = components,
    v12_prediction = reconstructed_v12
  )
}

conditional_bundle_probability <- function(
    prediction) {
  prediction <- validate_probability(
    prediction
  )
  inside_mass <- rowSums(
    prediction[, 1:3, drop = FALSE]
  )
  stopifnot(
    all(is.finite(inside_mass)),
    all(inside_mass > 1e-12)
  )
  output <- prediction[
    , 1:3, drop = FALSE
  ] / inside_mass
  stopifnot(
    !anyNA(output),
    all(is.finite(output)),
    all(output >= 0),
    max(abs(rowSums(output) - 1)) < 1e-10
  )
  output
}

softmax_weights <- function(theta) {
  eta <- c(as.numeric(theta), 0)
  eta <- eta - max(eta)
  output <- exp(eta)
  output / sum(output)
}

theta_from_weights <- function(weights) {
  weights <- pmax(as.numeric(weights), 1e-12)
  log(weights[-length(weights)] /
    weights[[length(weights)]])
}

project_to_simplex <- function(value) {
  value <- as.numeric(value)
  stopifnot(
    length(value) > 0L,
    all(is.finite(value))
  )
  ordered <- sort(value, decreasing = TRUE)
  threshold_candidates <- (
    cumsum(ordered) - 1
  ) / seq_along(ordered)
  active <- which(
    ordered - threshold_candidates > 0
  )
  stopifnot(length(active) > 0L)
  threshold <- threshold_candidates[[max(active)]]
  output <- pmax(value - threshold, 0)
  output / sum(output)
}

fit_convex_pool <- function(
    observed_probability,
    anchor,
    penalty) {
  observed_probability <- as.matrix(
    observed_probability
  )
  stopifnot(
    ncol(observed_probability) ==
      length(anchor),
    nrow(observed_probability) > 0L,
    !anyNA(observed_probability),
    all(is.finite(observed_probability)),
    all(observed_probability > 0),
    penalty >= 0
  )
  objective_weights <- function(weights) {
    chosen_probability <- as.numeric(
      observed_probability %*% weights
    )
    -mean(log(pmax(
      chosen_probability, 1e-15
    ))) + penalty * sum(
      (weights - anchor)^2
    )
  }
  gradient_weights <- function(weights) {
    chosen_probability <- as.numeric(
      observed_probability %*% weights
    )
    -colMeans(
      observed_probability /
        pmax(chosen_probability, 1e-15)
    ) + 2 * penalty * (weights - anchor)
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
    weights <- as.numeric(weights)
    list(
      weights = weights,
      objective = objective_weights(weights),
      stationarity = stationarity_residual(weights),
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
    stationarity_tolerance <- 1e-6
    !is.null(candidate) &&
      all(is.finite(candidate$weights)) &&
      all(candidate$weights >= -1e-10) &&
      abs(sum(candidate$weights) - 1) < 1e-9 &&
      is.finite(candidate$objective) &&
      is.finite(candidate$stationarity) &&
      candidate$stationarity <=
        stationarity_tolerance
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
      if (stationarity_residual(weights) <= 1e-8) {
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
        proposed_value <- objective_weights(proposed)
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
      gradient_change <-
        new_gradient - old_gradient
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
      solver = solver,
      convergence = convergence,
      message = sprintf(
        "projected-gradient iterations=%d",
        iterations
      )
    )
  }

  # The objective is convex in the simplex weights. Ordinary BFGS on the
  # softmax coordinates is the same stable strategy used by the project's
  # verified arithmetic-pooling code. The earlier bounded L-BFGS-B call could
  # report an abnormal line-search termination on Windows even at a valid
  # solution, so solver status is now backed by an explicit simplex KKT check.
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
        solver = sprintf(
          "BFGS_start_%d", start_index
        ),
        convergence = fitted$convergence,
        message = fitted$message
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
    # Independent weight-space fallback. constrOptim enforces the simplex
    # directly and therefore remains well behaved when an optimum lies very
    # close to a boundary. The 1e-10 interior floor is smaller than the
    # effective floor in the original [-20, 20] softmax bounds.
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
          solver = sprintf(
            "constrOptim_start_%d",
            start_index
          ),
          convergence = fitted$convergence,
          message = fitted$message
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
    # Final deterministic convex fallback. Spectral projected-gradient descent
    # works directly on the closed simplex, including exact zero weights, and
    # is accepted only after the same KKT residual check.
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
    for (
      start_index in seq_along(projected_starts)
    ) {
      candidate <- projected_gradient_candidate(
        projected_starts[[start_index]],
        solver = sprintf(
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
    stop(
      "Convex-pool optimiser failed the simplex ",
      "stationarity check:\n  ",
      paste(diagnostics, collapse = "\n  ")
    )
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
  stopifnot(
    all(is.finite(weights)),
    all(weights >= -1e-10),
    abs(sum(weights) - 1) < 1e-9,
    is.finite(best$objective),
    best$stationarity <= 1e-6
  )
  list(
    weights = weights,
    objective = best$objective,
    convergence = best$convergence,
    message = best$message,
    solver = best$solver,
    stationarity = best$stationarity
  )
}

observed_head_probabilities <- function(
    truth,
    components,
    rows) {
  truth <- as.matrix(truth)
  stopifnot(
    ncol(truth) == 4L,
    length(rows) > 0L
  )
  selected_truth <- truth[
    rows, , drop = FALSE
  ]
  optout_chosen <- selected_truth[, 4L]
  optout_observed <- matrix(
    NA_real_,
    nrow = length(rows),
    ncol = length(component_names),
    dimnames = list(NULL, component_names)
  )
  inside_rows_local <- which(
    optout_chosen == 0L
  )
  stopifnot(length(inside_rows_local) > 0L)
  bundle_choice <- max.col(
    selected_truth[
      inside_rows_local, 1:3, drop = FALSE
    ],
    ties.method = "first"
  )
  bundle_observed <- matrix(
    NA_real_,
    nrow = length(inside_rows_local),
    ncol = length(component_names),
    dimnames = list(NULL, component_names)
  )
  for (index in seq_along(component_names)) {
    name <- component_names[[index]]
    prediction <- components[[name]][
      rows, , drop = FALSE
    ]
    q <- prediction[, 4L]
    optout_observed[, index] <-
      ifelse(optout_chosen == 1L, q, 1 - q)
    conditional <- conditional_bundle_probability(
      prediction
    )
    bundle_observed[, index] <-
      conditional[
        cbind(inside_rows_local, bundle_choice)
      ]
  }
  stopifnot(
    all(optout_observed > 0),
    all(bundle_observed > 0)
  )
  list(
    optout = optout_observed,
    bundle = bundle_observed
  )
}

fit_two_heads <- function(
    truth,
    components,
    rows,
    penalty) {
  observed <- observed_head_probabilities(
    truth, components, rows
  )
  list(
    optout = fit_convex_pool(
      observed$optout,
      anchor_weights,
      penalty
    ),
    bundle = fit_convex_pool(
      observed$bundle,
      anchor_weights,
      penalty
    )
  )
}

predict_two_heads <- function(
    components,
    rows,
    optout_weights,
    bundle_weights) {
  stopifnot(
    identical(
      names(optout_weights), component_names
    ),
    identical(
      names(bundle_weights), component_names
    )
  )
  q_matrix <- matrix(
    NA_real_,
    nrow = length(rows),
    ncol = length(component_names)
  )
  bundle_array <- array(
    NA_real_,
    dim = c(
      length(rows), 3L,
      length(component_names)
    )
  )
  for (index in seq_along(component_names)) {
    name <- component_names[[index]]
    prediction <- components[[name]][
      rows, , drop = FALSE
    ]
    q_matrix[, index] <- prediction[, 4L]
    bundle_array[, , index] <-
      conditional_bundle_probability(prediction)
  }
  q <- as.numeric(q_matrix %*% optout_weights)
  conditional <- matrix(
    0, nrow = length(rows), ncol = 3L
  )
  for (index in seq_along(component_names)) {
    conditional <- conditional +
      bundle_weights[[index]] *
      bundle_array[, , index]
  }
  conditional <- conditional /
    rowSums(conditional)
  output <- cbind(
    (1 - q) * conditional,
    q
  )
  validate_probability(output, length(rows))
}

run_nested_outer <- function(
    train,
    truth,
    components,
    fold_map,
    outer_fold) {
  row_fold <- unname(
    fold_map[as.character(train$Case)]
  )
  outer_rows <- which(row_fold == outer_fold)
  tuning_rows <- which(row_fold != outer_fold)
  inner_folds <- setdiff(1:5, outer_fold)
  stopifnot(
    length(outer_rows) > 0L,
    length(tuning_rows) > 0L,
    length(intersect(
      unique(train$Case[outer_rows]),
      unique(train$Case[tuning_rows])
    )) == 0L
  )

  penalty_losses <- numeric(
    length(penalty_grid)
  )
  penalty_details <- vector(
    "list", length(penalty_grid)
  )
  for (
    penalty_index in seq_along(penalty_grid)
  ) {
    penalty <- penalty_grid[[penalty_index]]
    inner_prediction <- matrix(
      NA_real_, nrow(train), 4L
    )
    inner_weight_rows <- list()
    for (inner_fold in inner_folds) {
      fit_rows <- which(
        row_fold != outer_fold &
          row_fold != inner_fold
      )
      validation_rows <- which(
        row_fold == inner_fold
      )
      stopifnot(
        length(intersect(
          unique(train$Case[fit_rows]),
          unique(train$Case[validation_rows])
        )) == 0L,
        length(intersect(
          unique(train$Case[fit_rows]),
          unique(train$Case[outer_rows])
        )) == 0L,
        length(intersect(
          unique(train$Case[validation_rows]),
          unique(train$Case[outer_rows])
        )) == 0L
      )
      fitted <- fit_two_heads(
        truth,
        components,
        fit_rows,
        penalty
      )
      inner_prediction[validation_rows, ] <-
        predict_two_heads(
          components,
          validation_rows,
          fitted$optout$weights,
          fitted$bundle$weights
        )
      inner_weight_rows[[as.character(
        inner_fold
      )]] <- rbind(
        data.frame(
          outer_fold = outer_fold,
          inner_fold = inner_fold,
          penalty = penalty,
          head = "optout",
          component = component_names,
          weight = fitted$optout$weights,
          objective = fitted$optout$objective
        ),
        data.frame(
          outer_fold = outer_fold,
          inner_fold = inner_fold,
          penalty = penalty,
          head = "bundle",
          component = component_names,
          weight = fitted$bundle$weights,
          objective = fitted$bundle$objective
        )
      )
    }
    stopifnot(!anyNA(
      inner_prediction[tuning_rows, , drop = FALSE]
    ))
    penalty_losses[[penalty_index]] <-
      log_loss_matrix_local(
        truth[tuning_rows, , drop = FALSE],
        inner_prediction[
          tuning_rows, , drop = FALSE
        ]
      )
    penalty_details[[penalty_index]] <-
      do.call(rbind, inner_weight_rows)
  }

  # penalty_grid is ordered strongest to weakest, so which.min()
  # prefers stronger shrinkage if losses are exactly tied.
  best_index <- which.min(penalty_losses)
  selected_penalty <- penalty_grid[[best_index]]
  fitted <- fit_two_heads(
    truth,
    components,
    tuning_rows,
    selected_penalty
  )
  prediction <- predict_two_heads(
    components,
    outer_rows,
    fitted$optout$weights,
    fitted$bundle$weights
  )
  list(
    outer_fold = outer_fold,
    outer_rows = outer_rows,
    prediction = prediction,
    selected_penalty = selected_penalty,
    penalty_curve = data.frame(
      outer_fold = outer_fold,
      penalty = penalty_grid,
      inner_logloss = penalty_losses
    ),
    inner_weights = do.call(
      rbind, penalty_details
    ),
    final_weights = rbind(
      data.frame(
        outer_fold = outer_fold,
        selected_penalty = selected_penalty,
        head = "optout",
        component = component_names,
        weight = fitted$optout$weights,
        objective = fitted$optout$objective
      ),
      data.frame(
        outer_fold = outer_fold,
        selected_penalty = selected_penalty,
        head = "bundle",
        component = component_names,
        weight = fitted$bundle$weights,
        objective = fitted$bundle$objective
      )
    )
  )
}

run_nested_crossfit <- function(
    train,
    truth,
    components,
    fold_map,
    seed) {
  prediction <- matrix(
    NA_real_, nrow(train), 4L
  )
  results <- vector("list", 5L)
  for (outer_fold in 1:5) {
    cat(sprintf(
      "  seed %d: two-head outer fold %d/5\n",
      seed, outer_fold
    ))
    result <- run_nested_outer(
      train,
      truth,
      components,
      fold_map,
      outer_fold
    )
    prediction[result$outer_rows, ] <-
      result$prediction
    results[[outer_fold]] <- result
    flush.console()
  }
  list(
    prediction = validate_probability(
      prediction, nrow(train)
    ),
    fold_results = results,
    final_weights = do.call(
      rbind,
      lapply(results, function(x) {
        x$final_weights
      })
    ),
    penalty_curves = do.call(
      rbind,
      lapply(results, function(x) {
        x$penalty_curve
      })
    )
  )
}

respondent_gain <- function(
    truth,
    baseline,
    candidate,
    case) {
  row_gain <-
    row_log_loss_local(truth, baseline) -
    row_log_loss_local(truth, candidate)
  output <- tapply(row_gain, case, mean)
  output[order(as.integer(names(output)))]
}

bootstrap_case_means <- function(
    case_gain,
    replicates,
    seed = canonical_seed,
    chunk_size = 1000L) {
  case_gain <- as.numeric(case_gain)
  stopifnot(
    length(case_gain) > 1L,
    all(is.finite(case_gain)),
    replicates > 0L
  )
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

bootstrap_summary <- function(
    truth,
    baseline,
    candidate,
    case,
    replicates,
    seed) {
  case_gain <- respondent_gain(
    truth, baseline, candidate, case
  )
  summarized <- summarize_case_gain(
    case_gain, replicates, seed
  )
  list(
    summary = summarized$summary,
    respondent_gain = case_gain,
    bootstrap = summarized$bootstrap
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
  stopifnot(all(
    names(average_case_gain) %in%
      names(propensity)
  ))
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
        seed = 16600L + index
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

run_smoke_test <- function(train, truth) {
  load_v14_summary()
  v14 <- load_v14_artifact(
    train, truth, canonical_seed
  )
  loaded <- load_seed_components(
    train, truth, canonical_seed, v14
  )
  result <- run_nested_outer(
    train,
    truth,
    loaded$components,
    v14$fold_map,
    outer_fold = 1L
  )
  baseline_fold_loss <- log_loss_matrix_local(
    truth[result$outer_rows, , drop = FALSE],
    v14$candidate_prediction[
      result$outer_rows, , drop = FALSE
    ]
  )
  candidate_fold_loss <- log_loss_matrix_local(
    truth[result$outer_rows, , drop = FALSE],
    result$prediction
  )
  checkpoint <- list(
    experiment_id = experiment_id,
    selected_penalty =
      result$selected_penalty,
    final_weights = result$final_weights,
    prediction = result$prediction,
    prediction_signature =
      numeric_signature(result$prediction),
    baseline_fold_loss = baseline_fold_loss,
    candidate_fold_loss = candidate_fold_loss
  )
  path <- file.path(
    output_dir, "smoke_result.rds"
  )
  saveRDS(checkpoint, path)
  reloaded <- readRDS(path)
  stopifnot(
    identical(
      reloaded$experiment_id,
      experiment_id
    ),
    same_signature(
      reloaded$prediction_signature,
      numeric_signature(result$prediction)
    ),
    all(result$final_weights$weight >= 0),
    all(result$final_weights$weight <= 1),
    all(abs(
      aggregate(
        weight ~ head,
        data = result$final_weights,
        FUN = sum
      )$weight - 1
    ) < 1e-10)
  )
  cat(sprintf(
    paste0(
      "\nSmoke test completed successfully: %s\n",
      "Selected penalty: %.2f\n",
      "Smoke v14 fold loss: %.6f\n",
      "Smoke two-head fold loss: %.6f\n"
    ),
    path,
    result$selected_penalty,
    baseline_fold_loss,
    candidate_fold_loss
  ))
  invisible(checkpoint)
}

run_experiment <- function() {
  train <- read.csv(
    file.path("csv files", "train.csv")
  )
  train <- train[
    order(train$No), , drop = FALSE
  ]
  rownames(train) <- NULL
  test <- read.csv(
    file.path("csv files", "test.csv")
  )
  test <- test[
    order(test$No), , drop = FALSE
  ]
  rownames(test) <- NULL
  truth <- as.matrix(
    train[, paste0("Ch", 1:4), drop = FALSE]
  )
  stopifnot(
    nrow(train) == 21565L,
    nrow(test) == 4997L,
    length(unique(train$Case)) == 1135L,
    length(unique(test$Case)) == 263L,
    length(intersect(
      unique(train$Case),
      unique(test$Case)
    )) == 0L,
    all(table(train$Case) == 19L),
    all(rowSums(truth) == 1L)
  )

  cat("\nRunning", experiment_id, "\n")
  cat(
    "Frozen v14 repeated gain:",
    sprintf("%.10f", expected_v14_repeated_gain),
    "\n"
  )
  load_v14_summary()

  if (smoke_mode) {
    return(run_smoke_test(train, truth))
  }

  v14_registry <- lapply(
    all_outer_seeds,
    function(seed) {
      load_v14_artifact(train, truth, seed)
    }
  )
  names(v14_registry) <- as.character(
    all_outer_seeds
  )
  canonical_v14 <- v14_registry[[
    as.character(canonical_seed)
  ]]
  canonical_components <- load_seed_components(
    train, truth, canonical_seed, canonical_v14
  )
  canonical_fit <- run_nested_crossfit(
    train,
    truth,
    canonical_components$components,
    canonical_v14$fold_map,
    canonical_seed
  )
  canonical_candidate_loss <-
    log_loss_matrix_local(
      truth, canonical_fit$prediction
    )
  canonical_baseline_loss <-
    log_loss_matrix_local(
      truth,
      canonical_v14$candidate_prediction
    )
  canonical_bootstrap <- bootstrap_summary(
    truth,
    canonical_v14$candidate_prediction,
    canonical_fit$prediction,
    train$Case,
    bootstrap_replicates,
    canonical_seed
  )
  canonical_summary <- cbind(
    data.frame(
      experiment = experiment_id,
      stage = "canonical",
      baseline_name = "set_context_v14",
      baseline_logloss = canonical_baseline_loss,
      candidate_logloss =
        canonical_candidate_loss
    ),
    canonical_bootstrap$summary
  )
  canonical_summary$canonical_positive <-
    canonical_summary$point_gain > 0
  canonical_summary$canonical_pass <-
    canonical_summary$point_gain > 0 &
    canonical_summary$lower_95 > 0

  canonical_weights <- cbind(
    data.frame(seed = canonical_seed),
    canonical_fit$final_weights
  )
  canonical_penalties <- cbind(
    data.frame(seed = canonical_seed),
    canonical_fit$penalty_curves
  )
  write.csv(
    canonical_summary,
    file.path(output_dir, "canonical_summary.csv"),
    row.names = FALSE
  )
  write.csv(
    canonical_weights,
    file.path(
      output_dir, "canonical_head_weights.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    canonical_penalties,
    file.path(
      output_dir, "canonical_penalty_curves.csv"
    ),
    row.names = FALSE
  )
  saveRDS(
    list(
      experiment_id = experiment_id,
      summary = canonical_summary,
      baseline_prediction =
        canonical_v14$candidate_prediction,
      candidate_prediction =
        canonical_fit$prediction,
      fold_map = canonical_v14$fold_map,
      component_source =
        canonical_components$source,
      v14_source = canonical_v14$path,
      final_weights =
        canonical_fit$final_weights,
      penalty_curves =
        canonical_fit$penalty_curves,
      respondent_gain =
        canonical_bootstrap$respondent_gain,
      bootstrap = canonical_bootstrap$bootstrap
    ),
    file.path(output_dir, "canonical_result.rds")
  )
  cat("\nCanonical result:\n")
  print(canonical_summary, digits = 9)
  cat("\nCanonical head weights:\n")
  print(canonical_weights, digits = 7)

  run_repeated <- repeated_mode == "always" ||
    (
      repeated_mode == "auto" &&
      isTRUE(
        canonical_summary$canonical_positive
      )
    )
  if (!run_repeated) {
    reason <- if (repeated_mode == "never") {
      "Repeated CV disabled by TWO_HEAD_REPEATED=never."
    } else {
      paste0(
        "Repeated CV not triggered because the ",
        "canonical point gain was not positive."
      )
    }
    verdict <- paste(
      "REJECT two-head ensemble; retain exact v14.",
      reason,
      sprintf(
        "Canonical gain %.9f; 95%% CI [%.9f, %.9f].",
        canonical_summary$point_gain,
        canonical_summary$lower_95,
        canonical_summary$upper_95
      )
    )
    writeLines(
      verdict,
      file.path(output_dir, "verdict.txt")
    )
    cat("\n", verdict, "\n", sep = "")
    return(invisible(list(
      canonical = canonical_summary,
      repeated = NULL,
      verdict = verdict
    )))
  }

  cat(
    "\nRepeated-CV escalation activated (",
    repeated_mode,
    ").\n",
    sep = ""
  )
  case_levels <- sort(unique(train$Case))
  seed_names <- as.character(all_outer_seeds)
  case_gain_matrix <- matrix(
    NA_real_,
    nrow = length(case_levels),
    ncol = length(seed_names),
    dimnames = list(
      as.character(case_levels), seed_names
    )
  )
  case_gain_matrix[
    , as.character(canonical_seed)
  ] <- canonical_bootstrap$respondent_gain
  repeat_rows <- list(
    `4821` = data.frame(
      seed = canonical_seed,
      baseline_logloss = canonical_baseline_loss,
      candidate_logloss =
        canonical_candidate_loss,
      gain = canonical_baseline_loss -
        canonical_candidate_loss
    )
  )
  weight_rows <- list(
    `4821` = canonical_weights
  )
  penalty_rows <- list(
    `4821` = canonical_penalties
  )

  for (seed in additional_seeds) {
    seed_name <- as.character(seed)
    cat(sprintf(
      "\n######## repeated CV seed %d ########\n",
      seed
    ))
    repeat_v14 <- v14_registry[[seed_name]]
    repeat_components <- load_seed_components(
      train, truth, seed, repeat_v14
    )
    repeat_fit <- run_nested_crossfit(
      train,
      truth,
      repeat_components$components,
      repeat_v14$fold_map,
      seed
    )
    baseline_loss <- log_loss_matrix_local(
      truth, repeat_v14$candidate_prediction
    )
    candidate_loss <- log_loss_matrix_local(
      truth, repeat_fit$prediction
    )
    case_gain <- respondent_gain(
      truth,
      repeat_v14$candidate_prediction,
      repeat_fit$prediction,
      train$Case
    )
    case_gain_matrix[, seed_name] <- case_gain
    repeat_rows[[seed_name]] <- data.frame(
      seed = seed,
      baseline_logloss = baseline_loss,
      candidate_logloss = candidate_loss,
      gain = baseline_loss - candidate_loss
    )
    weight_rows[[seed_name]] <- cbind(
      data.frame(seed = seed),
      repeat_fit$final_weights
    )
    penalty_rows[[seed_name]] <- cbind(
      data.frame(seed = seed),
      repeat_fit$penalty_curves
    )
    saveRDS(
      list(
        experiment_id = experiment_id,
        seed = seed,
        baseline_prediction =
          repeat_v14$candidate_prediction,
        candidate_prediction =
          repeat_fit$prediction,
        fold_map = repeat_v14$fold_map,
        component_source =
          repeat_components$source,
        v14_source = repeat_v14$path,
        final_weights =
          repeat_fit$final_weights,
        penalty_curves =
          repeat_fit$penalty_curves,
        respondent_gain = case_gain
      ),
      file.path(
        output_dir,
        sprintf("repeat_result_%d.rds", seed)
      )
    )
  }

  repeated_by_seed <- do.call(
    rbind, repeat_rows
  )
  average_case_gain <- rowMeans(case_gain_matrix)
  pooled <- summarize_case_gain(
    average_case_gain,
    bootstrap_replicates,
    canonical_seed
  )
  repeated_summary <- cbind(
    data.frame(
      experiment = experiment_id,
      stage = "repeated_cv"
    ),
    pooled$summary,
    data.frame(
      positive_repeats =
        sum(repeated_by_seed$gain > 0),
      n_repeats = nrow(repeated_by_seed)
    )
  )
  propensity <- fit_test_propensity(train, test)
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
  repeated_summary$test_like_30_point_gain <-
    top_30$point_gain[[1L]]
  repeated_summary$test_like_30_lower_95 <-
    top_30$lower_95[[1L]]
  repeated_summary$promote <-
    repeated_summary$point_gain > 0 &
    repeated_summary$lower_95 > 0 &
    repeated_summary$positive_repeats >= 5L &
    repeated_summary$test_like_30_point_gain >= 0 &
    repeated_summary$test_like_30_lower_95 >=
      -0.001

  all_weights <- do.call(rbind, weight_rows)
  all_penalties <- do.call(rbind, penalty_rows)
  write.csv(
    repeated_by_seed,
    file.path(
      output_dir, "repeated_cv_by_seed.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    repeated_summary,
    file.path(
      output_dir, "repeated_cv_summary.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    all_weights,
    file.path(
      output_dir, "repeated_cv_head_weights.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    all_penalties,
    file.path(
      output_dir, "repeated_cv_penalty_curves.csv"
    ),
    row.names = FALSE
  )
  write.csv(
    test_like,
    file.path(
      output_dir,
      "test_like_respondent_diagnostic.csv"
    ),
    row.names = FALSE
  )
  saveRDS(
    list(
      experiment_id = experiment_id,
      repeated_by_seed = repeated_by_seed,
      repeated_summary = repeated_summary,
      case_gain_matrix = case_gain_matrix,
      average_case_gain = average_case_gain,
      pooled_bootstrap = pooled$bootstrap,
      head_weights = all_weights,
      penalty_curves = all_penalties,
      test_propensity = propensity$propensity,
      test_like_diagnostic = test_like
    ),
    file.path(
      output_dir, "repeated_cv_result.rds"
    )
  )

  if (isTRUE(repeated_summary$promote)) {
    verdict <- paste0(
      "PROMOTE two-head ensemble for a separate ",
      "full-data build audit. Do not submit until the ",
      "candidate is reproduced and checked."
    )
  } else {
    verdict <- paste0(
      "REJECT two-head ensemble; retain the exact ",
      "set-context v14 submission."
    )
  }
  writeLines(
    verdict,
    file.path(output_dir, "verdict.txt")
  )
  cat("\nRepeated-CV result:\n")
  print(repeated_summary, digits = 9)
  cat("\nTest-like respondent diagnostic:\n")
  print(test_like, digits = 9)
  cat("\n", verdict, "\n", sep = "")
  invisible(list(
    canonical = canonical_summary,
    repeated = repeated_summary,
    test_like = test_like,
    verdict = verdict
  ))
}

run_experiment()
