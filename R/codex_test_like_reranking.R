source("R/codex_shared_utility_common.R")

dir.create(
  shared_utility_output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

train <- read.csv("csv files/train.csv")
test <- read.csv("csv files/test.csv")
base <- readRDS("data_processed/oof_ensemble_v10.rds")
truth <- base$oof_truth
fold_map <- base$fold_of_case
row_fold <- unname(fold_map[as.character(train$Case)])

v11 <- 0.8 * base$oof_mlogit + 0.2 * base$oof_xgb
current <- readRDS(
  "data_processed/codex_behavioral_round/mlp_precision.rds"
)$crossfit_prediction
mlp_component <- readRDS(
  "data_processed/codex_behavioral_round/mlp_oof.rds"
)$oof[[1]]
triple_mlogit <- readRDS(
  "data_processed/codex_triples/triple_oof.rds"
)$oof$triple_price_income_miles
triple_weights <- list(
  mlogit = c(0.76, 0.72, 0.72, 0.72, 0.74),
  xgb = c(0.08, 0.10, 0.12, 0.14, 0.12),
  mlp = c(0.16, 0.18, 0.16, 0.14, 0.14)
)
triple_mlp <-
  triple_weights$mlogit[row_fold] * triple_mlogit +
  triple_weights$xgb[row_fold] * base$oof_xgb +
  triple_weights$mlp[row_fold] * mlp_component
full_stacking <- readRDS(
  "data_processed/codex_deep_stack/full_stacking.rds"
)
eight_component <-
  full_stacking$results$augmented8_arithmetic$prediction

candidates <- list(
  ensemble_v11 = v11,
  submitted_mlp_blend = current,
  triple_mlp = triple_mlp,
  eight_component_arithmetic = eight_component
)
stopifnot(
  abs(log_loss_matrix(truth, candidates$ensemble_v11) -
    1.145094) < 1e-6,
  abs(log_loss_matrix(truth, candidates$submitted_mlp_blend) -
    1.143789442) < 1e-8,
  abs(log_loss_matrix(truth, candidates$triple_mlp) -
    1.143327611174143) < 1e-12,
  abs(log_loss_matrix(
    truth, candidates$eight_component_arithmetic
  ) - 1.14211190984165) < 1e-12
)

train_respondent <- distinct_respondents(train)
test_respondent <- distinct_respondents(test)
train_domain <- train_respondent[, shift_resp_vars, drop = FALSE]
test_domain <- test_respondent[, shift_resp_vars, drop = FALSE]
train_domain$is_test <- 0L
test_domain$is_test <- 1L
domain <- rbind(train_domain, test_domain)

auc <- function(prediction, label) {
  rank_value <- rank(prediction)
  n_positive <- sum(label == 1L)
  n_negative <- sum(label == 0L)
  (
    sum(rank_value[label == 1L]) -
      n_positive * (n_positive + 1) / 2
  ) / (n_positive * n_negative)
}

set.seed(123)
domain_fold <- sample(rep(1:5, length.out = nrow(domain)))
domain_oof <- numeric(nrow(domain))
for (fold in 1:5) {
  fitted <- glm(
    is_test ~ .,
    data = domain[domain_fold != fold, , drop = FALSE],
    family = binomial()
  )
  domain_oof[domain_fold == fold] <- predict(
    fitted,
    newdata = domain[domain_fold == fold, , drop = FALSE],
    type = "response"
  )
}
domain_auc <- auc(domain_oof, domain$is_test)
stopifnot(abs(domain_auc - 0.634) < 0.001)

full_domain_fit <- glm(
  is_test ~ .,
  data = domain,
  family = binomial()
)
train_test_probability <- as.numeric(predict(
  full_domain_fit,
  newdata = train_domain,
  type = "response"
))
case_score <- data.frame(
  Case = train_respondent$Case,
  test_probability = train_test_probability,
  incomea = train_respondent$incomea,
  incomeind = train_respondent$incomeind,
  segmentind = train_respondent$segmentind
)
case_score <- case_score[
  order(case_score$test_probability, decreasing = TRUE), ,
  drop = FALSE
]
write_result_csv(
  case_score,
  file.path(shared_utility_output_dir, "test_like_case_scores.csv")
)

fractions <- c(1, 0.30, 0.25, 0.20)
score_rows <- list()
bootstrap_rows <- list()
for (fraction in fractions) {
  respondent_count <- if (fraction == 1) {
    nrow(case_score)
  } else {
    ceiling(nrow(case_score) * fraction)
  }
  selected_cases <- case_score$Case[seq_len(respondent_count)]
  selected_rows <- which(train$Case %in% selected_cases)
  subset_truth <- truth[selected_rows, , drop = FALSE]
  subset_scores <- vapply(candidates, function(prediction) {
    log_loss_matrix(
      subset_truth,
      prediction[selected_rows, , drop = FALSE]
    )
  }, numeric(1))
  ranks <- rank(subset_scores, ties.method = "min")
  label <- if (fraction == 1) {
    "all_train"
  } else {
    sprintf("top_%02d_percent", round(100 * fraction))
  }
  score_rows[[label]] <- data.frame(
    subset = label,
    fraction = fraction,
    respondents = respondent_count,
    tasks = length(selected_rows),
    candidate = names(subset_scores),
    logloss = as.numeric(subset_scores),
    rank = as.integer(ranks)
  )

  for (candidate in setdiff(
    names(candidates), "submitted_mlp_blend"
  )) {
    bootstrap <- respondent_bootstrap_comparison(
      subset_truth,
      candidates$submitted_mlp_blend[
        selected_rows, , drop = FALSE
      ],
      candidates[[candidate]][selected_rows, , drop = FALSE],
      train$Case[selected_rows],
      replicates = 20000L
    )
    bootstrap$subset <- label
    bootstrap$candidate <- candidate
    bootstrap_rows[[paste(label, candidate, sep = "_")]] <-
      bootstrap
  }
}

scores <- do.call(rbind, score_rows)
bootstrap <- do.call(rbind, bootstrap_rows)
diagnostic <- data.frame(
  adversarial_cv_auc = domain_auc,
  train_respondents = nrow(train_respondent),
  test_respondents = nrow(test_respondent)
)
write_result_csv(
  diagnostic,
  file.path(shared_utility_output_dir, "test_like_diagnostic.csv")
)
write_result_csv(
  scores,
  file.path(shared_utility_output_dir, "test_like_reranking.csv")
)
write_result_csv(
  bootstrap,
  file.path(shared_utility_output_dir, "test_like_bootstrap.csv")
)
saveRDS(
  list(
    diagnostic = diagnostic,
    scores = scores,
    bootstrap = bootstrap,
    case_scores = case_score,
    candidates = candidates
  ),
  file.path(shared_utility_output_dir, "test_like_reranking.rds")
)
print(diagnostic, digits = 9)
print(scores, digits = 9)
print(bootstrap, digits = 9)
