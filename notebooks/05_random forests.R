# ==============================================================================
# 05_random_forest.R
# Random Forest classification using the 'ranger' package
# ==============================================================================

library(tidyverse)
library(ranger)

# 1. Load Data
train_raw <- read_csv("../data/train.csv")
test_raw <- read_csv("../data/test.csv")
sample_sub <- read_csv("../data/sample_submission.csv")

# 2. Data Preprocessing
process_data <- function(df, is_train = TRUE) {
  df_processed <- df
  if (is_train) {
    df_processed <- df_processed %>%
      mutate(
        Choice = case_when(
          Ch1 == 1 ~ "Alternative_1",
          Ch2 == 1 ~ "Alternative_2",
          Ch3 == 1 ~ "Alternative_3",
          Ch4 == 1 ~ "Alternative_4"
        ),
        Choice = as.factor(Choice)
      ) %>%
      select(-Ch1, -Ch2, -Ch3, -Ch4)
  }
  # Drop ID columns
  df_processed <- df_processed %>%
    select(-any_of(c("Case", "No", "Task")))
  return(df_processed)
}

train_clean <- process_data(train_raw, is_train = TRUE)
test_clean <- process_data(test_raw, is_train = FALSE)

# 3. Train/Validation Split (80/20)
set.seed(42) 
train_index <- sample(1:nrow(train_clean), 0.8 * nrow(train_clean))
local_train <- train_clean[train_index, ]
local_val <- train_clean[-train_index, ]

# Helper function for Multi-Class Log Loss
calculate_log_loss <- function(actual_factor, predicted_probs) {
  actual_matrix <- model.matrix(~ actual_factor - 1)
  eps <- 1e-15
  predicted_probs <- pmax(pmin(predicted_probs, 1 - eps), eps)
  -mean(rowSums(actual_matrix * log(predicted_probs)))
}

# ==============================================================================
# 4. Train the Random Forest Model
# ==============================================================================
print("Training Ranger Random Forest on 80% split...")

# We set probability = TRUE to get percentage outputs instead of hard classifications
rf_model <- ranger(
  formula = Choice ~ ., 
  data = local_train, 
  num.trees = 500,        # Standard starting amount of trees
  probability = TRUE,     # Crucial for log-loss
  importance = 'impurity',
  seed = 42
)

# Predict probabilities on the 20% validation set
# The $predictions element extracts the actual probability matrix
rf_val_probs <- predict(rf_model, data = local_val)$predictions

# Evaluate Local Log Loss
rf_loss <- calculate_log_loss(local_val$Choice, rf_val_probs)
print(paste(">>> LOCAL VALIDATION LOG LOSS:", round(rf_loss, 5)))

# ==============================================================================
# 5. Final Test Predictions & Submission
# ==============================================================================
print("Retraining final model on 100% of data...")

# Retrain model on the full dataset
final_rf_model <- ranger(
  formula = Choice ~ ., 
  data = train_clean, 
  num.trees = 500,
  probability = TRUE,
  seed = 42
)

# Generate probabilities for the Kaggle test set
test_probs <- predict(final_rf_model, data = test_clean)$predictions

# Format submission dynamically based on sample_sub to guarantee no Kaggle errors
submission_005 <- sample_sub %>%
  mutate(
    Ch1 = test_probs[, "Alternative_1"],
    Ch2 = test_probs[, "Alternative_2"],
    Ch3 = test_probs[, "Alternative_3"],
    Ch4 = test_probs[, "Alternative_4"]
  ) %>%
  select(all_of(names(sample_sub)))

# Save the optimal submission file
write_csv(submission_005, "../submissions/submission_005_rf.csv")
print("submission_005_rf.csv has been successfully generated!")