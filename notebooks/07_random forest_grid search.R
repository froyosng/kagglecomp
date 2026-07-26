# ==============================================================================
# 07_rf_gridsearch.R
# Random Forest Hyperparameter Optimization (Grid Search)
# ==============================================================================

library(tidyverse)
library(ranger)

# 1. Load Data
train_raw <- read_csv("../data/train.csv")
test_raw <- read_csv("../data/test.csv")
sample_sub <- read_csv("../data/sample_submission.csv")

# 2. Data Preprocessing (Clean baseline, no K-means distraction)
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
  df_processed <- df_processed %>% select(-any_of(c("Case", "No", "Task")))
  return(df_processed)
}

train_clean <- process_data(train_raw, is_train = TRUE)
test_clean <- process_data(test_raw, is_train = FALSE)

# 3. Train/Validation Split (80/20)
set.seed(42) 
train_index <- sample(1:nrow(train_clean), 0.8 * nrow(train_clean))
local_train <- train_clean[train_index, ]
local_val <- train_clean[-train_index, ]

# Multi-Class Log Loss Helper Function
calculate_log_loss <- function(actual_factor, predicted_probs) {
  actual_matrix <- model.matrix(~ actual_factor - 1)
  eps <- 1e-15
  predicted_probs <- pmax(pmin(predicted_probs, 1 - eps), eps)
  -mean(rowSums(actual_matrix * log(predicted_probs)))
}

# ==============================================================================
# 4. Define the Search Space
# ==============================================================================
# mtry: Number of variables to split at each node. 
# min.node.size: Minimum size of terminal nodes. Higher values constrain the tree.
hyper_grid <- expand.grid(
  mtry = c(2, 4, 6, 8, 10),
  min.node.size = c(5, 10, 20),
  log_loss = NA
)

print(paste("Beginning grid search over", nrow(hyper_grid), "parameter combinations..."))

# ==============================================================================
# 5. Execute Grid Search
# ==============================================================================
for(i in 1:nrow(hyper_grid)) {
  
  # Train model with current grid parameters
  rf_model <- ranger(
    formula = Choice ~ ., 
    data = local_train, 
    num.trees = 500,
    mtry = hyper_grid$mtry[i],
    min.node.size = hyper_grid$min.node.size[i],
    probability = TRUE,
    seed = 42
  )
  
  # Predict and evaluate
  val_probs <- predict(rf_model, data = local_val)$predictions
  current_loss <- calculate_log_loss(local_val$Choice, val_probs)
  
  # Store result
  hyper_grid$log_loss[i] <- current_loss
  
  print(paste0("Run ", i, "/", nrow(hyper_grid), 
               " | mtry: ", hyper_grid$mtry[i], 
               " | min.node.size: ", hyper_grid$min.node.size[i], 
               " | Loss: ", round(current_loss, 5)))
}

# Identify the absolute minimum log-loss from the search space
best_params <- hyper_grid[which.min(hyper_grid$log_loss), ]
print("--------------------------------------------------")
print("OPTIMIZATION COMPLETE. BEST PARAMETERS FOUND:")
print(best_params)
print("--------------------------------------------------")

# ==============================================================================
# 6. Train Final Model & Export
# ==============================================================================
print("Retraining final model on 100% of data using optimized parameters...")

final_rf_model <- ranger(
  formula = Choice ~ ., 
  data = train_clean, 
  num.trees = 500,
  mtry = best_params$mtry,
  min.node.size = best_params$min.node.size,
  probability = TRUE,
  seed = 42
)

# Generate probabilities for Kaggle
test_probs <- predict(final_rf_model, data = test_clean)$predictions

submission_007 <- sample_sub %>%
  mutate(
    Ch1 = test_probs[, "Alternative_1"],
    Ch2 = test_probs[, "Alternative_2"],
    Ch3 = test_probs[, "Alternative_3"],
    Ch4 = test_probs[, "Alternative_4"]
  ) %>%
  select(all_of(names(sample_sub)))

# Save directly to the newly designated results folder
write_csv(submission_007, "../results/submission_007_rf_tuned.csv")
print("submission_007_rf_tuned.csv has been successfully generated in the results folder!")