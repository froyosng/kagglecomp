# ==============================================================================
# 011_rf_kfold_cv.R
# Random Forest Optimization with 5-Fold Cross Validation
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
  df_processed <- df_processed %>% select(-any_of(c("Case", "No", "Task")))
  return(df_processed)
}

train_clean <- process_data(train_raw, is_train = TRUE)
test_clean <- process_data(test_raw, is_train = FALSE)

# Multi-Class Log Loss Helper Function
calculate_log_loss <- function(actual_factor, predicted_probs) {
  actual_matrix <- model.matrix(~ actual_factor - 1)
  eps <- 1e-15
  predicted_probs <- pmax(pmin(predicted_probs, 1 - eps), eps)
  -mean(rowSums(actual_matrix * log(predicted_probs)))
}

# ==============================================================================
# 3. 5-Fold Cross-Validation Setup
# ==============================================================================
set.seed(42)
# Assign every row a random fold number between 1 and 5
fold_assignments <- sample(rep(1:5, length.out = nrow(train_clean)))

# Define a tighter search space to force simpler, generalized trees
hyper_grid <- expand.grid(
  mtry = c(2, 4, 6, 8),
  min.node.size = c(10, 20, 50), # Higher nodes = less overfitting
  cv_log_loss = NA
)

print(paste("Beginning 5-Fold CV Grid Search over", nrow(hyper_grid), "combinations..."))

# ==============================================================================
# 4. Execute Grid Search with K-Fold CV
# ==============================================================================
for(i in 1:nrow(hyper_grid)) {
  
  fold_losses <- numeric(5)
  
  for(f in 1:5) {
    # Isolate the current validation fold
    val_indices <- which(fold_assignments == f)
    fold_train <- train_clean[-val_indices, ]
    fold_val <- train_clean[val_indices, ]
    
    # Train the model on the remaining 4 folds
    rf_model <- ranger(
      formula = Choice ~ ., 
      data = fold_train, 
      num.trees = 500, 
      mtry = hyper_grid$mtry[i],
      min.node.size = hyper_grid$min.node.size[i],
      probability = TRUE,
      seed = 42
    )
    
    # Predict and evaluate on the isolated fold
    val_probs <- predict(rf_model, data = fold_val)$predictions
    fold_losses[f] <- calculate_log_loss(fold_val$Choice, val_probs)
  }
  
  # Calculate the robust average score across all 5 folds
  avg_loss <- mean(fold_losses)
  hyper_grid$cv_log_loss[i] <- avg_loss
  
  print(paste0("Run ", i, "/", nrow(hyper_grid), 
               " | mtry: ", hyper_grid$mtry[i], 
               " | node: ", hyper_grid$min.node.size[i], 
               " | 5-Fold Loss: ", round(avg_loss, 5)))
}

best_params <- hyper_grid[which.min(hyper_grid$cv_log_loss), ]
print("--------------------------------------------------")
print("OPTIMIZATION COMPLETE. BEST ROBUST PARAMETERS:")
print(best_params)
print("--------------------------------------------------")

# ==============================================================================
# 5. Train Final Model & Export
# ==============================================================================
print("Retraining final model on 100% of data using validated parameters...")

final_rf_model <- ranger(
  formula = Choice ~ ., 
  data = train_clean, 
  num.trees = 1500, # Increased to 1500 for highly smoothed Kaggle probabilities
  mtry = best_params$mtry,
  min.node.size = best_params$min.node.size,
  probability = TRUE,
  seed = 42
)

test_probs <- predict(final_rf_model, data = test_clean)$predictions

submission_011 <- sample_sub %>%
  mutate(
    Ch1 = test_probs[, "Alternative_1"],
    Ch2 = test_probs[, "Alternative_2"],
    Ch3 = test_probs[, "Alternative_3"],
    Ch4 = test_probs[, "Alternative_4"]
  ) %>%
  select(all_of(names(sample_sub)))

# Export to results
write_csv(submission_011, "../results/submission_011_rf_kfold.csv")
print("submission_011_rf_kfold.csv has been successfully generated in the results folder!")