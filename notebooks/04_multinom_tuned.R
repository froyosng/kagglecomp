# ==============================================================================
# 04_multinom_tuned.R
# Grid Search for Optimal L2 Regularization (Decay)
# ==============================================================================

library(tidyverse)
library(nnet)

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

# 4. Drop Zero-Variance Columns
valid_col_names <- names(local_train)[sapply(local_train, function(x) length(unique(na.omit(x))) > 1)]
local_train <- local_train[, valid_col_names]
local_val <- local_val[, intersect(valid_col_names, names(local_val))]

# 5. Feature Scaling
numeric_cols <- names(local_train)[sapply(local_train, is.numeric)]
train_centers <- apply(local_train[numeric_cols], 2, mean, na.rm = TRUE)
train_scales <- apply(local_train[numeric_cols], 2, sd, na.rm = TRUE)

local_train[numeric_cols] <- scale(local_train[numeric_cols], center = train_centers, scale = train_scales)
local_val[numeric_cols]   <- scale(local_val[numeric_cols], center = train_centers, scale = train_scales)

# Helper function for Multi-Class Log Loss
calculate_log_loss <- function(actual_factor, predicted_probs) {
  actual_matrix <- model.matrix(~ actual_factor - 1)
  eps <- 1e-15
  predicted_probs <- pmax(pmin(predicted_probs, 1 - eps), eps)
  -mean(rowSums(actual_matrix * log(predicted_probs)))
}

# ==============================================================================
# 6. Grid Search for Optimal Decay
# ==============================================================================

# Define the sequence of decay values to test
decay_grid <- c(0, 0.001, 0.01, 0.1, 0.5, 1, 2)
results <- data.frame(Decay = numeric(), LogLoss = numeric())

print("Starting Grid Search...")

# Loop through each decay value
for (d in decay_grid) {
  # Train model silently
  temp_model <- multinom(Choice ~ ., 
                         data = local_train, 
                         decay = d,       
                         maxit = 1000,      
                         MaxNWts = 5000,
                         trace = FALSE)
  
  # Predict and evaluate
  temp_probs <- predict(temp_model, newdata = local_val, type = "probs")
  temp_loss <- calculate_log_loss(local_val$Choice, temp_probs)
  
  # Store result
  results <- rbind(results, data.frame(Decay = d, LogLoss = temp_loss))
  print(paste("Tested decay:", d, "- Log Loss:", round(temp_loss, 5)))
}

# Find the absolute best decay value
best_decay <- results$Decay[which.min(results$LogLoss)]
best_loss <- min(results$LogLoss)

print(paste(">>> BEST DECAY FOUND:", best_decay, "with Log Loss:", round(best_loss, 5)))

# ==============================================================================
# 7. Final Test Predictions & Submission (Using Best Decay)
# ==============================================================================

# Sweep the full 100% dataset for columns with variance
valid_cols_full <- names(train_clean)[sapply(train_clean, function(x) length(unique(na.omit(x))) > 1)]
train_clean_filtered <- train_clean[, valid_cols_full]
test_clean_filtered <- test_clean[, intersect(valid_cols_full, names(test_clean))]

# Identify numeric columns and scale the full dataset
numeric_cols_full <- names(train_clean_filtered)[sapply(train_clean_filtered, is.numeric)]
full_centers <- apply(train_clean_filtered[numeric_cols_full], 2, mean, na.rm = TRUE)
full_scales <- apply(train_clean_filtered[numeric_cols_full], 2, sd, na.rm = TRUE)

train_clean_filtered[numeric_cols_full] <- scale(train_clean_filtered[numeric_cols_full], 
                                                 center = full_centers, scale = full_scales)
test_clean_filtered[numeric_cols_full] <- scale(test_clean_filtered[numeric_cols_full], 
                                                center = full_centers, scale = full_scales)

# Retrain final model on the 100% dataset using the optimally found decay
final_model <- multinom(Choice ~ ., 
                        data = train_clean_filtered, 
                        decay = best_decay, 
                        maxit = 1000, 
                        MaxNWts = 5000,
                        trace = FALSE)

# Generate probabilities for the test set
test_probs <- predict(final_model, newdata = test_clean_filtered, type = "probs")

# Format submission dynamically based on sample_sub to guarantee no Kaggle errors
submission_004 <- sample_sub %>%
  mutate(
    Ch1 = test_probs[, "Alternative_1"],
    Ch2 = test_probs[, "Alternative_2"],
    Ch3 = test_probs[, "Alternative_3"],
    Ch4 = test_probs[, "Alternative_4"]
  ) %>%
  select(all_of(names(sample_sub)))

# Save the optimal submission file
write_csv(submission_004, "../submissions/submission_004_tuned.csv")
print("submission_004_tuned.csv has been successfully generated!")