# ==============================================================================
# 03_multinom_upgraded.R
# Upgraded Multinomial Logistic Regression with Scaling and Regularization
# ==============================================================================

# 1. Load Libraries
library(tidyverse)
library(nnet)

# 2. Load Data
# (Adjust the file paths if your data folder is located elsewhere)
train_raw <- read_csv("../data/train.csv")
test_raw <- read_csv("../data/test.csv")
sample_sub <- read_csv("../data/sample_submission.csv")

# 3. Data Preprocessing
# Combine Ch1-Ch4 into a single 'Choice' column and drop IDs
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
  
  # Drop ID columns to prevent the model from learning them
  df_processed <- df_processed %>%
    select(-any_of(c("Case", "No", "Task")))
  
  return(df_processed)
}

train_clean <- process_data(train_raw, is_train = TRUE)
test_clean <- process_data(test_raw, is_train = FALSE)

# 4. Train/Validation Split (80/20)
set.seed(42) # Ensure reproducible splits
train_index <- sample(1:nrow(train_clean), 0.8 * nrow(train_clean))
local_train <- train_clean[train_index, ]
local_val <- train_clean[-train_index, ]

# 5. Feature Scaling (Crucial for the upgraded model)
# Identify the NAMES of the numeric columns
numeric_cols <- names(local_train)[sapply(local_train, is.numeric)]

# Explicitly calculate and save the mean and standard deviation for each numeric column
train_centers <- apply(local_train[numeric_cols], 2, mean, na.rm = TRUE)
train_scales <- apply(local_train[numeric_cols], 2, sd, na.rm = TRUE)

# Apply these EXACT training scales to all three datasets safely using column names
local_train[numeric_cols] <- scale(local_train[numeric_cols], center = train_centers, scale = train_scales)
local_val[numeric_cols]   <- scale(local_val[numeric_cols], center = train_centers, scale = train_scales)
test_clean[numeric_cols]  <- scale(test_clean[numeric_cols], center = train_centers, scale = train_scales)

# ==========================================
# 5.5 Drop Zero-Variance Columns
# Remove any columns that only have 1 unique value
# ==========================================
valid_cols <- sapply(local_train, function(x) length(unique(na.omit(x))) > 1)

# Apply this filter to both training and validation sets
local_train <- local_train[, valid_cols]
local_val <- local_val[, valid_cols]

# 6. Train the Upgraded Model
# ... (continue with your multinom code here) ...
# 6. Train the Upgraded Model
# Added decay for L2 regularization and maxit to prevent early stopping
model_upgraded <- multinom(Choice ~ ., 
                           data = local_train, 
                           decay = 0.1,       
                           maxit = 1000,      
                           MaxNWts = 5000,
                           trace = FALSE) # Set trace=TRUE if you want to watch the iterations print out

# 7. Local Validation & Log Loss Calculation
val_probs <- predict(model_upgraded, newdata = local_val, type = "probs")

# Helper function for Multi-Class Log Loss
calculate_log_loss <- function(actual_factor, predicted_probs) {
  actual_matrix <- model.matrix(~ actual_factor - 1)
  # Prevent log(0) by clipping probabilities slightly
  eps <- 1e-15
  predicted_probs <- pmax(pmin(predicted_probs, 1 - eps), eps)
  -mean(rowSums(actual_matrix * log(predicted_probs)))
}

my_loss <- calculate_log_loss(local_val$Choice, val_probs)
print(paste("Upgraded Local Validation Log Loss:", my_loss))

# ==============================================================================
# 8. Final Test Predictions & Submission
# ==============================================================================

# Sweep the full 100% dataset for columns with variance, but save the NAMES
valid_col_names <- names(train_clean)[sapply(train_clean, function(x) length(unique(na.omit(x))) > 1)]

# Apply filter to the training set
train_clean_filtered <- train_clean[, valid_col_names]

# For the test set, we only keep the valid columns that actually exist in the test data
# (This safely ignores the 'Choice' target column since test_clean doesn't have it)
test_cols_to_keep <- intersect(valid_col_names, names(test_clean))
test_clean_filtered <- test_clean[, test_cols_to_keep]

# Identify the numeric columns in this newly filtered dataset
numeric_cols_full <- names(train_clean_filtered)[sapply(train_clean_filtered, is.numeric)]

# Calculate centers and scales from the 100% dataset
full_centers <- apply(train_clean_filtered[numeric_cols_full], 2, mean, na.rm = TRUE)
full_scales <- apply(train_clean_filtered[numeric_cols_full], 2, sd, na.rm = TRUE)

# Scale both the full training set and the test set using these new parameters
train_clean_filtered[numeric_cols_full] <- scale(train_clean_filtered[numeric_cols_full], 
                                                 center = full_centers, scale = full_scales)
test_clean_filtered[numeric_cols_full] <- scale(test_clean_filtered[numeric_cols_full], 
                                                center = full_centers, scale = full_scales)

# Retrain model on the fully cleaned and scaled 100% dataset
final_model <- multinom(Choice ~ ., 
                        data = train_clean_filtered, 
                        decay = 0.1, 
                        maxit = 1000, 
                        MaxNWts = 5000,
                        trace = FALSE)

# Generate final probabilities for the test set
test_probs <- predict(final_model, newdata = test_clean_filtered, type = "probs")

# ==============================================================================
# Format submission perfectly based on sample_sub structure
# ==============================================================================

# Map your model's probability outputs back to the original Kaggle column names
submission_003 <- sample_sub %>%
  mutate(
    Ch1 = test_probs[, "Alternative_1"],
    Ch2 = test_probs[, "Alternative_2"],
    Ch3 = test_probs[, "Alternative_3"],
    Ch4 = test_probs[, "Alternative_4"]
  ) %>%
  # Force the columns to be in the EXACT order and spelling as sample_submission.csv
  select(all_of(names(sample_sub)))

# Save the new submission file
write_csv(submission_003, "../submissions/submission_003.csv")
print("submission_003.csv has been successfully generated and formatted!")