# ==============================================================================
# 06_rf_kmeans.R
# Random Forest with K-Means Clustering Feature Engineering
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

# 3. Train/Validation Split (80/20)
set.seed(42) 
train_index <- sample(1:nrow(train_clean), 0.8 * nrow(train_clean))
local_train <- train_clean[train_index, ]
local_val <- train_clean[-train_index, ]

# ==============================================================================
# 4. K-Means Feature Engineering (Local Validation)
# ==============================================================================
print("Engineering K-Means Clusters...")

# Identify numeric columns AND strictly drop zero-variance columns so scale() doesn't divide by 0
numeric_cols <- names(local_train)[sapply(local_train, function(x) {
  is.numeric(x) && length(unique(na.omit(x))) > 1
})]

# Scale the training data safely
train_num_scaled <- scale(local_train[numeric_cols])
train_centers <- attr(train_num_scaled, "scaled:center")
train_scales <- attr(train_num_scaled, "scaled:scale")

# Train K-means (using 5 clusters as a starting persona grouping, with increased max iterations)
set.seed(42)
km_model <- kmeans(train_num_scaled, centers = 5, nstart = 25, iter.max = 100, algorithm = "MacQueen")
# Add the new Cluster feature to the training set
local_train$Cluster <- as.factor(paste0("C_", km_model$cluster))

# To prevent data leakage, we MUST scale the validation set using the training parameters
val_num_scaled <- scale(local_val[numeric_cols], center = train_centers, scale = train_scales)

# Helper function to assign new data to the closest existing K-means cluster
predict_kmeans <- function(data, centers) {
  distances <- apply(data, 1, function(row) {
    apply(centers, 1, function(center) sum((row - center)^2))
  })
  return(apply(distances, 2, which.min))
}

# Assign clusters to the validation set and attach the feature
val_clusters <- predict_kmeans(val_num_scaled, km_model$centers)
local_val$Cluster <- as.factor(paste0("C_", val_clusters))

# ==============================================================================
# 5. Train and Evaluate Random Forest
# ==============================================================================
print("Training Ranger Random Forest on clustered data...")

rf_model <- ranger(
  formula = Choice ~ ., 
  data = local_train, 
  num.trees = 500,
  probability = TRUE,
  importance = 'impurity',
  seed = 42
)

# Multi-Class Log Loss Helper
calculate_log_loss <- function(actual_factor, predicted_probs) {
  actual_matrix <- model.matrix(~ actual_factor - 1)
  eps <- 1e-15
  predicted_probs <- pmax(pmin(predicted_probs, 1 - eps), eps)
  -mean(rowSums(actual_matrix * log(predicted_probs)))
}

rf_val_probs <- predict(rf_model, data = local_val)$predictions
rf_loss <- calculate_log_loss(local_val$Choice, rf_val_probs)
print(paste(">>> LOCAL VALIDATION LOG LOSS:", round(rf_loss, 5)))

# ==============================================================================
# 6. Final Test Predictions on 100% Dataset
# ==============================================================================
print("Retraining K-Means and RF on 100% of data for Kaggle submission...")

# Sweep for valid numeric columns on the full dataset
valid_numeric_cols_full <- names(train_clean)[sapply(train_clean, function(x) {
  is.numeric(x) && length(unique(na.omit(x))) > 1
})]

# Re-run K-means on the FULL training set safely with increased iterations
full_num_scaled <- scale(train_clean[valid_numeric_cols_full])
full_centers <- attr(full_num_scaled, "scaled:center")
full_scales <- attr(full_num_scaled, "scaled:scale")

set.seed(42)
km_final <- kmeans(full_num_scaled, centers = 5, nstart = 25, iter.max = 100, algorithm = "MacQueen")
train_clean$Cluster <- as.factor(paste0("C_", km_final$cluster))

# Assign clusters to the Kaggle test set
test_num_scaled <- scale(test_clean[valid_numeric_cols_full], center = full_centers, scale = full_scales)
test_clusters <- predict_kmeans(test_num_scaled, km_final$centers)
test_clean$Cluster <- as.factor(paste0("C_", test_clusters))

# Train final RF model
final_rf_model <- ranger(
  formula = Choice ~ ., 
  data = train_clean, 
  num.trees = 500,
  probability = TRUE,
  seed = 42
)

# Generate final probabilities
test_probs <- predict(final_rf_model, data = test_clean)$predictions

# Format submission dynamically
submission_006 <- sample_sub %>%
  mutate(
    Ch1 = test_probs[, "Alternative_1"],
    Ch2 = test_probs[, "Alternative_2"],
    Ch3 = test_probs[, "Alternative_3"],
    Ch4 = test_probs[, "Alternative_4"]
  ) %>%
  select(all_of(names(sample_sub)))

# Save submission
write_csv(submission_006, "../results/submission_006_rf_kmeans_FIXED.csv")
print("submission_006_rf_kmeans.csv has been successfully generated!")