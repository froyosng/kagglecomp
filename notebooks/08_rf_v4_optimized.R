# ==============================================================================
# 09_rf_advanced_tuned.R
# Advanced Random Forest Optimization (Smooth Probabilities & Class Weights)
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

# Calculate Inverse Class Weights for the 80% split
# This forces the model to pay equal attention to all choices, even the rare ones
class_freq <- table(local_train$Choice) / nrow(local_train)
inv_weights <- 1 / class_freq
# Ensure the weights match the exact order of the factor levels
ordered_weights <- inv_weights[levels(local_train$Choice)]

# Multi-Class Log Loss Helper Function
calculate_log_loss <- function(actual_factor, predicted_probs) {
  actual_matrix <- model.matrix(~ actual_factor - 1)
  eps <- 1e-15
  predicted_probs <- pmax(pmin(predicted_probs, 1 - eps), eps)
  -mean(rowSums(actual_matrix * log(predicted_probs)))
}

# ==============================================================================
# 4. Define the Expanded Search Space
# ==============================================================================
# Added splitrule to test ExtraTrees vs Standard Gini
hyper_grid <- expand.grid(
  mtry = c(2, 4, 6, 8, 10),
  min.node.size = c(5, 10, 20),
  splitrule = c("gini", "extratrees"),
  log_loss = NA
)

print(paste("Beginning advanced grid search over", nrow(hyper_grid), "parameter combinations..."))
print("This will take slightly longer due to 1,500 trees per model.")

# ==============================================================================
# 5. Execute Grid Search
# ==============================================================================
for(i in 1:nrow(hyper_grid)) {
  
  rf_model <- ranger(
    formula = Choice ~ ., 
    data = local_train, 
    num.trees = 1500,  # Massive ensemble for perfectly smooth probability curves
    mtry = hyper_grid$mtry[i],
    min.node.size = hyper_grid$min.node.size[i],
    splitrule = as.character(hyper_grid$splitrule[i]),
    class.weights = ordered_weights, # Injecting the anti-bias weights
    probability = TRUE,
    seed = 42
  )
  
  val_probs <- predict(rf_model, data = local_val)$predictions
  current_loss <- calculate_log_loss(local_val$Choice, val_probs)
  
  hyper_grid$log_loss[i] <- current_loss
  
  print(paste0("Run ", i, "/", nrow(hyper_grid), 
               " | mtry: ", hyper_grid$mtry[i], 
               " | node: ", hyper_grid$min.node.size[i], 
               " | rule: ", hyper_grid$splitrule[i],
               " | Loss: ", round(current_loss, 5)))
}

best_params <- hyper_grid[which.min(hyper_grid$log_loss), ]
print("--------------------------------------------------")
print("OPTIMIZATION COMPLETE. BEST PARAMETERS FOUND:")
print(best_params)
print("--------------------------------------------------")

# ==============================================================================
# 6. Train Final Model & Export
# ==============================================================================
print("Retraining final model on 100% of data using optimized parameters...")

# Recalculate weights for the 100% dataset
full_class_freq <- table(train_clean$Choice) / nrow(train_clean)
full_inv_weights <- 1 / full_class_freq
full_ordered_weights <- full_inv_weights[levels(train_clean$Choice)]

final_rf_model <- ranger(
  formula = Choice ~ ., 
  data = train_clean, 
  num.trees = 1500,
  mtry = best_params$mtry,
  min.node.size = best_params$min.node.size,
  splitrule = as.character(best_params$splitrule),
  class.weights = full_ordered_weights,
  probability = TRUE,
  seed = 42
)

test_probs <- predict(final_rf_model, data = test_clean)$predictions

submission_009 <- sample_sub %>%
  mutate(
    Ch1 = test_probs[, "Alternative_1"],
    Ch2 = test_probs[, "Alternative_2"],
    Ch3 = test_probs[, "Alternative_3"],
    Ch4 = test_probs[, "Alternative_4"]
  ) %>%
  select(all_of(names(sample_sub)))

# Save directly to the results folder
write_csv(submission_009, "../results/submission_009_rf_advanced_tuned.csv")
print("submission_009_rf_advanced_tuned.csv has been successfully generated in the results folder!")