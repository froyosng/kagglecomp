# Load necessary libraries
library(tidyverse)
library(nnet) 

# 1. Load the Data
# Ensure you are running this from your DataComp2026.Rproj root directory
train_data <- read_csv("data/train.csv")
test_data <- read_csv("data/test.csv")
sample_sub <- read_csv("data/sample_submission.csv")

# Source your teammate's custom R script
# (You may need to check exactly what the function inside is named)
source("R/log_loss.R")

# 2. Preprocess
# Combine Ch1, Ch2, Ch3, and Ch4 into a single 'Choice' column
train_data$Choice <- max.col(train_data[, c("Ch1", "Ch2", "Ch3", "Ch4")])

# IMPORTANT: Drop the original dummy target columns and ID columns
# so the model doesn't use them to cheat or get confused.
train_data <- train_data %>% 
  select(-Ch1, -Ch2, -Ch3, -Ch4, -Case, -No, -Task)

# Convert Choice to a factor for the multinom model
train_data$Choice <- as.factor(train_data$Choice)

# Create a local validation split (80% train, 20% validation)
set.seed(2026)
train_index <- sample(1:nrow(train_data), 0.8 * nrow(train_data))
local_train <- train_data[train_index, ]
local_val <- train_data[-train_index, ]

# 3. Train the Benchmark Model
# The '.' now uses all remaining columns (excluding the ones we dropped) as predictors
model_baseline <- multinom(Choice ~ ., data = local_train, MaxNWts = 2000)

# 4. Evaluate Locally
# Predict probabilities on the 20% validation set
val_probs <- predict(model_baseline, newdata = local_val, type = "probs")

# Convert the actual 'Choice' factor back into a one-hot encoded matrix
# so its dimensions perfectly match the 4 columns of val_probs
actual_matrix <- class.ind(local_val$Choice)

# Use Zhenhao's function to calculate your score
my_loss <- log_loss(actual = actual_matrix, pred = val_probs)
print(paste("Local Validation Log Loss:", my_loss))

# 5. Generate Final Predictions
# Retrain the model on 100% of the training data to get the best possible predictions
model_final <- multinom(Choice ~ ., data = train_data, MaxNWts = 2000)

# Predict probabilities on the Kaggle test set
test_probs <- predict(model_final, newdata = test_data, type = "probs")

# 6. Format Submission
# Map the matrix columns to the exact format Kaggle expects
submission_002 <- sample_sub %>%
  mutate(
    Alternative_1 = test_probs[, 1],
    Alternative_2 = test_probs[, 2],
    Alternative_3 = test_probs[, 3],
    Alternative_4 = test_probs[, 4]
  )

# Save the file to your submissions folder
write_csv(submission_002, "submissions/submission_002.csv")

