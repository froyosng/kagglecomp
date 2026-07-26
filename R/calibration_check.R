library(tidyverse)

oof <- readRDS("data_processed/oof_ensemble_v10.rds")
oof_mlogit <- oof$oof_mlogit
oof_xgb <- oof$oof_xgb
oof_truth <- oof$oof_truth

w <- 0.80
blend <- w * oof_mlogit + (1 - w) * oof_xgb
blend <- blend / rowSums(blend)

n <- nrow(blend)
true_class <- max.col(oof_truth)
pred_class <- max.col(blend)
p_top <- apply(blend, 1, max)
correct <- pred_class == true_class

bins <- cut(p_top, breaks = seq(0.2, 1.0, by = 0.05), include.lowest = TRUE)
cal <- tibble(bin = bins, p_top = p_top, correct = correct) %>%
  group_by(bin) %>%
  summarise(n = n(), mean_predicted_confidence = mean(p_top),
            actual_accuracy = mean(correct), .groups = "drop")
cat("Calibration check: predicted confidence (top pick) vs actual accuracy in that bin\n")
cat("If well-calibrated, actual_accuracy should track mean_predicted_confidence closely.\n\n")
print(as.data.frame(cal), row.names = FALSE)

cat("\nOverall: mean top-pick confidence =", round(mean(p_top), 4),
    " vs overall accuracy =", round(mean(correct), 4), "\n")

# also check calibration on ALL 4 class-probabilities pooled (not just the top pick)
# -- reliability across the whole probability range, using every (task, alt) cell
all_probs <- as.vector(blend)
all_actual <- as.vector(oof_truth)
bins2 <- cut(all_probs, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)
cal2 <- tibble(bin = bins2, p = all_probs, actual = all_actual) %>%
  group_by(bin) %>%
  summarise(n = n(), mean_predicted = mean(p), mean_actual = mean(actual), .groups = "drop")
cat("\nFull reliability diagram (every alt's predicted prob vs whether it was actually chosen):\n")
print(as.data.frame(cal2), row.names = FALSE)
