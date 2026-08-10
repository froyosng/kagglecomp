library(tidyverse)
source("R/log_loss.R")

oof <- readRDS("data_processed/oof_ensemble_v10.rds")
oof_mlogit <- oof$oof_mlogit
oof_xgb <- oof$oof_xgb
oof_truth <- oof$oof_truth

w <- 0.80
blend <- w * oof_mlogit + (1 - w) * oof_xgb
blend <- blend / rowSums(blend)

train <- read.csv("csv files/train.csv")
n <- nrow(train)

true_class <- max.col(oof_truth)
pred_class <- max.col(blend)
correct <- pred_class == true_class

p_true <- blend[cbind(1:n, true_class)]
p_top  <- apply(blend, 1, max)
gap    <- p_top - p_true  # 0 when correct; how much more confident the model was in its wrong pick

row_ll <- -log(pmax(p_true, 1e-15))  # each task's own log-loss contribution

cat("Overall accuracy (argmax):", round(mean(correct), 4), "\n")
cat("Overall pooled log loss:", round(mean(row_ll), 6), "(matches ensemble CV)\n\n")

cat("=== Among the", sum(!correct), "misses (", round(100*mean(!correct),1), "% of tasks ) ===\n")
miss_gap <- gap[!correct]
cat("Gap between top pick and true alt's probability, among misses:\n")
print(summary(miss_gap))
cat("\nBuckets:\n")
buckets <- cut(miss_gap, breaks = c(-Inf, 0.05, 0.10, 0.20, 0.30, Inf),
               labels = c("<0.05 (near coin-flip)", "0.05-0.10", "0.10-0.20", "0.20-0.30", ">0.30 (confident miss)"))
print(table(buckets))
print(round(100 * prop.table(table(buckets)), 1))

cat("\n=== Log-loss concentration: where does the total log loss actually come from? ===\n")
ord <- order(row_ll, decreasing = TRUE)
cum_ll <- cumsum(row_ll[ord]) / sum(row_ll)
for (pct in c(0.01, 0.05, 0.10, 0.20, 0.30, 0.50)) {
  k <- round(pct * n)
  cat(sprintf("Worst %2.0f%% of tasks (by their own log loss) account for %5.1f%% of TOTAL log loss\n",
              pct*100, 100*cum_ll[k]))
}

cat("\n=== Are 'confident misses' (gap > 0.30) concentrated anywhere identifiable? ===\n")
confident_miss <- !correct & gap > 0.30
train$confident_miss <- confident_miss
train$is_optout_true <- true_class == 4

cat("\nConfident-miss rate by true class (1=alt1,2=alt2,3=alt3,4=opt-out):\n")
print(round(tapply(confident_miss, true_class, mean), 4))

cat("\nConfident-miss rate by segment:\n")
print(round(tapply(confident_miss, train$segmentind, mean), 4))

cat("\nConfident-miss rate by Task position (binned):\n")
task_bin <- cut(train$Task, breaks = c(0,5,10,15,19))
print(round(tapply(confident_miss, task_bin, mean), 4))

cat("\nConfident-miss rate by region:\n")
print(round(tapply(confident_miss, train$regionind, mean), 4))
