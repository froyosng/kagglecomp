source("R/log_loss.R")

oof <- readRDS("data_processed/oof_ensemble_v10.rds")
oof_mlogit <- oof$oof_mlogit
oof_xgb <- oof$oof_xgb
oof_truth <- oof$oof_truth

w <- 0.80
oof_ensemble <- w * oof_mlogit + (1 - w) * oof_xgb
oof_ensemble <- oof_ensemble / rowSums(oof_ensemble)

train <- read.csv("csv files/train.csv")
cases <- unique(train$Case)
case_to_rows <- split(seq_along(train$Case), train$Case)

# Respondent-clustered estimate of public-LB-sample noise, per the reviewer's
# correct point that a row-level SE (~0.015) understates uncertainty since
# rows are clustered within respondents (19 tasks each) -- exactly the
# clustering this project's own validation design exists to respect.
# Public LB is ~70% of the 263 test respondents =~ 184 respondents; simulate
# that by drawing 184-respondent subsamples (without replacement) from the
# 1135 available training respondents as a proxy population, using existing
# OOF predictions (no new fitting).
n_public_equiv <- round(0.70 * 263)
cat("Simulating public-LB-sized draws of", n_public_equiv, "respondents (proxy: from the 1135 training respondents' OOF predictions)\n\n")

set.seed(99)
B <- 3000
single_draw_ll <- numeric(B)
paired_diff_ll <- numeric(B)

for (b in 1:B) {
  draw <- sample(cases, n_public_equiv, replace = FALSE)
  idx <- unlist(case_to_rows[as.character(draw)])
  single_draw_ll[b] <- log_loss(oof_truth[idx, ], oof_ensemble[idx, ])

  # a second, independent draw (simulating "if a competing team had an
  # equally good model, how different could THEIR public score look")
  draw2 <- sample(cases, n_public_equiv, replace = FALSE)
  idx2 <- unlist(case_to_rows[as.character(draw2)])
  ll2 <- log_loss(oof_truth[idx2, ], oof_ensemble[idx2, ])
  paired_diff_ll[b] <- single_draw_ll[b] - ll2
}

cat("=== Single public-sized draw (", n_public_equiv, "respondents) ===\n")
cat("Mean:", round(mean(single_draw_ll), 6), " SD:", round(sd(single_draw_ll), 6), "\n")
cat("95% range: [", round(quantile(single_draw_ll, 0.025), 4), ",", round(quantile(single_draw_ll, 0.975), 4), "]\n\n")

cat("=== Difference between two independent equal-sized draws of the SAME model ===\n")
cat("(this simulates 'how far apart could two equally-good models' public scores\n")
cat(" look, purely from which respondents happened to land in each draw')\n")
cat("Mean:", round(mean(paired_diff_ll), 6), " SD:", round(sd(paired_diff_ll), 6), "\n")
cat("95% range of |difference|: [0,", round(quantile(abs(paired_diff_ll), 0.95), 4), "]\n\n")

observed_gap <- 0.015
cat("Observed 1.202 vs 1.187 gap:", observed_gap, "\n")
cat("Fraction of simulated same-model draw-pairs with |difference| >=", observed_gap, ":",
    round(mean(abs(paired_diff_ll) >= observed_gap), 4), "\n")
cat("(if this fraction is large, a", observed_gap, "gap between two comparably-good\n")
cat(" models is unremarkable; if small, it suggests a real quality difference)\n")

cat("\n=== For comparison: naive row-level SE estimate (what a non-clustered calc would give) ===\n")
n <- nrow(train)
row_ll <- -log(pmax(oof_ensemble[cbind(seq_len(n), max.col(oof_truth))], 1e-15))
naive_se <- sd(row_ll) / sqrt(round(0.70 * 4997))
cat("Naive (wrongly treats all rows as independent) SE for a public-sized N:", round(naive_se, 6), "\n")
cat("Respondent-clustered SD from simulation above:", round(sd(single_draw_ll), 6), "\n")
cat("Ratio (clustering inflation factor):", round(sd(single_draw_ll) / naive_se, 2), "\n")
