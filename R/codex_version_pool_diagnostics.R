## Extra diagnostics on the full run: how peaked/robust was the fold-3 inner
## selection (the one fold where allowing k>0 changed the outcome materially),
## and how does the full k-grid result compare to restricting k=0 only.

grid_all <- read.csv("data_processed/codex_version_pool/full_grid_all.csv")
fold_summary <- read.csv("data_processed/codex_version_pool/full_fold_summary.csv")
k0_fold_summary <- read.csv("data_processed/codex_version_pool/smoketest_k0_fold_summary.csv")
k0_overall <- read.csv("data_processed/codex_version_pool/smoketest_k0_overall.csv")
full_overall <- read.csv("data_processed/codex_version_pool/full_overall.csv")

cat("=== Fold 3 inner-loss grid (baseline_inner_loss shown once) ===\n")
f3 <- grid_all[grid_all$outer_fold == 3, ]
f3 <- f3[order(f3$inner_loss), ]
cat("baseline inner loss:", f3$baseline_inner_loss[1], "\n")
print(f3[, c("k","lambda","inner_loss")])
cat("\ninner gain (baseline - inner_loss) by k (best lambda per k):\n")
best_by_k <- do.call(rbind, lapply(split(f3, f3$k), function(d) d[which.min(d$inner_loss), ]))
best_by_k$inner_gain <- best_by_k$baseline_inner_loss - best_by_k$inner_loss
print(best_by_k[order(best_by_k$k), c("k","lambda","inner_loss","inner_gain")])

cat("\n=== Fold-by-fold: k=0-only grid vs full k-grid (primary gain vs exact OOF) ===\n")
cmp <- merge(
  k0_fold_summary[, c("outer_fold","k_star","lambda_star","primary_gain")],
  fold_summary[, c("outer_fold","k_star","lambda_star","primary_gain")],
  by = "outer_fold", suffixes = c("_k0only", "_fullgrid")
)
print(cmp[order(cmp$outer_fold), ])

cat("\n=== Overall: k=0-only vs full grid ===\n")
cat(sprintf("k=0-only:  gain=%.7f  95%% CI=[%.7f, %.7f]\n",
            k0_overall$primary_gain, k0_overall$primary_boot_lower95, k0_overall$primary_boot_upper95))
cat(sprintf("full grid: gain=%.7f  95%% CI=[%.7f, %.7f]\n",
            full_overall$primary_gain, full_overall$primary_boot_lower95, full_overall$primary_boot_upper95))

## How many of the 40 (k,lambda) combos in fold 3 are within a tiny margin of
## the selected minimum? (peaked vs broad optimum -- a broad optimum spanning
## many k values, all similarly good, is more reassuring than an isolated spike.)
best_loss <- min(f3$inner_loss)
near_best <- f3[f3$inner_loss <= best_loss + 0.0002, ]
cat("\ncombos within 0.0002 inner-loss of the fold-3 minimum:\n")
print(near_best[order(near_best$inner_loss), c("k","lambda","inner_loss")])
