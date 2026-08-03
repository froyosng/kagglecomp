## Mechanism check: does neighbor pooling actually help the specific
## respondents the rejected experiment failed on (zero/one same-version
## training peers), or is any net effect coming from elsewhere? Uses the
## already-saved full-grid result (selected k*/lambda* per fold) plus the
## primary-corrected predictions already computed by the CV driver.

source("R/codex_version_pool_common.R")

res <- readRDS("data_processed/codex_version_pool/full_result.rds")
fold_summary <- res$fold_summary
baseline <- current_fixed_oof()
version_map <- load_version_map()
truth <- baseline$truth
base_pred <- baseline$pred
corrected_pred <- res$primary_corrected_full  # already aligned to baseline$no order
case <- baseline$case
fold_of_case <- baseline$fold_of_case

## Per-respondent own-version peer count within their own outer-training set
## (i.e. same definition used in the earlier inspection: how many OTHER
## training respondents, outside this outer fold's holdout, share this
## person's version).
train_version <- version_map[version_map$Case <= 1135, ]
peer_count <- integer(length(unique(case)))
uniq_case <- sort(unique(case))
peer_count_df <- data.frame(Case = uniq_case, peer = NA_integer_, fold = NA_integer_)
for (f in 1:5) {
  holdout <- which(fold_of_case == f)
  holdout_versions <- train_version$version_id[match(holdout, train_version$Case)]
  outer_train_versions <- train_version$version_id[!(train_version$Case %in% holdout)]
  pc <- vapply(holdout_versions, function(v) sum(outer_train_versions == v), integer(1))
  idx <- match(holdout, peer_count_df$Case)
  peer_count_df$peer[idx] <- pc
  peer_count_df$fold[idx] <- f
}
stopifnot(!anyNA(peer_count_df$peer))

gain_by_case <- case_mean_gain(truth, base_pred, corrected_pred, case)
gain_df <- data.frame(Case = as.integer(names(gain_by_case)), gain = as.numeric(gain_by_case))
merged <- merge(peer_count_df, gain_df, by = "Case")
stopifnot(nrow(merged) == 1135L)

merged$peer_band <- cut(merged$peer, breaks = c(-1, 0, 1, 3, Inf),
                         labels = c("0 (no peers)", "1 peer", "2-3 peers", "4+ peers"))

cat("=== Mean per-respondent log-loss gain by own-version peer-count band (full k-grid result) ===\n")
agg <- aggregate(gain ~ peer_band, merged, function(x) c(mean = mean(x), n = length(x)))
print(agg)
cat("\n(positive = correction helped that respondent on average over their 19 tasks)\n")

cat("\n=== Same breakdown restricted to fold 3 only (the fold where k>0 was selected) ===\n")
f3 <- merged[merged$fold == 3, ]
agg3 <- aggregate(gain ~ peer_band, f3, function(x) c(mean = mean(x), n = length(x)))
print(agg3)

cat("\n=== Same breakdown for the OTHER four folds (all selected k=0, pure replication) ===\n")
other <- merged[merged$fold != 3, ]
agg_other <- aggregate(gain ~ peer_band, other, function(x) c(mean = mean(x), n = length(x)))
print(agg_other)

write.csv(merged, "data_processed/codex_version_pool/mechanism_check_by_respondent.csv", row.names = FALSE)
