## Second inspection pass: validate fold_of_case indexing assumption across
## ALL 5 outer folds using the cached nested base-fit artifacts, and explore
## the questionnaire-version structure restricted to train respondents (the
## only ones with usable outcomes for the g/h Newton statistics).

ens <- readRDS("data_processed/oof_ensemble_v10.rds")
fold_of_case <- ens$fold_of_case
stopifnot(length(fold_of_case) == 1135L)

for (f in 1:5) {
  holdout_case <- which(fold_of_case == f)
  final_path <- sprintf("data_processed/codex_version_shrinkage/outer%d_final_base_fit.rds", f)
  final_fit <- readRDS(final_path)
  target_case_unique <- unique(final_fit$target_case)
  source_case_unique <- unique(final_fit$source_case)
  cat(sprintf(
    "outer%d: holdout_n=%d final_target_n=%d final_source_n=%d target==holdout:%s source_holdout_overlap:%d\n",
    f, length(holdout_case), length(target_case_unique), length(source_case_unique),
    setequal(holdout_case, target_case_unique),
    length(intersect(source_case_unique, holdout_case))
  ))
  inner_ids <- setdiff(1:5, f)
  all_inner_targets <- c()
  for (ii in inner_ids) {
    ipath <- sprintf("data_processed/codex_version_shrinkage/outer%d_inner%d_base_fit.rds", f, ii)
    iobj <- readRDS(ipath)
    all_inner_targets <- c(all_inner_targets, unique(iobj$target_case))
    overlap_h <- length(intersect(unique(iobj$target_case), holdout_case)) +
      length(intersect(unique(iobj$source_case), holdout_case))
    if (overlap_h != 0) cat(sprintf("  ! outer%d inner%d overlaps holdout! (%d)\n", f, ii, overlap_h))
  }
  cat(sprintf(
    "  inner target union n=%d unique=%d equals source_unique:%s\n",
    length(all_inner_targets), length(unique(all_inner_targets)),
    setequal(all_inner_targets, source_case_unique)
  ))
}

cat("\n=== Version structure (train-only vs all) ===\n")
qf <- readRDS("data_processed/questionnaire_fingerprints.rds")
version_id_all <- match(qf$fingerprints, unique(qf$fingerprints))
map <- data.frame(Case = as.integer(qf$case_ids), version_id = as.integer(version_id_all))
train_map <- map[map$Case <= 1135, ]
cat("train respondents:", nrow(train_map), " distinct versions among train:", length(unique(train_map$version_id)), "\n")
train_counts <- table(train_map$version_id)
cat("train-only version size distribution:\n")
print(summary(as.integer(train_counts)))
cat("versions with exactly 1 train respondent:", sum(train_counts == 1), "\n")
cat("versions with 0 train respondents (test-only):", 299 - length(unique(train_map$version_id)), "\n")

# per outer fold: how many holdout respondents have zero SAME-FOLD... no,
# same as before: zero TRAINING peers (i.e. their version has no OTHER
# train respondent outside this holdout fold).
for (f in 1:5) {
  holdout_case <- which(fold_of_case == f)
  holdout_versions <- train_map$version_id[match(holdout_case, train_map$Case)]
  outer_train_versions <- train_map$version_id[!(train_map$Case %in% holdout_case)]
  peer_counts <- vapply(holdout_versions, function(v) sum(outer_train_versions == v), integer(1))
  cat(sprintf("outer%d: holdout resp with 0 outer-training peers: %d, with exactly 1 peer: %d (of %d)\n",
              f, sum(peer_counts == 0), sum(peer_counts == 1), length(holdout_case)))
}
