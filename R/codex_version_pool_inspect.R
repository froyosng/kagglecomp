## One-off inspection script: understand the shape of the artifacts this
## experiment will reuse (canonical fold assignment, questionnaire fingerprint
## artifact, cached nested inner/outer base-model fits from the rejected
## codex-version-shrinkage experiment). Not part of the final pipeline.

ens <- readRDS("data_processed/oof_ensemble_v10.rds")
cat("=== oof_ensemble_v10.rds ===\n")
print(names(ens))
cat("fold_of_case length:", length(ens$fold_of_case), "\n")
print(table(ens$fold_of_case))
cat("dim oof_mlogit:", paste(dim(ens$oof_mlogit), collapse="x"), "\n")
cat("dim oof_xgb:", paste(dim(ens$oof_xgb), collapse="x"), "\n")
cat("dim oof_truth:", paste(dim(ens$oof_truth), collapse="x"), "\n")
cat("class fold_of_case:", class(ens$fold_of_case), "\n")
print(head(ens$fold_of_case))

cat("\n=== questionnaire_fingerprints.rds ===\n")
qf <- readRDS("data_processed/questionnaire_fingerprints.rds")
print(names(qf))
cat("n_unique:", qf$n_unique, "\n")
cat("length case_ids:", length(qf$case_ids), "\n")
cat("length fingerprints:", length(qf$fingerprints), "\n")
cat("class fingerprints:", class(qf$fingerprints), "\n")
cat("sample fingerprint (truncated to 300 chars):\n")
cat(substr(qf$fingerprints[1], 1, 300), "\n")
cat("range case_ids:", range(qf$case_ids), "\n")
version_id <- match(qf$fingerprints, unique(qf$fingerprints))
tab <- table(version_id)
cat("versions with size distribution (summary):\n")
print(summary(as.integer(tab)))
cat("num versions with size==1:", sum(tab==1), "\n")

cat("\n=== mlp_oof.rds ===\n")
mlp <- readRDS("data_processed/codex_behavioral_round/mlp_oof.rds")
print(names(mlp))
print(names(mlp$oof))
cat("dim h08_d0.100:", paste(dim(mlp$oof[["h08_d0.100"]]), collapse="x"), "\n")

cat("\n=== outer1_final_base_fit.rds ===\n")
f1 <- readRDS("data_processed/codex_version_shrinkage/outer1_final_base_fit.rds")
print(names(f1))
cat("length source_case:", length(f1$source_case), " length target_case:", length(f1$target_case), "\n")
cat("dim source_pred:", paste(dim(f1$source_pred), collapse="x"), " dim target_pred:", paste(dim(f1$target_pred), collapse="x"), "\n")
cat("source_loss:", f1$source_loss, " target_loss:", f1$target_loss, "\n")
cat("overlap source/target case:", length(intersect(f1$source_case, f1$target_case)), "\n")

cat("\n=== outer1_inner2_base_fit.rds ===\n")
i2 <- readRDS("data_processed/codex_version_shrinkage/outer1_inner2_base_fit.rds")
print(names(i2))
cat("length source_case:", length(i2$source_case), " length target_case:", length(i2$target_case), "\n")
cat("overlap source/target case:", length(intersect(i2$source_case, i2$target_case)), "\n")

## Cross check: outer fold 1 holdout cases (fold_of_case == 1) must never
## appear as source OR target in ANY of the 4 inner fits for outer fold 1.
outer1_holdout_cases <- names(ens$fold_of_case)[ens$fold_of_case == 1]
# fold_of_case might be a plain integer vector keyed by case order instead of named; handle both
if (is.null(names(ens$fold_of_case))) {
  outer1_holdout_cases <- which(ens$fold_of_case == 1)
}
cat("\nouter1 holdout case count:", length(outer1_holdout_cases), "\n")

inner_ids <- c(2,3,4,5)
all_inner_targets <- c()
for (ii in inner_ids) {
  path <- sprintf("data_processed/codex_version_shrinkage/outer1_inner%d_base_fit.rds", ii)
  obj <- readRDS(path)
  overlap_holdout_source <- length(intersect(obj$source_case, outer1_holdout_cases))
  overlap_holdout_target <- length(intersect(obj$target_case, outer1_holdout_cases))
  cat(sprintf("inner%d: n_source=%d n_target=%d overlap_holdout_source=%d overlap_holdout_target=%d\n",
              ii, length(obj$source_case), length(obj$target_case),
              overlap_holdout_source, overlap_holdout_target))
  all_inner_targets <- c(all_inner_targets, obj$target_case)
}
cat("total inner target cases (should be 908, all outer-training resp, no dup):", length(all_inner_targets), "\n")
cat("unique inner target cases:", length(unique(all_inner_targets)), "\n")
cat("outer1 final source_case count (should==908):", length(f1$source_case), "\n")
cat("set equal to union of inner targets?:", setequal(all_inner_targets, f1$source_case), "\n")
