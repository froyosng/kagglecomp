library(tidyverse)

attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
           "PP","KA","SC","TS","NV","MA","LB","AF","HU","Price")

train <- read.csv("csv files/train.csv")
test <- read.csv("csv files/test.csv")

design_cols <- as.vector(outer(attrs, 1:4, paste0))  # e.g. CC1..CC4, ..., Price1..Price4

fingerprint <- function(df) {
  do.call(paste, c(df[, design_cols], sep = "_"))
}
# permutation-invariant fingerprint: same 4 bundles regardless of which
# alt-slot (1/2/3) they landed in, in case designs recur with alts reshuffled
fingerprint_permuted <- function(df) {
  bundles <- sapply(1:4, function(a) do.call(paste, c(df[, paste0(attrs, a)], sep = "_")))
  apply(bundles, 1, function(row) paste(sort(row), collapse = "|"))
}

train$fp <- fingerprint(train)
test$fp <- fingerprint(test)
train$fp_perm <- fingerprint_permuted(train)
test$fp_perm <- fingerprint_permuted(test)

cat("=== Exact design overlap (alt-slots matter) ===\n")
cat("Duplicate fingerprints WITHIN train:", sum(duplicated(train$fp)), "of", nrow(train), "\n")
cat("Duplicate fingerprints WITHIN test:", sum(duplicated(test$fp)), "of", nrow(test), "\n")
cat("Train fingerprints that also appear in test:", sum(train$fp %in% test$fp), "\n")
cat("Test fingerprints that also appear in train:", sum(test$fp %in% train$fp), "\n\n")

cat("=== Permutation-invariant overlap (same 4 bundles, any alt order) ===\n")
cat("Duplicate fp_perm WITHIN train:", sum(duplicated(train$fp_perm)), "of", nrow(train), "\n")
cat("Duplicate fp_perm WITHIN test:", sum(duplicated(test$fp_perm)), "of", nrow(test), "\n")
cat("Train fp_perm that also appear in test:", sum(train$fp_perm %in% test$fp_perm), "\n\n")

cat("=== Do the same designs repeat by Task position (block design)? ===\n")
cat("Duplicate (Task, fp) within train (same task position, same exact design, different respondent):",
    sum(duplicated(train[, c("Task","fp")])), "\n")
tab <- train %>% count(Task, fp) %>% count(Task, name = "n_distinct_designs")
cat("Distinct designs per Task position (median across the 19 positions):",
    median(tab$n_distinct_designs), " (if this << number of respondents, tasks are block-reused)\n")
print(tab)
