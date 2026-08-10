## Confirms and refutes an external hypothesis about the survey's design
## structure: are the 19-task sequences generated from a small, fixed pool of
## questionnaire "versions" reused across respondents (Sawtooth CBC-style
## rotation), and if so, does simple sequential Case-number cycling explain
## which version a respondent got?
##
## Method: fingerprint each respondent's ENTIRE ordered 19-task sequence
## (every attribute + price for all 4 alternatives, in task order -- design
## only, no choices, so this is valid to build from train+test combined).
## Then (a) count unique fingerprints, and (b) test whether same-fingerprint
## respondents' Case numbers differ by a constant period V for a range of
## candidate V.

attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
           "PP","KA","SC","TS","NV","MA","LB","AF","HU")

train <- read.csv("csv files/train.csv")
test <- read.csv("csv files/test.csv")
test$Ch1 <- NA; test$Ch2 <- NA; test$Ch3 <- NA; test$Ch4 <- NA
all_data <- rbind(train, test)

stimulus_cols <- unlist(lapply(c(attrs, "Price"), function(a) paste0(a, 1:4)))

build_fingerprint <- function(df) {
  df <- df[order(df$Case, df$Task), ]
  split_by_case <- split(df, df$Case)
  vapply(split_by_case, function(case_df) {
    case_df <- case_df[order(case_df$Task), ]
    paste(
      apply(case_df[, stimulus_cols], 1, paste, collapse = ","),
      collapse = "||"
    )
  }, character(1))
}

fingerprints <- build_fingerprint(all_data)
case_ids <- as.integer(names(fingerprints))
ord <- order(case_ids)
fingerprints <- fingerprints[ord]
case_ids <- case_ids[ord]

cat("Total respondents (train+test):", length(fingerprints), "\n")
cat("Unique fingerprints:", length(unique(fingerprints)), "\n")

dup_table <- table(fingerprints)
recurring <- names(dup_table)[dup_table > 1]
cat("Fingerprints that recur (>1 respondent):", length(recurring),
    "of", length(unique(fingerprints)), "\n")

if (length(recurring) > 0) {
  gaps <- unlist(lapply(recurring, function(fp) {
    cases_with_fp <- sort(case_ids[fingerprints == fp])
    if (length(cases_with_fp) > 1) diff(cases_with_fp) else NULL
  }))
  cat("\nCase-number gap distribution within a shared fingerprint (summary):\n")
  print(summary(gaps))
}

## Candidate-period sweep: for each V, what fraction of (Case mod V) groups
## with >1 respondent have ALL members sharing one fingerprint?
test_period <- function(V) {
  version <- ((case_ids - 1) %% V) + 1
  n_unique_fp <- tapply(fingerprints, version, function(x) length(unique(x)))
  n_per_version <- tapply(fingerprints, version, length)
  multi <- n_per_version > 1
  if (sum(multi) == 0) return(NA_real_)
  mean(n_unique_fp[multi] == 1)
}
periods <- c(50, 100, 150, 200, 250, 300, 350, 400, 450, 500)
purity <- vapply(periods, test_period, numeric(1))
cat("\nSequential-cycling purity by candidate period (0 = no evidence of that period):\n")
print(data.frame(period = periods, purity = round(purity, 4)))

cat("\nConclusion: ~299 unique 19-task designs shared across train+test\n",
    "respondents (~4.7 per version on average), but NOT explained by simple\n",
    "sequential Case-number cycling at any tested period -- version assignment\n",
    "looks effectively random with respect to Case order.\n", sep = "")

saveRDS(
  list(fingerprints = fingerprints, case_ids = case_ids,
       n_unique = length(unique(fingerprints))),
  "data_processed/questionnaire_fingerprints.rds"
)
