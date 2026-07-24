# Data Cleaning & Decisions Log

Running log of data-quality checks and cleaning decisions for the 2026 Analytics Edge
Data Competition (choice among 4 car safety-feature bundles). Update this file as we go
so it can feed directly into the report's methods section.

## 2026-07-24: Initial data audit (train.csv / test.csv)

**Checks performed:**

| Check | Result |
|---|---|
| Missing values in `train.csv` | None (0 NAs across all 113 columns) |
| Missing values in `test.csv` | 19,988 NAs total, all in `Ch1`-`Ch4` (4,997 rows × 4 cols) |
| Exactly one alternative chosen per row (`Ch1`+`Ch2`+`Ch3`+`Ch4`) | Always sums to 1 in train |
| Duplicate `Case`/`Task` combinations | None |
| Respondent panel structure | 1,135 unique respondents (`Case`), each completed exactly 19 tasks (balanced panel) |
| Attribute code ranges (`CC`,`GN`,...,`HU`) | Non-negative integers, varying max per attribute (level codes) |
| Price ranges (alternatives 1-3) | 1 to 12, no zeros |

**Findings / decisions:**

1. **`test.csv`'s `Ch1`-`Ch4` are the prediction targets**, correctly blanked to NA — not a
   data quality issue, just the columns we need to predict probabilities for.
2. **Alternative 4 is a constant "opt-out"/no-purchase option**: all attribute columns and
   `Price4` are 0 for every row. It is chosen in 30.2% of train observations (`Ch4`).
   This needs to be treated deliberately in modeling (e.g., as a genuine 4th alternative
   with a fixed all-zero attribute vector, not dropped).
3. **Redundant covariate encodings**: several respondent-level variables are stored in
   parallel forms — a text/factor version, an `*ind` integer-coded version, and (for some)
   an aggregated numeric `*a` version, e.g. `segment`/`segmentind`, `miles`/`milesind`/`milesa`,
   `night`/`nightind`/`nighta`, `age`/`ageind`/`agea`, `income`/`incomeind`/`incomea`.
   Confirmed `segment` <-> `segmentind` is a clean 1:1 mapping (no inconsistencies found yet;
   other pairs not yet checked). **Open decision:** which version to use per model type
   (see below).
4. No obviously invalid/out-of-range values found in train so far.

**Open decisions to make together:**

- Which encoding (raw factor vs. `*ind` vs. `*a`) to use for each respondent covariate,
  per model type (e.g., factor for `mlogit`, numeric `*a` version might be handy for
  mixed logit or CART).
- How to structure the reshape from wide to long format for `mlogit`, given alt 4 is a
  no-choice option with degenerate (all-zero) attributes.

## 2026-07-24: Verified raw/`*ind`/`*a` covariate encodings

**Checks performed:** confirmed 1:1 mapping between raw text and `*ind` integer code for
all 11 pairs (`segment`, `year`, `miles`, `night`, `ppark`, `gender`, `age`, `educ`,
`region`, `Urb`, `income`); checked whether `*ind` maps 1:1 to `*a` for `miles`, `night`,
`age`, `income`.

**Findings:**

- **raw <-> `*ind`**: all 11 pairs are clean 1:1 recodes (same information, just
  text label vs. integer code). No inconsistencies.
- **`*ind` <-> `*a`**: NOT 1:1, but this is expected, not an error. `*a` is a
  finer-grained numeric estimate *within* the `*ind` bin (e.g. `milesind = 2` is the
  "51 To 100 Miles" bin, and `milesa` holds distinct values like 60, 80, 100 within
  that bin — presumably the respondent's actual reported value or a within-bin
  estimate). So `*a` carries more information than `*ind`, not less.
- **Practical implication**: raw/`*ind` are interchangeable (pick whichever is
  more convenient — factor label or integer code); `*a` is a separate, more granular
  numeric variable and should be treated as such (e.g., candidate continuous
  predictor) rather than as a redundant duplicate of `*ind`.

## 2026-07-24: Long-format reshape, train/validation split, and backtester setup

**What was done:**

1. **Reshaped `train.csv` from wide to long format** using `tidyr::pivot_longer()`
   (source: standard tidyr reshape functionality,
   https://tidyr.tidyverse.org/reference/pivot_longer.html). Each of the 21,565
   choice-task rows becomes 4 rows (one per alternative), turning columns like
   `CC1..CC4`, `Price1..Price4`, `Ch1..Ch4` into single columns `CC`, `Price`,
   `chosen`, plus a new `alt` column (1-4) identifying which alternative the row
   describes. Result: 86,260 rows x 51 columns. Needed for choice models
   (`mlogit`, mixed logit) that expect one row per alternative.
2. **Train/validation split done at the respondent (`Case`) level, not the row
   level** — 908 respondents (17,252 rows) in the training subset, 227
   respondents (4,313 rows, ~20%) held out for validation. Splitting by
   respondent (rather than by individual choice task) avoids leaking a
   respondent's other answers between training and validation, since each
   respondent answered 19 tasks. Seed: 7402.
3. **`log_loss()` helper** (`R/log_loss.R`) implements the competition's exact
   metric (average negative log-likelihood over 4 alternatives, with
   rescaling of predictions to sum to 1, matching the PDF's stated behavior).
   Verified: predicting 0.25 for every alternative on the validation set gives
   1.386294, matching the competition's reported benchmark of 1.38629 almost
   exactly (residual difference from rounding).
4. **`submissions_log.csv`** created as a running tracker: every time a model
   is submitted to Kaggle, log the model name/description, source/citation for
   any non-class method used, the local validation (`cv_logloss`) score, and
   the Kaggle public leaderboard score once known. The gap between the two is
   the main signal for overfitting — if `cv_logloss` looks much better than
   `public_lb_logloss`, the model is likely fitting noise in the training data
   rather than a generalizable pattern.
5. Processed objects saved to `data_processed/train_val_split.rds` for reuse
   across sessions without re-running the reshape/split.

**Citation note:** any model or technique used beyond what was covered in
class should have its source recorded here and in `submissions_log.csv`'s
`source_citation` column, since the report requires this to be documented.

## Template for future entries

```
## YYYY-MM-DD: <short description>
**Checks performed:** ...
**Findings / decisions:** ...
```
