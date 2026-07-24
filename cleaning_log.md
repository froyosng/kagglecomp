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

## 2026-07-24: Modeling notes for the report -- ASC identification issue, factor vs. continuous, and mixed logit

**1. Alternative-specific intercept (ASC) identification conflict with factor-coded attributes**

When attribute levels are dummy-coded (factors) with 0 as the reference level, alternative
4 (the opt-out) is *always* at the reference level for every attribute simultaneously --
and no other alternative (1-3) ever has this exact all-reference-level profile (verified:
0 occurrences among alts 1-3 in `train`). This means an ASC for alternative 4 cannot be
separately identified from the "all-baseline" utility once attributes are dummy-coded --
they are perfectly confounded, producing a computationally singular Hessian in `mlogit`.
This is a structural identification issue caused by the opt-out's design (constant,
all-zero attributes), not a data error. Two ways to work around it, both tried:

- Drop ASCs entirely (`mod2b`): alt 4's utility is fixed at 0 by construction, with no
  free parameter needed; all variation is captured through the attribute coefficients
  of alts 1-3 relative to that fixed 0.
- Keep only alt-specific dummies for alternatives 2 and 3 (not 4) (`mod3`): recovers
  some "position/order" effect without re-introducing the identification conflict.

Continuous-coded attributes (`mod1`) don't have this problem because 0 is just a
data value multiplied by a slope (contributing exactly 0 to utility), not a
dropped reference category requiring separate identification.

**2. Attribute levels: continuous vs. categorical (factor) coding**

`mod1` (continuous, generic linear slope per attribute) assumes utility changes
linearly with each attribute's level code (0-6 depending on attribute). This is a
strong assumption with no particular justification -- there's no reason a level of "4"
should be exactly twice as good/bad as a level of "2". Refitting with attributes as
factors (`mod2b`, `mod3`) relaxes this and improved validation log loss from 1.236
(continuous) to ~1.219-1.220 (factors), suggesting the attribute-level effects are
somewhat non-linear. Trade-off: factor coding uses ~3x more parameters (61 vs ~21),
so there's more estimation noise per coefficient, but the net effect on held-out
validation was still an improvement.

**3. Mixed logit (random parameters) -- a cautionary overfitting result**

Fit `mod4`: a panel mixed logit (`mlogit`, `panel = TRUE`, grouped by respondent
`Case`) with independent normal random parameters on all 20 continuous attributes
(uncorrelated, R = 100 Halton draws, `method = "bhhh"`; source: `mlogit` package /
Croissant's mixed logit vignette, `vignette("c5.mxl", "mlogit")`, and Train, K. (2009)
*Discrete Choice Methods with Simulation*, referenced therein).

- Training log-likelihood improved substantially: -16,449 (mod4) vs. -20,432 (mod1
  continuous, fixed parameters) -- a large apparent improvement in fit.
- But validation log loss was *worse*: 1.247 (mod4) vs. 1.236 (mod1) and ~1.219
  (factor models mod2b/mod3).

This is a textbook illustration of overfitting: allowing every respondent's attribute
sensitivities to vary freely (20 random parameters estimated from only 19 choice
observations per respondent) lets the model fit person-specific idiosyncrasies in the
training data that don't generalize. This directly supports the earlier concern about
the public leaderboard not necessarily reflecting a model's true quality -- a model can
look much better by an in-sample/likelihood measure while doing worse on held-out data.
**Not submitted to Kaggle** given the validation result is already worse than existing
factor models; logged as a negative result in `submissions_log.csv`.

**Practical takeaway for model selection:** favor models with better *validation* log
loss over models with better *training* log-likelihood -- the two diverged clearly in
this comparison, which is exactly the kind of check the validation split was built for.

## 2026-07-25: Lighter mixed logit (Price only) and a CART comparison

**Mixed logit with a single random parameter (Price only).** Refit the panel mixed
logit with only Price as a random (normal) parameter and all other 19 attributes
fixed/generic. Training log-likelihood: -16,641 (between mod1's -20,432 and the
fully-random mod4's -16,449). Validation log loss: 1.235, essentially matching mod1
(1.236) and clearly better than the fully-random version (1.247). sd.Price was large
and highly significant, consistent with genuine respondent-level heterogeneity in price
sensitivity. Conclusion: restricting the random-parameters structure to a single,
substantively-motivated coefficient avoids the overfitting seen when all 20 attributes
were allowed to vary.

**CART (rpart) as a non-logit comparison.** Fit a default-complexity classification
tree (rpart, method="class") on the wide-format data (all 80 alternative-specific
attribute/price columns plus respondent covariates as factors), predicting which of the
4 alternatives was chosen directly (source: class material, rpart package). Validation
log loss: 1.293, better than the uniform benchmark (1.386) but worse than every
mlogit-family model tried (1.219-1.247). This fits expectations: the choice task has an
underlying random-utility structure that logit models are built to exploit, while a
single decision tree partitions the covariate space more crudely.

Current best model by validation log loss: the factor-coded conditional logit with
alt2/alt3 dummies (mod2b/v2b, 1.219), closely followed by the factor-coded model
without alternative-specific terms (mod2a, 1.220).

## Template for future entries

```
## YYYY-MM-DD: <short description>
**Checks performed:** ...
**Findings / decisions:** ...
```
