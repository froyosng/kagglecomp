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

## 2026-07-25: Observed heterogeneity via covariate interactions (new best model)

Acted on the LASSO finding that respondent covariates carry signal the conditional
logit models weren't using. In a conditional logit, respondent-level covariates are
constant across the 4 alternatives, so they cannot enter as main effects (they'd
cancel in the choice probabilities) -- they must be **interacted** with something
that varies across alternatives.

**mod6 -- Price x covariate interactions.** Starting from mod2b (factor-coded
attributes + Price + alt2/alt3 dummies, no ASCs), added Price interacted with four
standardized respondent covariates: income, age, miles driven/yr, and night-driving
percentage. Because Price varies across alternatives while the covariate is constant
within respondent, the product varies across alternatives and gets a generic
coefficient -- no ASC identification conflict. Interpretation: price sensitivity
itself differs by respondent. Validation log loss **1.205 vs mod2b's 1.219** -- the
largest single improvement since introducing factor coding.

**mod7 -- add inside-good x covariate interactions (current best).** Added an
"inside" dummy (1 for the three real bundles, 0 for the opt-out alt 4) interacted
with seven standardized covariates (income, age, miles, night, gender, urbanicity,
education). These let the propensity to buy *any* bundle vs. decline vary by
respondent -- identifiable because alt 4 is the fixed reference. Validation log loss
**1.2024**. Largest opt-out-heterogeneity effects: inside x gender (0.145) and
inside x urbanicity (0.090). Gain over mod6 is small (~0.0025) vs mod6's large gain
over mod2b, so returns are clearly diminishing.

**Scaling correctness note.** All standardization constants (mean/sd) are computed
on the *training* subset only and applied identically to validation and test, so the
coefficients see the same transformation everywhere. An earlier pass that re-scaled
each split by its own mean/sd gave nearly identical numbers (respondent covariates
are similarly distributed across the random respondent split), but the training-based
scaler is the correct choice and is what the saved model and submission use.

Source for the observed-heterogeneity interaction approach: Train, K. (2009)
*Discrete Choice Methods with Simulation*, ch. 2-3 (systematic taste variation via
interactions of alternative attributes with decision-maker characteristics).

Submission written: `submission_mlogit_v7_covariate_interactions.csv`. Current best
by validation log loss: **mod7 (1.2024)**, then mod6 (1.2049), then mod2b (1.2186).

## Template for future entries

```
## YYYY-MM-DD: <short description>
**Checks performed:** ...
**Findings / decisions:** ...
```

## 2026-07-25: LASSO multinomial and regsubsets screening

**LASSO-regularized multinomial logit (glmnet).** Fit a cross-validated LASSO
multinomial logistic regression (family="multinomial", type.multinomial="grouped")
on wide-format data, using all 80 alt-specific attribute/price columns as
independent per-class predictors (not tied to a single shared slope like
mlogit's conditional logit) plus respondent covariates. Used 5-fold CV grouped
by respondent (Case) to avoid leaking a respondent's 19 repeated tasks across
folds. Validation log loss: 1.226 -- better than the continuous conditional
logit (1.236) but worse than the factor-coded conditional logit (1.219-1.220).
At lambda.min, 58 of 151 candidate predictors were retained per class,
including several respondent covariates (segment, miles, night, ppark, gender,
age, educ, region, Urb, income) not used in any mlogit model so far -- worth
considering as candidate additions/interactions in a refined choice model.
Source: glmnet package (Friedman, Hastie, Tibshirani).

**Regsubsets screening (linear-probability heuristic).** Used `leaps::regsubsets`
on a linear-probability approximation (chosen ~ 20 attributes, long format) purely
as a variable-importance screening tool -- not a final model, since regsubsets
requires a continuous/lm response and can't natively fit the categorical choice
outcome. Result: adjusted R^2 kept improving through all 20 variables (no
attribute could be safely dropped), but the entry order showed Price entering
first and alone explaining most of the variation (R^2 = 0.066 of a total 0.077
across all 20 attributes), followed by CC, KA, BU, NS. This confirms the full
attribute set used in the conditional logit models is justified rather than
overfit padding, and that Price is by far the dominant driver of choice.

**Current best model remains** the factor-coded conditional logit with alt2/alt3
dummies (mod2b/v2b, 1.219 validation).

## 2026-07-25: mod7 public LB and segment interactions (mod8, new best)

**mod7 public LB result.** Submitted to Kaggle: public 1.230 vs validation 1.2024,
gap 0.028 -- comparable to (slightly under) mod1's 0.034 gap, so the covariate
interactions generalize rather than overfit, and mod7 clearly beats mod1's public
1.270. Validation stays optimistic by a small, consistent margin and remains a
trustworthy tool for ranking models.

**Segment interactions (mod8, new best).** Added Price x segment and inside-good x
segment interactions to mod7. `segment` is the respondent's car-market segment (6
levels: Full-size Pickup, Midsize Car, Midsize Luxury Utility, Midsize Utility,
Prestige Luxury Sedan, Small Car), a respondent-level covariate. Built 5 Price x
segment + 5 inside x segment dummy columns explicitly (segment 1 = reference).
Rationale: people shopping different vehicle segments plausibly differ in both price
sensitivity and baseline willingness to buy any bundle vs. opt out. Validation log
loss 1.1896 vs mod7's 1.2024 -- a ~0.013 gain, larger than mod7's gain over mod6, so
segment is a genuinely strong new signal. The Prestige Luxury Sedan segment is the
least price-sensitive (P_seg5 = 0.166). If mod7's ~0.028 gap holds, mod8's expected
public is ~1.217. Submission: submission_mlogit_v8_segment_interactions.csv.

Current best by validation: mod8 (1.1896) > mod7 (1.2024) > mod6 (1.2049) >
mod2b (1.2186).

## 2026-07-25: Three extensions to mod8 tested -- all confirm mod8 is best

Explored three ways to push past mod8 (segment interactions, val 1.1896). None
improved meaningfully; mod8 remains the best model and we have hit diminishing
returns for the conditional-logit family here.

1. **Attribute x segment (mod9).** Interacted the four dominant attributes
   (CC, KA, BU, NS) with segment as continuous level-effects (20 extra params).
   Validation 1.1907 -- slightly WORSE than mod8. Feature valuation does not vary by
   segment beyond what Price x segment and inside x segment already capture.

2. **Random-Price mixed logit on mod8 (mod10).** Panel mixed logit, Price as a
   single random normal parameter (R=50). Validation 1.18925 vs mod8 1.18956 -- a
   negligible ~0.0003 gain (sd.Price=0.51). Observed heterogeneity already soaks up
   most price-taste variation, leaving little unobserved heterogeneity to model.
   (Implementation note: predict() on a random-parameter mlogit returns an unnamed
   matrix; align its rows using the rownames from a fixed-model predict() on the same
   newdata, not unique(chid) order, which silently mis-aligns and inflates log loss.)

3. **Pruned mod8 (mod11).** Dropped the 6 clearly non-significant interactions
   (P_income, P_night, In_income, In_age, In_miles, In_night; all p>0.17).
   Validation 1.18987 with 78 params vs mod8's 1.18956 with 84 -- essentially tied,
   6 fewer parameters. A clean, near-equivalent model worth citing for parsimony;
   mod8 keeps the marginally-better validation so stays the submission of record.

**Conclusion.** mod8 (mod2b + Price/inside x covariate + Price/inside x segment) is
the best model by validation (1.1896). The strongest heterogeneity signals are
segment-specific price sensitivity (Prestige Luxury least sensitive) and
segment/gender/urbanicity-specific opt-out propensity.

## 2026-07-25: xgboost comparison and 5-fold CV on mod8

**Gradient-boosted trees (different model family).** Fit xgboost (multi:softprob,
4 classes) on wide-format data: 80 alt-specific attribute/price columns + 15
respondent-covariate columns. nrounds tuned via 5-fold respondent-grouped xgb.cv
(best 73; eta 0.1, max_depth 4, subsample/colsample 0.8), then evaluated once on the
same held-out validation respondents. Validation log loss 1.2042 -- worse than mod8
(1.1896) and about level with mod7. A structure-free tree ensemble does not beat the
conditional logit because the choice task's random-utility structure (within-task
comparison of alternatives, generic attribute slopes, a designed opt-out) is encoded
directly by the logit but must be learned from raw features by xgboost. Source:
xgboost (Chen & Guestrin 2016).

**Respondent-level 5-fold CV on mod8.** The 1.1896 figure for mod8 came from a single
80/20 respondent split. Refit mod8 across 5 respondent-grouped folds (each fold: train
908 respondents, validate 227), recomputing the standardization scaler within each
fold's training data, and pooled the held-out negative log-likelihood over all 21,565
choice tasks. Pooled CV log loss 1.1665 -- tighter and slightly more optimistic than
the single split (1.1896), indicating the single validation fold was a bit harder than
average. Best estimate of mod8's generalization log loss is ~1.167.

## 2026-07-25: Regularized interaction selection (glmnet stratified-Cox)

Implemented the "next step" the report proposes: a regularized conditional logit that
selects interactions automatically while keeping the choice-model structure. Used the
equivalence between conditional logit and a stratified Cox partial likelihood (each
choice task = one stratum, chosen alternative = the event), which lets glmnet
(family="cox", alpha=1) fit an L1-penalized choice model. Design pool: 63 core
unpenalized terms (19 attribute level dummies + Price + alt2/alt3 dummies) plus 195
penalized candidate interactions (Price/inside x 7 covariates, Price/inside x segment,
and every attribute x segment and attribute x {income,age,miles,night}). lambda chosen
by 5-fold respondent-grouped cv.glmnet.

Results (validation log loss):
- lambda.1se: 1.21858, drops ALL interactions -> reproduces mod2b (1.2186). This is a
  clean correctness check confirming the Cox = conditional-logit equivalence.
- lambda.min: 1.19542, keeps 13 interactions.
- relaxed refit (unpenalized MLE on core + 13 selected): 1.19369.
- hand-built mod8: 1.18956 (still best).

The LASSO independently rediscovered mod8's core heterogeneity structure -- Price x
segment (seg3, seg5), Price x age, inside x gender, inside x urbanicity -- from a
195-term pool, plus a couple of attribute interactions mod8 lacks (NS_seg3, KA_nighta).
This confirms mod8's hand-chosen interactions are genuine signal rather than overfit
padding, but automated selection does not beat mod8: the cross-validated L1 penalty is
more parsimonious (13 vs ~21 interactions) at a small cost (~0.004), and even removing
shrinkage bias via the relaxed refit does not close the gap. mod8 remains the best
model. Source: glmnet (Friedman/Hastie/Tibshirani); Cox partial-likelihood equivalence.

## 2026-07-25: Bottom-up fresh review -- task-fatigue, region/ppark, and an ensemble

Took a deliberate fresh-eyes pass over the whole project to find signal we had not
adapted. Three findings, all acted on.

**1. Test respondents are entirely new people.** train = Case 1–1135, test = Case
1136–1398 (263 respondents × 19 tasks = 4,997 rows), zero overlap. This is a key
structural fact: we can never personalize to a specific test respondent, so
respondent-specific random effects (mixed logit) cannot transfer -- only OBSERVED
heterogeneity (covariate/segment/region interactions) generalizes. It explains cleanly
why every mixed-logit attempt failed to beat the fixed observed-heterogeneity models,
and confirms the respondent-level train/val split and 5-fold CV are the correct
validation design (they mimic "predict for unseen respondents").

**2. Survey fatigue (Task position).** Opt-out share rises monotonically from ~24%
(Task 1) to ~34% (Tasks 15–19). Adding In_task (inside × centered Task) and P_task
(Price × centered Task) to mod8 helped: single-split val 1.1866 vs 1.1896, and 5-fold
CV 1.16221 vs mod8's 1.16618. P_task is highly significant and In_task is not -- so the
fatigue effect runs through PRICE SENSITIVITY (respondents get more price-sensitive as
the survey drags on), not a bare opt-out drift. Transfers to test (same 19 tasks).

**3. Region and parking (previously unused covariates).** Added inside × and Price ×
interactions for regionind (5 levels) and pparkind (5 levels), 16 new params. Single-
split val 1.1696 (a large jump), and -- crucially -- 5-fold CV 1.15671 vs m8t's 1.16221,
so the gain is CV-confirmed, not single-split overfitting. This "m8tr" (mod8 + task +
region + ppark) is the best single conditional-logit model. year remains unused (near-
constant / uninformative).

**4. Two-family ensemble.** Blended the best conditional logit (m8tr) with xgboost
(multi:softprob, nrounds=73). Even though xgboost alone (CV 1.1787) is worse than m8tr
(CV 1.1567), the families make different errors, so a weighted average helps. Weight
chosen by 5-fold CV on out-of-fold predictions (no leakage), flat optimum 0.65–0.75;
picked 0.70 m8tr / 0.30 xgb -> CV 1.15166. Final models refit on ALL training data;
test predictions blended and written to `submission_ensemble_v9_mlogit_xgb.csv`.
This is the current best. Source: xgboost (Chen & Guestrin 2016); ensemble averaging.

CV numbers this section use seed-4821 respondent folds; under those folds plain mod8 =
1.16618, so the progression is mod8 1.1662 -> +task 1.1622 -> +region/ppark 1.1567 ->
+xgboost ensemble 1.1517.

Reproducibility: the unified feature builder is `build_all(df, ctr, scl)` (segment +
task + region + ppark interactions, scaler from training). The CV loop refits all
mlogit specs and xgboost per fold and stores OOF matrices (OOF_m8, OOF_m8t, OOF_m8tr,
OOF_xgb) for the weight search.

## 2026-07-26: First public-LB results past mod7, and teammates' independent models

**ensemble_v9 (mod8+task+region/ppark blended with xgboost) submitted: public 1.204.**
Current best score the team has on the board by a wide margin (previous best was
mod7's 1.230). Gap vs. CV (1.204 - 1.1517 = 0.052) is noticeably larger than mod1's
(0.034) or mod7's (0.028) -- expected, since this model has far more surface area
(segment/task/region/ppark interactions plus a CV-tuned xgboost component) than the
earlier ones, so more room to fit CV-specific noise. Still a clear net win even with
the larger gap; worth flagging in the report's public-vs-private discussion rather than
assuming the gap stays constant as model complexity grows.

**Teammate Imelda Lee's independent model line** (branch `imelda`, `notebooks/
experiments/04a_mnl.Rmd`, `04b_mixed_logit.Rmd`, `05_improvements.Rmd`). She built her
own pipeline from wide-format `dfidx` reshaping (vs. this project's long-format
`pivot_longer` approach) but used the **same respondent-level split, seed 7402**, so
her numbers are directly comparable to ours. Useful cross-check: her plain MNL baseline
(ASCs + linear attrs + Price) scored **1.235696** -- identical to mod1's validation log
loss to six decimal places, confirming both independent implementations of the same
model spec agree exactly.

From her `05_improvements.Rmd` component tests (all vs. her 1.2357 baseline):
- `I(Price^2)` curvature: 1.23263 -- small gain, consistent with this project's own
  finding that a linear-in-level Price term is mis-specified (though we address it via
  full dummy-coding rather than a quadratic).
- Price x income interaction: 1.23495 -- essentially no gain on top of the linear
  baseline (contrast with this project's mod6: the same idea, income-price
  interaction, gave a real gain of ~0.03 here, but only once attributes were already
  dummy-coded first -- suggests the interaction needs a properly-specified fixed part
  under it to show up).
- Nested logit (3 real bundles nested against the opt-out): 1.23469 -- no improvement.
  Consistent with our own finding that the opt-out is best handled via a fixed
  zero-utility reference cell rather than nesting structure.
- Mixed logit, random Price only (linear attrs, not dummy-coded): 1.27954 -- clearly
  WORSE than her baseline. Differs from this project's mod10 (random Price added on
  top of the fully-featured mod8), which gave a negligible ~0.0003 *gain* -- the
  difference is that injecting simulation noise on top of an under-specified linear
  fixed part hurts, whereas adding it on top of a well-specified one (dummy-coded +
  covariate/segment interactions) has nothing left to explain.
- `mlogit_v3_combined` (her best, self-reported): 1.22028 -- factor-coded selective
  attributes (NS, BU, FP, SC, MA, LB) + covariates + random Price + Price x income.
  Source code for this specific run was never committed (it lived in a notebook she
  deleted, `03_model_experiments.Rmd`, which turned out on inspection to be an empty
  scaffold template -- the actual combined-model code was run locally and never saved),
  so it is not independently reproducible from her branch history; citing her reported
  number as-is.
- Ensemble of MNL + Price^2 + mixed(random Price), refit on full data: submitted to
  Kaggle 2026-07-26, **public 1.263**. CV 1.22777 (note: her own submissions_log.csv
  logs 1.22694 for this row, but that figure is actually a *different*, 4-component
  ensemble that includes a Price x income model which was never part of the 3-component
  refit that actually produced the submitted file -- a mismatch from rerunning notebook
  cells out of order. 1.22777 is the correct CV for the model that generated the
  submission). Gap ~0.035, unremarkable, same order as mod1/mod7.

None of Imelda's models beat mod2b (1.2186) let alone the team's later models, but the
exact match on the MNL baseline is a valuable independent correctness check on the
shared modeling approach, and the negative results (income-price, nesting, naive mixed
logit not helping on an under-specified base) corroborate findings this project reached
via a different route.

**Teammate Clarence Elvareta's xgboost attempt** (branch `clarence`, merged to `main`
via PR #1, not yet folded into this branch). Flagged as **not yet trustworthy**: her
validation split is `Task <= 12` (train) vs. `Task > 12` (test), i.e. split by task
number rather than by respondent -- the same respondent's tasks appear on both sides,
which is exactly the leakage this project's own data audit identified and designed
around (test respondents in the real competition are 263 entirely new people, never
seen in train). No log-loss number from her script was ever saved anywhere reproducible
(only printed to console via xgboost's training watchlist), so there is nothing to log
here yet. Her script also has a hardcoded `setwd("C:\\SUTD\\...")` that will not run on
another machine. Her test-set predictions exist as a file regardless (submission not
yet made) -- worth re-validating on `data_processed/train_val_split.rds` and fixing the
path before trusting or submitting it.

**Update, 2026-07-27: submitted, public 1.221.** Her `submission_ensemble1.csv` was
submitted to Kaggle. Since no internal log-loss number was ever captured (only printed
to a console watchlist and now gone), this is the first trustworthy signal of any kind
about this model's real quality -- there's no leaked/optimistic number to compare
against or discount, just an honest public score. Puts it in context against the rest
of the team's logged public scores: worse than `ensemble_v11` (1.202, current best) and
the standalone mlogit (1.213), but meaningfully better than Zeening's random forest
(1.259) despite both having leakage-flawed internal validation splits (Clarence's is
task-based, Zeening's is fully row-level-random -- a more severe form of the same
leakage). Doesn't change the current-best recommendation, but is worth noting for the
report/team record: the leakage in her validation methodology was a real problem for
trusting her *internal* number, but it says nothing bad about the model itself, which
turns out to be reasonably competent in practice. Her branch still isn't merged into
`zhenhao` and the validation split itself is still unfixed.

## 2026-07-26: Price as a saturated factor + choice-set context effects (new best)

Two ideas that had never been tried despite being obvious in hindsight: (1) every one
of the 19 attributes was dummy-coded specifically because there's no reason utility is
linear in the level code -- but Price, the single dominant driver, was left as one
linear slope the whole project; (2) respondents plausibly evaluate price partly
*relative to the choice set in front of them*, not on an absolute scale, which nothing
in the model could express.

**1. Price as a 12-level factor.** Recoded Price from continuous to dummies for levels
2-12 (reference = level 1). Levels 0 (opt-out) and 1 share the reference cell
deliberately: Price=0 occurs only for the opt-out, and the 19-attribute block already
encodes inside-vs-opt-out perfectly on its own (see the identification note below), so
giving Price=0 its own dummy on top of that is redundant, not free information.
Heterogeneity terms (Price x covariate/segment/task/region/ppark) stay linear in Price
-- only the main effect is freed up.

**Identification pitfall hit and fixed:** a full 0-12 factor (all 12 dummies, reference
level 0) makes `mlogit`'s Hessian exactly singular. Diagnosed via `model.matrix()` +
`qr()` rank (much faster than repeatedly re-fitting mlogit to guess): regressing the
dropped column on all others gave a perfect fit whose coefficients revealed the cause --
every inside alternative has **exactly 9 of its 19 attributes at a non-reference level**
(a constant of this partial-profile conjoint design, confirmed by checking the
frequency table), so `sum(attribute != 0 indicators) / 9` reproduces the "inside"
indicator exactly, just like `sum(all 12 price dummies)` does. Two different column
sets both exactly reconstructing "inside" is a rank-1 collinearity. Fixed by dropping
one price level's dummy (using levels 2-12 only) -- any one of the ~30 implicated
columns would have worked identically since it's a true structural redundancy, not
information loss.

**2. Choice-set context effects.** `is_cheapest` / `is_dearest`: 1 if this alternative
has the lowest/highest price among the task's 3 inside alternatives (computed from
`price_min`/`price_max` grouped by `chid`; ties simply both flag if they occur). Not
collinear with the price level itself since the same nominal price can be cheapest in
one task and dearest in another depending on what the other two alternatives cost.

**Results.** On top of m8tr: 5-fold CV (seed 4821) **1.1516** vs m8tr's 1.15671 -- a
real ~0.0051 gain, confirmed (not a single-split fluke; single-split was 1.165734 vs
m8tr's 1.1696). This is a SINGLE conditional logit matching the entire mod8tr+xgboost
ensemble (1.1517). Price coefficients are monotonic and convex (level2 -0.66 down to
level12 -3.64) -- confirms every prior model's linear-Price assumption was leaving
signal on the table. `is_cheapest` +0.215 (p<1e-11); `is_dearest` -0.076 (p=0.05,
weaker signal).

**Re-blended with xgboost:** same 0.70/0.30-style search (`R/cv_ensemble_v10.R`) gives
a new optimum at **0.75 mlogit / 0.25 xgboost**, pooled OOF CV **1.14823** -- beats
ensemble_v9 (1.1517) by ~0.0035. Submission file ready
(`submission_ensemble_v10_pricefactor_context.csv`), not yet submitted.

**Negative result, tested and rejected on top of this:** Price x {gender, urbanicity,
education} (symmetric completion of the existing inside x {gender,urb,educ} terms) and
a quadratic `P_task^2` fatigue term. Individually, `P_educ` and `P_task^2` both looked
strongly significant (p<1e-8), but single-split validation got WORSE (1.1699 vs
1.1657) with them added. Left out -- exactly the kind of significance-vs-validation
disconnect this project has seen before (mod11's pruning), and not worth spending a
5-fold CV run to confirm given the ensemble's CV-to-public gap is already growing with
model complexity (0.028 on mod7 -> 0.052 on ensemble_v9).

**Negative result: choice-structured xgboost.** Hypothesized that xgboost's wide-format
4-class objective (`multi:softprob`) has to infer the "one winner per 4-row task"
structure entirely from scratch, and that refitting it as binary chosen/not-chosen on
long-format rows (one row per alternative) with predictions renormalized to sum to 1
within each task would give it that structure directly and close some of the gap to
the logit. Tested (`R/test_xgb_binary_choice.R`, same eta/depth/subsample as the
existing xgboost, nrounds=200 picked from the validation-logloss curve): validation
log loss 1.2053, statistically indistinguishable from (marginally worse than) the
original wide-format xgboost's 1.2042. Simple post-hoc renormalization apparently does
not meaningfully teach the model the within-task comparison the way a proper
listwise/ranking objective might -- xgboost's native multiclass softmax was already
capturing about as much of that structure as this surrogate does. Not pursued further
(a real ranking-loss reformulation would take considerably more effort for an unproven
payoff, and xgboost is already the minority partner in the ensemble).

## 2026-07-26: Extending the price-context idea further -- price-gap magnitude (new best)

With price-as-factor and is_cheapest/is_dearest confirmed working, tried extending the
same idea two more directions: (1) does the *size* of a covariate's category matter,
the same way Price's level did, and (2) does the *magnitude* of being cheap/dear
matter, not just the rank.

**Binned covariate x Price interactions (mostly negative).** Replaced the continuous
Price x {age, miles, night} interactions with categorical bin versions, on the
hypothesis that these might hide the same kind of non-monotonicity Price did. All
three combined: validation log loss blew up to 1.1934 (vs 1.1657 without them) --
`P_nightind10`'s coefficient came out at -0.815, wildly out of scale with every other
coefficient in the model. Checked the frequency table: `nightind` levels 9-10 have
only 133 and 114 rows total (roughly 6 respondents each) -- far too sparse to support
a Price interaction, classic quasi-separation. Tested individually: age alone (5
balanced levels, smallest group ~2,070 rows) gave a small genuine improvement (1.16495
vs 1.16573); miles alone (9 levels, some thin categories) came out worse (1.16814).
**Lesson: the continuous-to-categorical trick that worked so well for Price does not
generalize automatically -- it only helps when there's enough data per cell, and needs
checking category-by-category, not assumed.** Income (25 levels) wasn't even attempted
given the sparsity risk is worse than night's. Not folded into the model given the
gain (where it existed at all) was too small to justify the complexity.

**Price-gap magnitude (real, large gain).** `is_cheapest`/`is_dearest` only encode
*rank* within the choice set -- being RM1 more expensive than the cheapest option and
being RM10 more expensive both just flip the same flag. Added `price_gap_min` /
`price_gap_max`: the actual distance (in price levels) from this alternative's price
to the task's cheapest/dearest, on top of (not replacing) the rank flags. Single-split
screen alone: 1.159681 vs the confirmed base's 1.165734 -- the single largest
incremental gain of the session, from just 2 parameters. 5-fold CV (seed 4821)
confirms it: **1.147021**, a further ~0.0046 gain over the price-factor+context model
(1.1516). Both terms are strongly significant and stable in magnitude across every
specification tried in this session (`price_gap_min` ~0.13, `price_gap_max` ~0.026) --
rank and magnitude are both real, complementary pieces of the same context effect.

**Re-blended ensemble.** Same weight search as before (`R/cv_ensemble_v10.R`) with the
stronger logit: optimum shifts to 0.80 logit / 0.20 xgboost, pooled OOF CV
**1.145094** -- beats ensemble_v10 (1.14823) by ~0.0031 and the original ensemble_v9
(1.1517) by ~0.0066 total. Submission file ready
(`submission_ensemble_v11_pricegap.csv`), not yet submitted.

Model progression this session (5-fold CV, seed 4821): m8tr 1.15671 -> +price-factor
+context 1.1516 -> +price-gap 1.147021 (logit alone) -> ensemble 1.145094.

**Negative result: proper stacking instead of a fixed blend weight.** Replaced the
single arithmetic blend weight with a real stacking model -- a conditional logit
(respects the true 4-way softmax likelihood, unlike a naive per-row binary GLM) using
log(p_mlogit) and log(p_xgb) as covariates, i.e. log-linear ("geometric") pooling
instead of linear pooling, with one more free parameter. Nested 5-fold CV (meta-model
refit on 4 folds' OOF predictions, evaluated on the held-out 5th): 1.146126, marginally
WORSE than the simple fixed blend (1.145064 at w=0.82, same OOF data). The extra
flexibility didn't help because there's really only one meaningful degree of freedom
in a two-model ensemble where one model (xgboost) gets a small minority weight anyway
-- the simple weighted average already finds it. Kept the simple blend as the
submission of record.

**On whether log loss can be pushed drastically lower (e.g. into the 1.0x range):**
almost certainly not through further legitimate feature engineering on this dataset.
mod4 (fully-random mixed logit, the most flexible/overfit model tried, with 20
respondent-specific random coefficients) achieved a training log-likelihood of -16449
-- converting to log loss puts even that theoretical ceiling (best possible fit ON
ALREADY-SEEN respondents, with unlimited respondent-specific flexibility) at roughly
0.76-0.95, and none of that respondent-specific flexibility transfers to the actual
test respondents (263 entirely new people). The achievable log loss using only what
generalizes (observed heterogeneity -- exactly what this project has spent all its
effort on) necessarily sits above that floor. Today's gains have followed a classic
diminishing-returns curve (mod6->mod7 ~0.03, segment ~0.013, price-factor ~0.005,
price-gap ~0.005, stacking ~0.000), consistent with approaching the practical floor
for this kind of repeated stated-preference conjoint survey, where genuine
respondent-level inconsistency/fatigue/satisficing is not explained by any observable
-- this is worth citing directly in the report's limitations section.

## 2026-07-26: Error/calibration diagnostic -- is there more legitimate signal left?

After the price-factor/context/price-gap gains and two negative results (stacking,
binned covariates beyond age), ran a diagnostic on the ensemble's OOF predictions
(`R/error_analysis.R`, `R/calibration_check.R`) to check whether the remaining error
looks like fixable, generalizable signal or genuine irreducible noise.

**Reliability/calibration is excellent.** Binning every predicted probability (across
all 4 alternatives x all tasks) against whether that alternative was actually chosen:
predicted and actual match within 1-2 percentage points across the entire 0-0.8 range
(e.g. predicted ~0.45 -> actual ~0.46; predicted ~0.64 -> actual ~0.63), only drifting
at the very top of the range where sample sizes are tiny (n<250). When the model says
"60% chance," it is right about 60% of the time.

**"Confident misses" are the expected flip side of calibrated confidence, not
miscalibration.** 49.5% of tasks are argmax-misses; among those, 36.3% have a gap
>0.30 between the top pick's probability and the true alternative's probability. This
sounds alarming in isolation, but a model that is genuinely 60% confident *should* be
wrong 40% of the time, and those misses will show large gaps precisely because the
model wasn't hedging on a close second choice. The calibration check confirms this is
exactly what's happening, not a sign of a fixable bias.

**No identifiable subgroup drives the misses.** Confident-miss rate (gap>0.30) is flat
across every slice checked: true class (0.154-0.208), segment (0.154-0.201), task
position bucket (0.171-0.195), region (0.175-0.182). If a generalizable pattern were
being missed, some slice should stand out; none does.

**Log loss is not concentrated in a few catastrophic failures.** The worst 10% of
tasks (by their own log-loss contribution) account for 21.4% of total log loss, worst
30% for 50.4% -- broad, roughly proportional spread, not a small number of badly-wrong
predictions that a targeted fix could clean up.

**Conclusion:** combined with mod4's training-log-likelihood ceiling (~0.95, achieved
only via non-transferable respondent memorization -- see above) and xgboost's failure
to out-predict the hand-built logit despite having the same raw covariates and full
flexibility to find missed interactions, this is a reasonably strong, multi-angle case
that the ensemble (CV 1.145) is close to the practical floor for this dataset using
legitimate, generalizable modeling. Worth citing directly in the report's
insights/limitations section as evidence-based, not just an assertion.

## 2026-07-26: External review round -- adversarial validation, design overlap, bootstrap uncertainty, latent class

Got two independent LLM reviews of the project (using the AGENTS.md summary as the
brief) and ran the concrete, checkable suggestions rather than just taking them on
faith. Two real, previously-unknown structural facts came out of it; two follow-up
fixes tested null/ambiguous; one new model idea looks promising but isn't CV-confirmed
yet.

**1. Adversarial validation: real covariate shift, driven by income
(`R/adversarial_validation.R`).** Fit a 5-fold-CV logistic classifier to distinguish
train vs. test respondents from covariates alone. AUC = **0.634** (vs 0.5 for no
shift) -- a real, detectable difference in who's in the two panels. `incomeind` is by
far the most significant term (p<0.0001). Confirmed via raw comparison, not just an
artifact of outliers: median `incomea` is 60,000 (train) vs 80,000 (test), a genuine
~33% shift, and the top income brackets are 3x+ over-represented in test (bracket 28:
3.4% of test respondents vs 1.1% of train; bracket 14: 9.1% vs 2.5%). This is a real,
partial explanation for the growing CV-to-public gap that neither this project nor
either reviewer had checked before.

**2. Design/block structure: the conjoint design is heavily reused, but the reuse
isn't exploitable (`R/design_fingerprint_check.R`, `R/design_cell_shrinkage.R`).**
Fingerprinting each choice task by its exact 4-bundle attribute/price configuration
(excluding respondent identity) revealed the experiment is **blocked**: each of the 19
task positions draws from a fixed pool of only ~296 distinct designs, each shown to
~3.8 respondents on average. Critically, **98.5% of test choice tasks (4,921/4,997)
use an exact design that also appears somewhere in train.** This is a genuine,
previously-unknown structural fact about the dataset. Tried to exploit it directly:
for each task, blend the model's prediction with the empirical choice-share among
OTHER training respondents who saw that exact design (properly cross-fitted per CV
fold -- a respondent's own choice never informs their own prediction), shrunk toward
the model via `(n*p_empirical + alpha*p_model)/(n+alpha)`. Result: **negligible gain**
-- best alpha (80-160, i.e. very heavy shrinkage) gives 1.144742 vs the ensemble's
1.145094 baseline, an improvement of 0.00035, inside the noise floor established below.
Light shrinkage (alpha<20) actively hurts a lot (up to 1.29) since only ~3-4
respondents see each design -- too few to estimate a reliable empirical frequency.
Worth citing in the report as an insight (a real, non-obvious fact about the
experimental design) even though it didn't yield a usable feature; it also indirectly
supports the "near the practical ceiling" conclusion, since a model missing real
combination-specific effects should have benefited more from this.

**3. Bootstrap CV uncertainty (`R/bootstrap_cv_uncertainty.R`).** Neither this project
nor either review had ever put a formal noise band on the CV deltas being compared
(e.g. 1.147 vs 1.145). Bootstrapped the 1135 training respondents (with replacement,
500 resamples) using the saved OOF predictions. Absolute CV log loss has SD ~=0.0099
(a 95% CI of roughly +/-0.02 around any single point estimate) -- much wider than the
0.003-0.006 deltas discussed all session. However, *paired* comparisons (same
resamples, same respondents, for two models at once) are much tighter, SD ~=0.001,
because respondent-level variation cancels in the difference. Under that lens: the
big wins (price-factor, price-gap, each ~0.005) are comfortably real, several
noise-SDs wide. The xgboost blend's own contribution (ensemble vs mlogit alone,
0.0019) is real but close to the edge (ensemble wins in 96.6% of resamples, not
99%+). The stacking "null result" (~0.001 apart) sits entirely inside the noise
band -- confirms that call was correct, not just a coin flip we got lucky on.

**4. Importance-weighted shift diagnostic (`R/importance_shift_diagnostic.R`):
ambiguous, not actionable as-is.** Given the confirmed income shift, checked whether
the model is specifically weaker on test-like respondents by reweighting the OOF
evaluation using density-ratio importance weights (from the adversarial classifier).
Weighted mean log loss (1.15768) is worse than unweighted (1.147021) -- suggestive.
But a simpler univariate check (log loss by training income tercile) shows the
OPPOSITE pattern: the high-income tercile has the BEST log loss (1.140), not the
worst. So the shift is real, but it's not simply "the model is bad at rich people" --
some more specific multivariate combination is involved, and per Claude's review's own
caveat, a genuine distribution-shift correction can't be honestly validated via
in-training CV (CV will always prefer no correction, since held-out training folds
share the training distribution, not test's). Not implemented; flagged as real but
needing either a submission-slot experiment or a more careful joint-covariate
investigation to pin down, rather than forcing an unvalidated fix.

**5. Latent-class task-fatigue model: promising on a single split, not yet
CV-confirmed (`R/latent_class_screen.R`, `R/latent_class_evaluate.R`).** Both external
reviews independently flagged finite-mixture/latent-class logit as the one
structurally different idea worth trying (unlike continuous mixed logit, class
membership is predicted from *observed* covariates, so it transfers to new
respondents by construction). `gmnl` (the standard R package for this) turned out to
have two practical blockers: no `predict()` method with `newdata` support at all
(only in-sample `fitted()`), and no native support for a *restricted* class structure
(shared attributes, few class-specific terms) -- its standard interface gives every
variable a fully separate coefficient per class, which would mean ~230 parameters
from 908 respondents. Hand-rolled a scoped alternative instead: fixed the entire
confirmed m8trpg utility as a shared baseline (via `log(predicted prob)` as an offset,
which sidesteps fragile manual design-matrix alignment against new data since softmax
is shift-invariant to an additive constant), then used EM to fit a small 2-class
extension on just the task-fatigue terms (`P_task`/`In_task`, chosen because they're
safe from the price-factor collinearity found earlier), with class membership
predicted from segment/income/age. All 3 random EM restarts converged to the same
parameters (reassuring against the multimodality risk both reviews warned about).
Single-split result looked promising (1.160589 vs the current model's 1.165734), but
two things followed that killed it.

**Bug found before trusting the single-split number.** The EM's M-step duplicated
every row into a class-1-weighted and class-2-weighted copy and fit ONE combined
weighted GLM across both copies to get "class-specific" coefficients. Since the two
weights sum to 1 for every row identically (w1=1-post2, w2=post2), a single-formula
fit across both duplicates is mathematically IDENTICAL to an unweighted fit on the
original data -- it doesn't depend on the class posterior at all. This explained the
earlier "reassuring" observation that all 3 random EM restarts converged to identical
parameters: the M-step literally could not produce a different answer regardless of
initialization. Fixed by running two SEPARATE weighted GLMs, one per class, each
using only its own posterior weight vector (`R/latent_class_screen.R`,
`R/cv_latent_class.R`).

**Full 5-fold CV with the fix (`R/cv_latent_class.R`): null result, model is
unstable.** Pooled CV: shared baseline with no task-fatigue term at all = 1.150765;
2-class latent mixture = **1.147826** -- actually slightly WORSE than the existing
model's shared single task-fatigue coefficient (1.147021). Worse still, the
class-specific coefficients are wildly unstable across folds and even flip sign:
fold 1 (beta_task2=-0.124, beta_intask2=+0.581), fold 3 (+0.004, -0.104), fold 4
(+0.165, -0.956). This is exactly the multimodal-likelihood risk both external
reviews warned about, now demonstrated concretely rather than just anticipated: with
only 908 training respondents and 3 membership covariates, the EM finds a different,
non-generalizable "class 2" depending on which respondents happen to be in that
fold's training portion. The real, CV-confirmed signal here is just that task-fatigue
matters at all (1.150765 -> ~1.147 either way) -- something already known; the
latent-class structure adds instability without adding predictive value. NOT
adopted. This closes out the external-review round: two genuinely new, real
structural findings (income shift, blocked-design overlap), and every concrete idea
tested from either review (design-cell shrinkage, latent-class) came back null once
properly validated -- consistent with the calibration-based "near the practical
ceiling" conclusion from earlier in the day.

## 2026-07-27: A competing team's public score (1.187) prompts one more test -- hypothesis refuted, but informative

A teammate reported another team's public leaderboard score of 1.187 -- only 0.015
below our 1.202, not the "low-1.1x" gap originally assumed. Proposed a concrete,
testable explanation before speculating further: ensemble_v11's CV-to-public gap
(0.057) is unusually large for the tiny CV gain the xgboost blend actually provides
(0.0019, right at the edge of the bootstrap noise floor established earlier) -- maybe
the blend was adding an "ensemble complexity tax" that a leaner single model could
avoid, and a standalone mlogit_m8trpg submission might transfer better and land
closer to 1.187.

**Tested directly rather than left as a guess: submitted mlogit_m8trpg alone (no
xgboost), `submission_mlogit_m8trpg_only.csv`.** Result: public **1.213** -- WORSE
than the full ensemble (1.202), with a gap of 0.065979, the largest of any model in
the project. Hypothesis refuted.

**What this actually shows, which is more useful than the original guess:** blending
in xgboost -- a model that never beats the logit alone on CV (1.1787 vs 1.147) --
genuinely reduces the public-facing generalization gap rather than adding to it. This
is a clean, empirically-confirmed instance of classic ensemble variance reduction:
averaging two model families with different error patterns produces a more robust
prediction even when one family is individually weaker, and that robustness shows up
specifically when moving from the training distribution to a shifted one (matching
the confirmed income-based covariate shift from the review round). Practical
takeaway: keep the xgboost blend, don't simplify it away in pursuit of a smaller
gap -- the gap size alone is not a reliable signal of which model will generalize
better. Still does not explain the competing team's 1.187 -- that remains an open
question the project hasn't found a lever for.

Updated CV-to-public gap table (all models submitted so far):

| Model | CV/Val | Public | Gap |
|---|---|---|---|
| mod1 | 1.236 | 1.270 | 0.034 |
| mod7 | 1.202 | 1.230 | 0.028 |
| ensemble_v9 | 1.152 | 1.204 | 0.052 |
| ensemble_v11 | 1.145 | 1.202 | 0.057 |
| mlogit_m8trpg (standalone, no xgboost) | 1.147 | 1.213 | **0.066** |

The gap is not monotonic in CV quality or even in model complexity alone -- it
depends on which *kind* of complexity (a diverse second model family vs. more
interaction terms in the same family). Worth stating carefully in the report rather
than the simpler "gap grows with complexity" framing used earlier in the day; the
more precise version is "gap grows with model-specific overfitting risk, and
ensembling across diverse families appears to mitigate rather than compound it."

## 2026-07-27: Quantifying how much of the gap the income shift explains (`R/income_gap_decomposition.R`)

Follow-up on the confirmed income shift, per external-reviewer feedback pushing for a
single decomposed number rather than the earlier ambiguous weighted-vs-unweighted
comparison. Reused the existing OOF predictions and adversarial-classifier importance
weights (no new fitting, no submission cost).

**Gap decomposition:** importance-weighted CV log loss (mimicking test's covariate
distribution) = 1.15768 vs unweighted 1.147021. Difference = **0.010659**, which is
**~18.7% of ensemble_v11's total CV-to-public gap (0.057)**. The confirmed shift
explains a real but MINORITY share of the gap -- the rest is genuinely something
else (public-sample noise, or factors this project hasn't identified).

**Per-income-bracket OOF loss (finer than the earlier tercile check): mixed, not a
clean story.** Checked the specific brackets most over-represented in test:
- Bracket 14 (2.5% of train respondents -> 9.1% of test, a 3.7x jump): OOF log loss
  1.246, meaningfully worse than the overall mean (1.147). Matches the hypothesized
  "extrapolation into a test-common but train-rare region" mechanism.
- Bracket 28 (1.15% -> 3.4%, a similar-sized jump): OOF log loss 1.063, BETTER than
  average -- the opposite pattern.

So it is not "high income predicts badly" as a general rule; one specific bracket
looks like a real weak spot, another does not. Bracket 14's estimate rests on ~28
respondents (532 rows), so a 0.10 deviation is suggestive but not overwhelming --
rough scaling from the bootstrap SD established earlier puts a 28-respondent
subgroup's own noise around +/-0.06. **Important correction to a specific suggested
fix:** it was proposed that "replacing binned income in the interactions with
continuous incomea" would help -- but P_income/In_income already use continuous
incomea, not incomeind, confirmed earlier in the project (see the "*a* covariates
are genuinely richer than *ind*" data-quality finding). If bracket 14 is real, the
mechanism would have to be linear-extrapolation risk into a sparse income tail, not
discrete-bin sparsity -- a different, less clear-cut story than the nightind
sparsity case, and not one this project has enough evidence to act on yet (one
flagged bracket confirming, one contradicting, on a modest sample). Not implementing
a speculative nonlinear-income respecification from this alone; logging it as solid,
nuanced report material for the public-vs-private section instead.

## 2026-07-27: Closing the income-shift and 1.187 investigation with proper uncertainty

A second round of external-reviewer feedback pushed for more statistical rigor on
three fronts: a log-transformed income model test, a properly respondent-clustered
standard error for judging the competing team's 1.187 score, and a cleaner
(univariate, not conflated-with-other-covariates) income-weighted decomposition with
an effective-sample-size check and bootstrap CIs. All three were worth doing and
closed the question more solidly than the earlier pass.

**Model test: log(1+income) instead of raw linear income -- null (`R/test_log_income.R`).**
Motivated by the (correct, standalone) observation that income is likely right-skewed
and a raw linear z-score could give undue leverage to extreme values -- independent
of the separate (incorrect) claim that our interactions use binned income; they use
continuous `incomea` already, confirmed twice now. Single-split result: 1.1601 vs the
confirmed base's 1.159681 -- a negligible +0.0004, clearly null. Makes sense in
hindsight given the per-bracket picture below: the weakness (if real) is narrow and
bracket-specific, not a broad "extreme values dominate" problem a global reshaping
would fix.

**Respondent-clustered public-sample noise (`R/public_sample_noise_clustered.R`):
confirms the reviewer's clustering concern, and the conclusion is even stronger than
before.** Simulated public-LB-sized draws (184 respondents, ~70% of the 263 test
respondents) from the training OOF predictions. Clustered SD for a single draw =
0.0225 -- **2.14x larger** than the naive row-level estimate (0.0105) that ignores
within-respondent correlation, confirming the row-level SE understated uncertainty
as flagged. More importantly: simulating the DIFFERENCE between two independent
same-model draws (i.e., "how far apart could two equally-good models' public scores
look purely from sampling luck") gives SD 0.0325, and **64.6% of simulated
same-model draw-pairs show a gap >= 0.015** -- the observed 1.202 vs 1.187 gap is not
just "plausible," it is the MAJORITY outcome even when there is truly no underlying
quality difference. Strengthens (not just confirms) dropping this as an open mystery.

**Cleaner income-weighted decomposition (`R/income_weighted_decomposition_v2.R`):
point estimate confirmed, but the confidence interval includes zero.** Using a clean
univariate density-ratio weight (test income-bracket share / train income-bracket
share, at the respondent level, not conflated with the other 14 covariates the
adversarial classifier used) gives delta = 0.010311 -- consistent with the earlier
0.010659 from the multivariate weight, a reassuring cross-check. Effective sample
size n_eff = 718 of 1135 respondents, NOT dominated by a handful of people. However,
the bootstrap 95% CI on delta is **[-0.0057, 0.0290] -- includes zero.** The point
estimate (~18% of the total gap) is our best guess, but we cannot statistically rule
out that the income shift's true contribution to the gap is zero.

**Per-bracket bootstrap CIs: the "mechanism" story does not hold up.** Bracket 14
(n=28 respondents): mean OOF loss 1.246, 95% CI [1.095, 1.408] -- CONTAINS the
overall mean (1.147). Bracket 28 (n=13): mean 1.063, 95% CI [0.941, 1.183] -- also
contains the overall mean. Neither flagged bracket is statistically distinguishable
from the average once its own sampling uncertainty is accounted for. The apparent
"one bracket bad, one bracket good" pattern from the point estimates alone is fully
consistent with noise at these sample sizes (13-28 respondents) -- exactly the
caution the reviewer itself raised, now confirmed by the data rather than assumed.
No targeted, bracket-specific fix is supported by this evidence.

**Conclusion, now on solid statistical footing rather than point estimates alone:**
the income shift may contribute something to the CV-public gap (point estimate ~18%,
but not statistically distinguishable from zero), there is no reliable evidence of a
specific fixable mechanism within it, the log-income model test confirms this with a
null result, and the competing team's 1.187 score is not just "not clearly anomalous"
but is the typical, majority outcome under ordinary sampling variation between two
comparably-good models. This fully closes the external-review investigation that
began 2026-07-26.

## 2026-07-27: Testing the CENTRAL latent-class idea properly -- price-sensitivity classes, not just task-fatigue

Both external reviews' original latent-class pitch was a **price-sensitivity/opt-out**
segmentation (a price-insensitive "enthusiast" class vs. a price-sensitive one) --
the task-fatigue version tested earlier was a scoped-down, collinearity-safe
substitute, not the central idea. Went back and tested the real thing properly.

**Why price-sensitivity couldn't be added the same (additive) way as task-fatigue.**
An additive class-specific shift on "inside" or "Price_num" directly would recreate
the exact collinearity found earlier (price dummies sum to "inside", same as the
19-attribute active-count identity). Solution: a class-specific **multiplicative
scale** on the price-related portion of the linear predictor instead (a discrete
version of the "heteroskedastic scale" framing one review raised for the fatigue
effect, applied here to price sensitivity specifically) -- `eta_class_q =
eta_nonprice + lambda_q * eta_price`, where `eta_price` sums every price-related
term's fitted contribution (the 11 price-level dummies, all `P_*` covariate/segment/
region/ppark/task interactions, `is_cheapest`/`is_dearest`, `price_gap_min/max`) and
`eta_nonprice` is everything else. Only 1 free utility parameter per class (the
scale) plus the membership model -- more parsimonious than the task-fatigue version.

**Implementation note:** getting `eta_price`/`eta_nonprice` for new (held-out) data
required a different trick than the offset approach used for task-fatigue, since a
single combined offset can't be split into two pieces after the fact. Computed
`eta_price` directly from known feature columns x fitted coefficients (no
`model.matrix()` on new data -- that approach mismatched columns before), and
`eta_nonprice` as `log(predicted prob) - eta_price`. `log(predict())` only recovers
the true linear predictor up to a per-task additive constant (softmax
normalization), but that constant is identical across all 4 alternatives in a task,
so it cancels in the final softmax regardless of how the two pieces are recombined
-- verified this to machine precision (8.9e-16) before trusting anything, and
separately verified that setting both classes' lambda to 1 exactly reproduces the
known-correct single-population baseline number (1.159681) to 6 decimal places.

**Result: stable, real, but a wash.** Single-split screen (`R/latent_class_price_scale_screen.R`)
found lambda = (0.495, 1.904) -- one class at roughly half normal price sensitivity,
another at nearly double -- IDENTICALLY across all 4 random EM restarts (unlike the
unstable, sign-flipping task-fatigue attempt). Full 5-fold CV
(`R/cv_latent_class_price_scale.R`) confirms the stability: lambda pairs across the
5 folds are (0.52,1.94), (0.49,1.85), (0.50,1.82), (0.51,1.93), (0.52,2.01) -- a
genuinely reproducible, well-identified split, not an artifact of one fold's
respondents. But the pooled CV log loss is **1.147629 vs the shared single-population
model's 1.147021** -- a difference of 0.0006, well inside the established noise floor
(~0.001 SD for paired comparisons). Per-fold results are mixed (fold 1 and 5 favor
the mixture by ~0.004-0.005, folds 2-4 favor the shared model by ~0.001-0.008),
consistent with a true null rather than a real effect in either direction.

**Interpretation:** this is a materially different, more informative null result than
the task-fatigue attempt. The heterogeneity is real and stable -- roughly two
populations with meaningfully different price sensitivities, predictable to some
degree from segment/income/age -- but capturing it as a discrete class provides no
net predictive advantage over the continuous covariate interactions (P_income,
P_seg, P_age, etc.) already in the model. Most likely explanation: those continuous
terms already capture the same underlying heterogeneity, just parameterized
smoothly rather than as a hard 2-class split, so the discrete structure is redundant
rather than wrong. This closes the latent-class investigation properly: both the
safe (task-fatigue, unstable) and central (price-sensitivity, stable-but-redundant)
versions have now been tested to the same standard as the project's confirmed wins,
and neither survives. Not adopted; ensemble_v11 remains the best model.

## 2026-07-27: Reviewing Codex's modeling push -- ranking xgboost, retuned xgboost, reconstructed glmnet-Cox, and a 4-way ensemble candidate

Asked Codex (a separate coding agent) to work on its own branch (`codex-modeling`,
commit `b37ecc8`) toward a genuine push below 1.200 public, using the "genuinely
still open" leads already flagged in AGENTS.md's negative-results list -- most
notably a real ranking-loss xgboost objective, which the earlier "choice-structured
xgboost" attempt (naive binary renormalization) never tried. Nothing in the
existing `R/` scripts or logs was touched; Codex worked only in 5 new
`R/codex_*.R` files plus `codex_findings.md`. Reviewed the actual code line by
line rather than taking the write-up on faith, per the instruction to
independently verify before deciding whether to submit.

**What Codex built, and what checked out.** `R/codex_modeling_common.R` reuses the
canonical fold split (`fold_of_case` from `data_processed/oof_ensemble_v10.rds`,
never regenerated) and provides shared truth/OOF matrix builders plus a
softmax-weight blend-search utility -- sound on inspection.
`R/codex_rank_xgb.R` fits genuine `rank:ndcg`/`rank:pairwise` xgboost with `qid`
grouping by choice task (the real ranking-loss approach the earlier null result
never tried), converting margins to probabilities via a **cross-fitted** softmax
temperature (scale learned on 4 folds, applied to the 5th, so no fold calibrates
its own conversion). `R/codex_xgb_retune.R` re-screens xgboost's own
hyperparameters more broadly than the original ensemble_v11 search.
`R/codex_glmnet_cox_ensemble.R` rebuilds the 2026-07-25 stratified-Cox regularized
conditional logit from scratch (the original script was never committed) and adds
proper nested 5-fold CV it never had; the reconstructed design matrix was checked
via `stopifnot` to match the original's exact 63-core/195-candidate column counts,
and correctly uses a **raw** softmax (no temperature) to convert the Cox linear
predictor to probabilities -- correct precisely because the stratified-Cox/
conditional-logit equivalence is exact, unlike the ranking margins which have no
such guarantee and legitimately need calibration. `R/codex_ensemble_diagnostics.R`
combines all four OOF sources (mlogit, original xgboost, rank:ndcg, retuned
xgboost, glmnet-Cox) into every candidate pool, selects weights via convex
optimization (softmax-parameterized, so weights are automatically non-negative and
sum to 1), and runs a respondent-clustered bootstrap matching the methodology
already established in `R/bootstrap_cv_uncertainty.R`.

Traced every OOF alignment path by hand (all three new components use the same
`match(..., train$No)`-based pattern to fill an `nrow(train)`-row matrix in
canonical row order) and confirmed the nested/cross-fitted weight selection has no
leakage: for each outer fold, the blend weights used on that fold's held-out rows
are fit only on the other four folds' OOF predictions and truth. Independently
recomputed everything from the actual saved intermediate files (`data_processed/codex/*.rds`,
gitignored but still present on disk from Codex's run) rather than trusting the
prose. Cross-checked several numbers against independent sources rather than just
internal consistency: the reconstructed 2-way (mlogit+xgboost) blend's global OOF
log loss (1.14506) matches the officially-logged ensemble_v11 CV number (1.145094)
almost exactly, and the `v11_pred` baseline used for the bootstrap comparison uses
`w = 0.80`, matching `R/submit_ensemble_v11.R`'s actual submission weight exactly
(not a guess).

**One confirmed inconsistency, now explained (stale artifact, not nondeterminism).**
`data_processed/codex/rank_cv.csv` reports the `rank:ndcg` config's cross-fitted
log loss as 1.163610 -- also the number quoted in `codex_findings.md`. But
`component_scores.csv` (written by `codex_glmnet_cox_ensemble.R`) recomputes the
same nominal quantity from `rank_oof.rds[[1]]$pred` and gets 1.162710 instead, a
~0.0009 gap. Initially suspected xgboost's multi-threaded floating-point
non-associativity; Codex corrected this and it checks out against the file
timestamps (`data_processed/codex/`, all times SGT): `component_scores.csv` and
`cox_oof.rds` were written at 13:10:37, **before** the ranker's temperature
calibration was cross-fitted -- that fix landed in a later run of
`codex_rank_xgb.R` that overwrote `rank_oof.rds`/`rank_cv.csv` at 13:18:11.
`codex_glmnet_cox_ensemble.R` was never re-run afterward to refresh
`component_scores.csv`, so that one file's `rank_ndcg` row is a stale artifact
from the pre-calibration-fix version, not a live measurement. **1.163610 (with
cross-fitted calibration) is the canonical number.** Critically, the actual
decision-relevant numbers are unaffected: `codex_ensemble_diagnostics.R` (which
produces `ensemble_meta_cv.csv`/`ensemble_bootstrap.csv`, timestamped 13:19:06)
reads `rank_oof.rds` directly and runs strictly after its final 13:18:11 write,
so the headline 4-way blend result (1.144029/1.144363) and the bootstrap CI
already used the corrected, post-calibration-fix ranker OOF. Sensible hygiene
fixes going forward: force `nthread=1` for exact run-to-run reproducibility
regardless, and regenerate every dependent artifact in one uninterrupted
pipeline run rather than treating intermediate `.rds` files as stable across
partial re-runs -- but there is no actual uncertainty in the reported headline
numbers from this issue.

**Results.** Individually: `rank:ndcg` xgboost alone scores 1.163610 (vs the
original xgboost's 1.178668) and blends with mlogit_m8trpg to 1.144904; retuned
xgboost alone scores 1.176029 and blends to 1.144610; the reconstructed glmnet-Cox
scores 1.164331 alone and takes a stable 9-13% weight in every blend it enters
(more stable across folds than the two xgboost variants). The best overall pool
(mlogit 68% / rank:ndcg 10% / retuned xgboost 12% / glmnet-Cox 11%) scores
1.144029 with in-sample-optimized weights and **1.144363 under honest
fold-cross-fitted weight selection**, vs. ensemble_v11's official 1.145094 -- a
~0.0007 gain. A respondent-clustered bootstrap (1000 resamples of the 1135
training respondents) puts this at mean gain 0.000742, win rate 93.3%, **95% CI
[-0.000214, 0.001775] -- the interval crosses zero.**

**Decision: not adopted, no submission made.** Two independent reasons, not one.
First, the bootstrap CI includes zero -- by this project's own established
standard for treating a result as real (see the bootstrap-uncertainty section
above), a gain whose CI crosses zero is not distinguishable from noise. Second,
and separately, this project has already confirmed (see the CV-to-public gap
table in AGENTS.md and the `mlogit_m8trpg_standalone` result) that adding model
complexity here tends to WIDEN the public-leaderboard generalization gap even when
CV genuinely improves -- and this candidate roughly doubles the model count behind
the current best (2 components -> 4) for a gain that isn't even confidently real
in CV, let alone likely to survive the public-LB discount this project has
observed at every previous complexity increase. If a future result clears both
bars at once (CI clearly excluding zero, and not adding meaningfully more moving
parts) it would be worth a submission slot; this one doesn't. All 5
`R/codex_*.R` scripts and `codex_findings.md` are kept in the repo for
reproducibility/transparency, consistent with every other tested-but-not-adopted
model in this project (task-fatigue latent class, stacking, K-means personas,
design-cell shrinkage, etc.) -- a well-executed, honestly-reported null result is
still worth keeping on record.

## 2026-07-27: Covariate-shift refit and attribute-rank experiments (both negative)

Asked Codex to chase two specific, previously-untested leads rather than another
generic "optimize more" pass, given how much ground the project had already
covered: (1) an actual weighted-likelihood refit targeting the confirmed
train-test income/covariate shift (AUC 0.634) -- everything tried against that
shift so far had only reweighted the *evaluation* of the existing fit, never
retrained anything; and (2) extending the price_gap/is_cheapest/is_dearest
choice-set-rank mechanism (the single biggest confirmed win of the session) to
the other 19 attributes, since a *sum*-based version of "relative feature load"
had been tried and was noise, but the *rank*-based version specifically was
never tested. Work landed on branch `codex-shift-ranks` (commit `51ad3d8`,
merged into `zhenhao`) as 3 new scripts + `codex_shift_rank_findings.md`;
nothing existing was touched. Independently reviewed both experiments' code and
cross-checked every reported number against the raw generated CSVs in
`data_processed/codex_shift/` (gitignored but present on disk from the actual
run) before accepting the write-up.

**1. Importance-weighted refit -- decisively negative, not just null.** Fit a
respondent-level logistic domain classifier (same 15 covariates as
`R/adversarial_validation.R`) to distinguish training-fold respondents from
actual test respondents, converted its output to a density ratio via the
standard Bayes'-rule identity (`p(test|x)/(1-p(test|x)) * n_source/n_target`),
capped it at 20 and mean-normalized it, then used `ratio^alpha` as `mlogit`
case weights for alpha in {0, 0.25, 0.5, 0.75, 1} -- a genuine weighted-MLE
refit (Shimodaira 2000), not a reweighted evaluation. Crucially, each fold's
evaluation weights were estimated by a classifier that excludes that fold's own
respondents from its source class (no self-referential leakage), and every
alpha was scored against the *same* full-strength (alpha=1) target-weighted
held-out loss, so only the fit itself varies across the comparison. Verified
`alpha=0` exactly reproduces the known baseline (1.159681 single-split /
1.147021 CV) before trusting anything else -- confirms the refit machinery
itself is correct. Result: every nonzero alpha makes both ordinary and
target-weighted loss worse, monotonically, in all 5 folds individually. At the
gentlest setting tested (alpha=0.25), the respondent-clustered, importance-
weighted bootstrap (Hajek ratio estimator, matching how the weighted mean
itself is computed) puts the loss vs. baseline at -0.001416, **95% CI
[-0.002614, -0.000422] -- excludes zero on the harmful side**, not just an
unconfirmed gain but a confirmed harm. Re-optimizing the mlogit/xgboost blend
weight per alpha doesn't rescue it (blend CI [-0.001398, -0.000276], same
story). Cause: capping/normalizing the ratio still costs real effective sample
size (908 -> ~521 of 908 respondents at alpha=1), and a finite parametric
choice model pays a variance price for fitting a sparser, reweighted sample
that outweighs any targeting benefit. This doesn't contradict the confirmed
covariate shift itself -- it shows this specific, carefully-implemented
correction makes the model worse on its own honest target-risk proxy, closing
off the one previously-open, actually-untested lever from the covariate-shift
diagnostic.

**2. Attribute min/max rank flags -- null, didn't replicate.** Added 38 terms
(min/max indicator per attribute among the 3 inside alternatives, ties keep
multiple flags) to m8trpg. An all-4-alternative version was also tried and is
computationally singular (alt 4's all-zero profile recreates the known
inside/opt-out identification issue) -- correctly discarded rather than
regularized around. The inside-only version looked promising on the single
split (1.158879 vs. 1.159681, -0.000803) but reversed under 5-fold CV
(1.147275 vs. 1.147021, +0.000254, worse in 4 of 5 folds); bootstrap 95% CI
[-0.001138, 0.000572] crosses zero with only 26.7% of resamples favoring it.
Plausible explanation, and an important limit on the price-rank analogy: price
codes are ordered with a stable economic direction, but most attribute level
codes are categorical labels without a consistent cardinal ordering, so
min/max comparisons on them can encode arbitrary rather than meaningful splits.
The mechanism that won for price doesn't transfer automatically just because
it's the same mathematical operation.

**Decision: neither adopted, no submission made.** Closes out both leads
identified as genuinely untried after the ensemble-candidate review above. The
project has now tested an actual fix (not just a diagnostic) for the confirmed
covariate shift, and the natural extension of its biggest single win, and both
came back negative under the same CV + respondent-bootstrap standard used
throughout. `ensemble_v11` remains the best and current submission.

## 2026-07-27: Continuous higher-order interaction experiments -- one genuine but unconfirmed clue

One more targeted follow-up to Codex (branch `codex-triple-products`, commit
`d987efa`, merged into `zhenhao`), explicitly scoped to avoid repeating the
earlier binned-covariate failure: that attempt binned continuous covariates
into categories and blew up from sparse-cell quasi-separation (a 6-respondent
`nightind` bin drove a coefficient to -0.815). This round tests **continuous**
three-way product terms instead (`Price x z(covariate1) x z(covariate2)`,
standardized, no binning) -- a fundamentally different risk profile, since
there's no discrete cell to be sparse in. Also tested segment-specific
covariate x price slopes as a second design, with segment respondent counts
checked first (128/383/49/211/58/306 in the full training set -- the two
smallest are comparable in size to the small K-means clusters flagged earlier
as high-variance, so segment-specific results were treated cautiously even
though they don't share the categorical sparse-bin mechanism). Independently
reviewed the code and cross-checked every reported number against the raw
generated CSVs in `data_processed/codex_triples/` -- exact match throughout,
same clean result as the previous round.

Screened three covariate pairs with income (age, mileage, night) in both
joint (Price + inside) and atomic (price-only / inside-only) forms, plus one
combined 6-term and one 20-term segment-slope model. Only income x mileage
terms and the segment-mileage-slope model beat the single-split baseline;
everything else (income x age, income x night, the other segment covariates,
the 20-term combined segment model) was screen-negative and correctly not
promoted to CV.

**Best candidate: `Price x z(income) x z(mileage)`.** Single-split screen:
1.157576 vs. m8trpg's 1.159681. Five-fold CV: mlogit alone improves
1.147021 -> 1.146782 (+0.000239); blended into the *fixed* 0.80/0.20
ensemble_v11 weight (not re-optimized -- the script hard-asserts it
reproduces the official 1.145094 blend before comparing anything): 1.145094
-> 1.144599 (+0.000495). The coefficient is directionally stable and negative
in all 5 individual folds (-0.0541 to -0.0270) -- a real, consistent signal in
the *parameter* -- but the *predictive* gain is not: 3 folds improve, 2
worsen. Respondent-clustered bootstrap (2000 resamples): 95% CI
**[-0.000689, 0.001660] -- crosses zero.** A re-optimized (rather than fixed)
blend weight diagnostic pushes the point estimate slightly higher
(0.000523) but is explicitly flagged in the write-up as optimistic since the
weight was chosen on the same evaluated data -- doesn't change the
conclusion. The joint Price+inside version of the same interaction is
similar but slightly weaker (+0.000452 blend gain); the inside-only version
is negligible (+0.000053).

**Segment-specific mileage-x-price slopes: decisively harmful.** CV blend
gain -0.002640, bootstrap CI [-0.005229, -0.000563] -- excludes zero on the
harmful side, worse in 4 of 5 folds including a large fold-4 regression.
Checked the fold-level coefficients specifically to rule out a repeat of the
earlier categorical-model coefficient explosion: magnitudes stayed moderate,
so this is ordinary high-variance overfitting from estimating several
subgroup-specific slopes on modest per-segment samples, not the same failure
mode as before -- a materially different (and less alarming, but still
negative) way to fail.

**Decision: not adopted, no submission made.** This is the most interesting
single result of the whole modeling push -- a coefficient that is genuinely
stable in sign and magnitude across every fold, consistent with a real (if
weak) interaction where mileage's effect on price sensitivity depends on
income. But stability of the *parameter* is not the same as a confirmed
*predictive* gain, and the bootstrap CI on the actual score improvement
crosses zero. Per the bar established after the ensemble-candidate review, a
positive point estimate with a CI that merely crosses zero does not clear the
threshold for a submission slot. Flagged as a clue worth revisiting only if
independent evidence appears (e.g. from the report's residual analysis or a
teammate's model), not as a validated improvement. `ensemble_v11` remains the
best and current submission.

**Where this leaves the modeling search.** Three consecutive, independently-
verified rounds of genuinely new ideas -- a 4-way ranking/regularized-logit
ensemble, an actual covariate-shift refit plus attribute-rank features, and
now continuous higher-order interactions -- have each returned either a null
result or a gain too small to distinguish from noise. Combined with the
earlier multi-angle diagnostic (excellent calibration, no exploitable
subgroup, xgboost unable to out-predict the logit), this is now a strong,
repeatedly-tested case that `ensemble_v11` (CV 1.145, public 1.202) is at or
very near the practical ceiling for legitimate, generalizable modeling on
this dataset, given the ~5 days remaining before the competition closes
(2026-08-01). Further effort is better spent on the report than on additional
modeling rounds unless a genuinely new structural idea surfaces.

## 2026-07-27: One more exhaustive pass -- noise-floor check, post-hoc calibration, and seed-bagging

Explicitly pushed to make sure "near the practical ceiling" wasn't premature
convergence rather than a genuinely exhausted search. Two sanity/diagnostic
checks first, since a real audit starts by questioning whether the target
(beating the leaderboard leaders) is even the right thing to chase, then one
more concrete technique tried directly (not via Codex, using cached OOF
artifacts already on disk).

**Noise-floor reality check.** The gap to the reported leaders (1.187, 1.190
vs. our 1.202 -- 0.015 and 0.012) is *smaller* than this project's own
established public-LB sampling noise floor: the respondent-clustered
simulation (`R/public_sample_noise_clustered.R`, logged 2026-07-27 earlier)
found 64.6% of same-model draw-pairs differ by >=0.015 from sampling luck
alone. This doesn't mean there's nothing left to find -- it means the
leaderboard gap itself is not strong evidence of a missing lever, and only
CV+bootstrap-confirmed results should move the model, exactly the discipline
already in place.

**Post-hoc calibration / temperature scaling against the target-weighted loss
-- clean no.** If the existing model were specifically overconfident on the
income-shifted, test-like respondents (as opposed to just having a coefficient
gap), flattening its predictions post-hoc should improve the already-built
target-weighted evaluation proxy (density-ratio weights from the covariate-
shift refit round) even without retraining. Swept temperature scaling
(0.75-1.35) and shrinkage toward both uniform and the global empirical
Ch1-Ch4 shares (0-20%) directly on the existing `ensemble_v11` OOF
predictions -- no refitting, no leakage risk, pure post-processing. Result:
**identity (no adjustment) is optimal on both the ordinary loss AND the
target-weighted loss**; every step away from it makes both worse
monotonically. This is a clean, decisive negative that closes off post-hoc
recalibration as a lever entirely, and complements the earlier failed
reweighted-refit finding: it shows the model's *confidence*, not just its
*coefficients*, is already close to what a target-weighted objective would
want, at least along the axes tested.

**Seed-bagging xgboost -- a real effect that doesn't reach the ensemble.**
Averaged the exact `cv_ensemble_v10.R` xgboost config (eta=0.1, depth=4,
subsample/colsample=0.8, nrounds=73) across 20 random seeds within each
canonical CV fold (`R/xgb_seed_bagging.R`) -- pure variance reduction, no new
model complexity, directly testing the one mechanism already confirmed to
work here (diverse-model blending reduces the generalization gap). Two
distinct findings:
- **The bagged xgboost alone is genuinely, statistically better** than the
  single-seed version: 1.178668 -> 1.176836, respondent-bootstrap 95% CI
  [0.000336, 0.003297] -- excludes zero on the positive side, 99.3% win rate.
  Real, confirmed, textbook bagging working as expected. Learning curve
  (1-20 seeds) shows the expected diminishing-returns shape, plateauing
  around 8-14 seeds.
- **But blended into ensemble_v11 at the same fixed 0.80/0.20 weight, the
  improvement almost entirely disappears**: 1.145094 -> 1.145087
  (+0.000008), bootstrap CI [-0.000294, 0.000302] -- centered on zero.
  Because xgboost only carries 20% of the blend weight, and the blend with
  mlogit was already absorbing most of xgboost's individual noise, making the
  weak component even less noisy barely moves the thing that's actually
  submitted.

**Implication, not yet tested:** bagging the *dominant* (80%-weight) mlogit
component via bootstrap-resampled respondents is the more promising version
of this idea, since any variance reduction there would propagate almost
directly into the blend. Not attempted in this pass -- correctly resampling
respondents with replacement for a conditional logit requires relabeling
duplicate-sampled respondents' `Case`/`chid` to avoid ID collisions in the
panel structure, a real implementation risk not worth rushing. Flagged as a
follow-up for Codex alongside two other genuinely new leads (data-driven
interaction discovery via xgboost SHAP values, since every interaction tested
all session was a human hypothesis rather than data-nominated; and partial
pooling / ridge-shrunk segment-specific slopes via the existing glmnet-Cox
penalized-likelihood machinery, targeting the exact failure mode the
segment-slope experiment just demonstrated -- full separate slopes overfit,
full pooling is the current baseline, nobody has tried the shrinkage middle
ground).

**Net effect on the conclusion:** unchanged. `ensemble_v11` remains the best
and current submission. But this pass adds two more independently-verified
negative/inconsequential results (post-hoc calibration, xgboost seed-bagging)
to the pile, and converts "we've stopped finding things" from possible
premature convergence into "we checked the two most obvious remaining classes
of technique (recalibration, variance reduction via bagging) and both are
genuinely exhausted for the model in its current form" -- a stronger claim,
honestly earned rather than assumed.

## 2026-07-27: Final round -- mlogit bootstrap-bagging, partial pooling, and SHAP-guided interactions

Third and final Codex round of the day (branch `codex-bagging-pooling-shap`,
commit `ac6e844`, merged into `zhenhao`), bundling the three leads queued after
the noise-floor/calibration/seed-bagging audit above. Independently reviewed
all three scripts line by line and cross-checked every reported number against
the raw generated CSVs in `data_processed/codex_final_round/` -- exact match
throughout, same clean track record as every prior round.

**1. Bootstrap-bagging the dominant m8trpg component -- bagging genuinely
hurts here.** Following directly from the xgboost seed-bagging result (real
effect, but on the wrong/minority-weight component), this bagged the
*dominant* 80%-weight mlogit component instead: 15 bootstrap resamples of
respondents per canonical CV fold (`R/codex_mlogit_bagging.R`), each
resample relabeling duplicated respondents with a synthetic `Case`/`No`/
`chid` to avoid ID collisions in `dfidx` -- verified this relabeling is
correct and defensively asserted (unique `chid` per resampled-respondent-
task, exactly 19 complete 4-alternative tasks per synthetic respondent; all
75 bootstrap fits across 15 bags x 5 folds succeeded on the first attempt).
Result: **bagging makes m8trpg WORSE**, not better -- 15-bag average
1.147764 vs single-fit 1.147021 (-0.000743); blended, 1.145658 vs
ensemble_v11's 1.145094 (-0.000563). Every point on the 1-to-15-bag learning
curve is on the harmful side, and 15 bags is the *least* harmful count tested
(not a cherry-picked stopping point) -- bootstrap CI [-0.001611, 0.000451]
for the blend crosses zero but the direction is consistently negative.
Interpretation: bagging reduces variance most for high-variance, unstable,
greedy learners like decision trees. A conditional-logit MLE with ~85
parameters on ~907 respondents is already a smooth, comparatively low-
variance estimator; resampling respondents with duplication injects
finite-sample coefficient noise/bias that outweighs any averaging benefit.
A genuinely useful, somewhat counterintuitive result: bagging is not a
universal remedy, and applying it to an already-stable parametric estimator
can actively hurt.

**2. Partial pooling for segment slopes -- independently confirms full
pooling is correct.** Extended the glmnet stratified-Cox equivalence with
the *full* m8trpg design as unpenalized core (a more direct comparison than
the earlier standalone-Cox version) plus new penalized candidates
(`Price x segment x z(mileage)`, `Price x segment x z(income)`, both
jointly), letting nested respondent-grouped CV pick both the elastic-net
`alpha` (0=ridge to 1=LASSO) and `lambda`. All three candidate sets selected
ridge (alpha=0, by lowest inner CV deviance) but the chosen penalty strength
drove every new coefficient to ~1e-40 -- genuinely, numerically zero, not a
small residual effect. Held-out loss (1.159721) matches m8trpg (1.159681) up
to optimizer-level noise. No candidate passed the screen. This is a
materially useful negative: it corroborates, via a *completely different*
estimation method (penalized Cox-equivalent likelihood vs. raw mlogit MLE),
the same conclusion as the earlier fully-unpooled segment-slope experiment --
there is no exploitable segment-specific mileage/income heterogeneity beyond
what the continuous covariate interactions already capture, at any nonzero
magnitude the data will support. Two independent methods agreeing is
stronger evidence than either alone.

**3. SHAP-guided interaction discovery -- finds real structure, still too
weak.** The first data-driven (not human-hypothesis-driven) interaction
search of the project: exact multiclass SHAP interaction values from the
reference wide-format xgboost (fit only on screen-training respondents,
`nthread=1` for determinism), aggregated over 100 sampled respondents in
memory-bounded batches, restricted to genuinely continuous/ordinal
covariates (categorical codes like segment/region correctly excluded),
deduplicated across coarse/fine encodings of the same concept, and filtered
against the 3 already-tested pairs (income x age/mileage/night, cross-checked
against `submissions_log.csv`). Of the top 8 untried pairs by SHAP rank, 4
beat the single-split screen (age x mileage, education x mileage, mileage x
urbanicity, age x gender) and were CV-confirmed. Best candidate, mileage x
urbanicity, blended: 1.144976 vs ensemble_v11's 1.145094 (+0.000118),
bootstrap CI [-0.001228, 0.001431] -- crosses zero, the smallest and least
confident positive estimate of the whole day. The other three were net
negative in the blend. Confirms SHAP successfully nominates genuine
nonlinear structure (unlike a random guess, it consistently surfaced
candidates that beat the screen) but the dataset's remaining nonlinear
signal is uniformly too weak to distinguish from noise, whether the
interaction is hypothesis-driven or data-nominated.

**Decision: none adopted, no submission made.** `ensemble_v11` remains the
best and current submission. This closes out the deepest single-day modeling
push of the project -- six independently-verified experiments in this final
stretch alone (noise-floor check, post-hoc calibration, xgboost seed-bagging,
mlogit bootstrap-bagging, partial pooling, SHAP-guided interactions), on top
of the four earlier Codex rounds (4-way ensemble, shift-refit + attribute
ranks, continuous triple interactions). Every genuinely new mechanism tried --
model-family diversity, distribution-shift correction, choice-set rank
features, higher-order interactions (both hypothesis- and data-driven),
post-hoc recalibration, and bagging in both directions -- has now been tested
to the same CV + respondent-bootstrap standard, and none clears the bar. This
is as close to an exhaustive search as the remaining time before the
competition closes (2026-08-01) reasonably allows.

## 2026-07-27: A behavioral-modeling round finally clears the bar -- design history, RRM, and a neural ensemble member

Pushed further after the "practical ceiling" conclusion above, asking Codex for
three genuinely different angles: (1) history/state-dependence effects beyond
the existing linear task-fatigue trend, (2) Random Regret Minimization, a
different behavioral paradigm from utility maximization, and (3) a non-tree
model for real ensemble diversity, since every flexible model tried all day
has been tree-based. Branch `codex-history-rrm-mlp`, commit `f7bc47c`, merged
into `zhenhao`. Reviewed all three scripts and the dedicated high-precision
reconstruction scripts line by line, and cross-checked every number against
the raw generated CSVs in `data_processed/codex_behavioral_round/` -- exact
match throughout.

**1. Design-exposure history -- corrected before implementation, still null
after multiplicity correction.** The original framing (a respondent's actual
*previous choices* predicting their next one) has a fatal flaw caught before
Codex built it: a test respondent's full 19-task sequence is unlabeled
*simultaneously* -- there is no point where past choices are revealed before
future ones need predicting, so a feature built from observed choice history
is fundamentally uncomputable at test time, even though it would look
perfectly fine in CV (training labels exist, so it would silently appear to
work there). The corrected version uses only the *design* sequence -- which
alternatives were shown, not which were chosen -- fully observable for both
train and test. Price-reference variants failed the screen; the best
surviving specification (attribute-exposure novelty/familiarity/similarity)
has consistent-sign coefficients across all 5 folds and a fold-cross-fitted
blend gain of +0.000226573, ordinary bootstrap 95% CI [+0.0000078,
+0.0004471] technically excluding zero. But this was the best of 8 CV-tested
history variants, and after that selection is accounted for, both the 99% CI
and a Bonferroni family-wise 95% CI cross zero. Not adopted -- a real but
too-small-to-call effect once search is priced in.

**2. Random Regret Minimization -- promising screen, null honest CV, but a
clean implementation.** `apollo` (the standard R package for this) wasn't
installed, so the regret likelihood and its analytic gradient were
implemented directly and verified against centered finite differences on 12
random parameters before trusting anything (max discrepancy <5e-10 -- the
custom math is correct). Single-split screen was promising (+0.000985 gain),
but 5-fold CV reverses it for both continuous- and factor-coded price
variants (honest fold-cross-fitted gains -0.000122 and -0.000226, both CIs on
the negative side of zero). RRM is a respectable standalone competitor to
m8trpg (component CV 1.148-1.150) but doesn't add ensemble diversity beyond
what the existing context/price-gap features already capture -- a different
theoretical lens arriving at essentially the same information.

**3. A neural net ensemble member -- the first result all day to clear the
bar.** `keras`/`tensorflow`/`torch` weren't available, so a small
single-hidden-layer softmax net was built with `nnet` (base R). Screened 9
size/decay configurations (3 seeds each); confirmed via a diagnostic
600-iteration rerun that 200 iterations is the right early-stopping point
(more iterations improved training fit but made validation worse -- textbook
overfitting, caught before it could contaminate the CV run). Fold-cross-fitted
blend against the fixed (not re-optimized) `ensemble_v11`: **1.143789442 vs
1.145094213, gain 0.001304771.** A dedicated high-precision script
(`R/codex_mlp_precision.R`) reconstructs this from saved OOF predictions with
hard-asserted baseline checks and runs a 100,000-replicate respondent
bootstrap: **ordinary 95% CI [+0.000092, +0.002513] -- excludes zero**, 98.24%
win rate. This is the only result all session to clear that bar. It does not
survive stricter scrutiny: the 99% CI and a 9-configuration Bonferroni-adjusted
95% CI (correctly accounting for the fact that 9 architectures were screened
before this one was picked) both cross zero. An extended 5-family ensemble
(adding rank-xgboost, retuned xgboost, and glmnet-Cox alongside the MLP)
reached 1.143129, but its incremental gain over the simple 2-way blend was
only 0.000661 with a CI crossing zero -- the extra complexity isn't justified.

Independently re-verified beyond the headline numbers: fold-cross-fitted
weight selection has no leakage (each fold's blend weight chosen using only
the other 4 folds' data); confirmed via elapsed-time cross-checking across
every saved fit (200-iteration fits ran ~37-49s, the 600-iteration diagnostic
rerun took ~105-118s, and every CV-stage fit matches the 200-iteration
timing profile) that the CV run used the intended 200-iteration cap, not the
script's unrelated 600-iteration default for a different stage -- and the
actual submission-file generator hardcodes `max_iterations = 200L` explicitly,
removing any ambiguity for what would actually be deployed. Also ran the same
outlier-sensitivity check used on the triple-interaction candidate: the MLP
blend correlates 0.9965 with `ensemble_v11`'s predictions, with a max
deviation of 0.07 (vs. 0.61 for the triple-interaction candidate) and zero
test rows showing a swing >0.15 in any alternative -- a bounded softmax output
doesn't have the unbounded-product blowup risk that made the triple
interaction sensitive to extreme-income outliers. Structurally, this is the
safest candidate produced all day.

**Decision: CV-confirmed candidate, not yet submitted.** This clears the
project's pre-stated ordinary-CI submission bar -- the first and only lead all
day to do so across five full rounds of testing. The honest caveat (stricter
multiplicity-adjusted intervals cross zero) means it should be described as a
credible, well-verified candidate rather than a guaranteed improvement.
Recommended as the top-priority candidate for the next available Kaggle
submission slot, ahead of the triple-interaction and 4-way-ensemble candidates
already prepared, given it is the only one to clear the baseline bar and the
most structurally robust to outlier respondents. `submission_codex_mlp_v12_candidate.csv`
is generated and ready; `ensemble_v11` remains the officially adopted model
until a submission confirms or refutes this candidate.

## 2026-07-28: MLP seed-bagging follow-up -- point estimate improves, interval widens

Direct follow-up on the MLP candidate above, testing whether the same
mechanism that helped xgboost seed-bagging (variance reduction on a
*minority-weight* ensemble component) also helps here, since the MLP sits in
the same structural role (small blend weight) that xgboost did. Branch
`codex-mlp-seed-bagging`, commit `066ac0b`, merged into `zhenhao`. Increased
the MLP's internal seed-averaging from 5 to 20, holding everything else
frozen (architecture, 200-iteration cap, canonical folds, fold-cross-fitted
weight selection). Independently verified the seed formula
(`4821 + fold*100 + seed_index - 1`) makes the first 5 of the 20 seeds per
fold mathematically identical to the original candidate's seeds -- confirmed
by the cumulative 5-seed OOF reproducing the original to 2.22e-16, a true
apples-to-apples extension rather than an independent re-randomization.
Cross-checked every number in the write-up against the raw generated CSVs --
exact match.

**Result: instructive, not simply negative.** The MLP component alone
improved substantially (1.190543 -> 1.168530 at 20 seeds, nearly monotonic
along the learning curve). But the ensemble's respondent-bootstrap interval
**widened rather than tightened** (95% CI width grew from 0.002421 at 5 seeds
to 0.003339 at 20), and the direct paired comparison of 20-vs-5-seed blends
has a 95% CI of [-0.000122, +0.001416] -- crossing zero, meaning 20 seeds is
not established as better than 5.

**The mechanism is verified, not just asserted.** As the MLP component gets
better, the fold-cross-fitted weight selection (correctly, since it's
choosing the loss-minimizing weight on held-out data) gives it MORE blend
weight: 0.13-0.17 at 5 seeds rises to 0.20-0.26 at 20 seeds. More weight on a
component that still has some idiosyncratic respondent-level error amplifies
both its average benefit to the ensemble AND its contribution to the
ensemble's respondent-to-respondent variance -- a real, coherent statistical
trade-off (a "stronger but more polarizing" component), not a computational
error. Confirmed by checking the actual fold weights at each seed count,
which climb steadily as claimed.

A methodologically important detail handled correctly: the learning curve
shows 15 seeds gives the single best point estimate (crossfit 1.143073,
slightly better than 20 seeds' 1.143139) -- but this was correctly **not**
promoted, since it's only visible after inspecting the full 1-20 curve
post hoc, exactly the kind of cherry-picking-after-the-fact this project has
guarded against with every multiplicity correction applied all session.

**Decision: no new candidate generated.** The original 5-seed
`submission_codex_mlp_v12_candidate.csv` remains the recommended submission
candidate -- more seed-averaging does not make it clearly better by the
project's own standard, despite improving the point estimate. `ensemble_v11`
remains officially adopted; the MLP candidate remains queued as the
top-priority submission for the next available slot.

## 2026-07-28: MLP candidate submitted -- new best model, first public gain since ensemble_v11

`submission_codex_mlp_v12_candidate.csv` was submitted to Kaggle. **Public
score: 1.201**, beating `ensemble_v11`'s standing 1.202 -- the first public
leaderboard improvement of the whole project since ensemble_v11 became the
best model. **This is now the new best model, both CV (1.143789) and
public.**

**Context for the result.** The CV gain over ensemble_v11 was 0.001305, with
a bootstrap 95% CI of [+0.000092, +0.002513] -- excluding zero at the
project's ordinary bar, but not at the stricter 99%/Bonferroni-adjusted
levels (both of which crossed zero). The actual observed public improvement
was smaller (0.001) than the CV point estimate (0.0013), but critically in
the SAME direction, not a reversal -- consistent with a genuine, if modest,
real effect rather than the CV signal being pure noise. The resulting
CV-to-public gap (0.057211) matches the project's established ~0.057
pattern for ensemble-class models almost exactly (ensemble_v9: 0.052,
ensemble_v11: 0.057), rather than introducing a new anomaly -- another point
in favor of this being a genuine, well-behaved improvement rather than a
lucky/unlucky draw of the public sample.

**Why this one, and not the other two prepared candidates.** Of the three
candidates prepared this session (triple-interaction mlogit, 4-way ensemble,
MLP blend), the MLP was the only one whose CV bootstrap CI cleared the
ordinary 95% bar at all, and it was also the structurally safest (no
outlier-sensitivity risk analogous to the triple-interaction candidate's
extreme-income sensitivity). The result validates that prioritization: a
real, if modest, gain, in the model that was both the best-supported and the
safest of the three.

**Updated project state.** `submission_codex_mlp_v12_candidate.csv`
(ensemble_v11 + 15% five-seed MLP) is now the officially adopted best model,
public 1.201 / CV 1.143789. `submission_triple_income_miles.csv` and
`submission_ensemble_v12_4way.csv` remain queued for future submission
slots, both still testing genuinely open questions (the income x mileage
interaction; the 4-way ranking/regularized-logit ensemble) independent of
this result.

## 2026-07-28: Combining the triple interaction with the MLP -- a new best CV number, mixed significance

Pushed on the explicit goal of clearing public 1.2 (other teams reportedly at
1.186). The triple-interaction mlogit and the MLP were each validated
independently this session but never blended together -- a natural
combination given they operate through different mechanisms (a utility-
specification refinement vs. a genuinely different function class for
ensemble diversity). Tested this directly (own analysis, not yet a
dedicated Codex round) using already-cached OOF predictions -- no refitting
needed for the CV comparison: `data_processed/codex_triples/triple_oof.rds`
(triple-interaction mlogit), `data_processed/codex_behavioral_round/mlp_oof.rds`
(MLP), and `data_processed/oof_ensemble_v10.rds` (xgboost). All three
verified to reproduce their already-confirmed baseline losses before
combining anything.

**Result: essentially ties the session's best-ever CV number, but doesn't
clearly beat the model actually in production.** Honest fold-cross-fitted
3-way blend (mlogit/xgboost/MLP weight chosen per fold using only the other
4 folds): **1.143328** -- almost identical to the earlier 5-family
ensemble's 1.143129 (which itself wasn't significantly better than the
simple MLP blend). Versus the CURRENT BEST (v11+MLP, 1.143789): gain
+0.000462, respondent bootstrap 95% CI **[-0.000754, +0.001642] -- crosses
zero**, not confirmed as an improvement over what's actually deployed.
Versus plain v11 (1.145094): gain +0.001767, CI **[+0.000054, +0.003420] --
excludes zero**, though barely.

**What this is actually worth, empirically.** Ran a paired public-LB-sized
simulation (same respondent-clustered methodology as
`R/public_sample_noise_clustered.R`, but comparing this candidate against
the current best on the SAME simulated draw rather than independent draws,
and anchored to the current best's REAL observed public score of 1.201
rather than a hypothetical) -- implies a 95% range of **[1.1979, 1.2033]**
for this candidate's public score, with a **63.6% chance of beating the
current best** on the same draw. A real lean toward improvement and a
genuine, non-trivial chance of landing below 1.2, but not a lock.

**Known caveat, re-confirmed for this new combination.** The triple
interaction's unbounded `Price x z(income) x z(mileage)` product is still
sensitive to the same extreme-income respondent identified earlier (test
`No 22637`, income 26.9 SDs above the training mean): 48 of 4997 test rows
show a >0.15 probability swing versus the current best, max deviation 0.52.
Since test has proportionally more such extreme respondents than training
(3 of 263 vs. 1 of 1135), the real-world public-LB variance for this
specific candidate could exceed what the training-respondent-based
simulation suggests.

**Submission prepared.** `R/submit_triple_mlp_v13.R` fits the
triple-interaction mlogit and xgboost fresh on the full training data, and
reuses the ALREADY-FIT MLP full-data test predictions from
`data_processed/codex_behavioral_round/mlp_full_test_candidate.rds` (no MLP
refit) -- blended at 0.732/0.112/0.156, the average of the fold-cross-fitted
weights, the same principle used for the current best's 15% MLP weight.
`submission_triple_mlp_v13.csv` generated and validated (4997 rows, correct
`No` order, row sums to 1, no NAs) but **not yet submitted**. Recommended as
the next candidate to test -- best CV number of anything not yet submitted,
and the only queued candidate with a specific mechanism (stacking two
independently-real effects) rather than just a single untested lever.

## 2026-07-28: Teammate Imelda's mnl+xgb -- honest validation, still a large surprise gap

Imelda's `submission_mnl_xgb.csv` (branch `imelda`,
`notebooks/experiments/ensemble_mnl_xgb.Rmd`) was submitted: **public 1.255**.
Reviewed her notebook to understand the model before logging it, the same
standard applied to Clarence's and Zeening's submissions.

**Her validation methodology is actually sound**, unlike Clarence's
(task-based split) or Zeening's (fully row-level-random split): 20
independent 80/20 splits sampling unique `Case` values, properly
respondent-grouped. Her own log records the ensemble's internal mean as
**1.18616 (sd 0.01424)**, beating her standalone MNL (1.20174) and xgboost
(1.18629) components as expected -- a believable, honestly-obtained number,
not an artifact of leakage.

**The gap anyway: 0.06884 -- the largest of any honestly-validated model in
this project**, bigger than this project's own standalone-mlogit gap
(0.066), which was itself the largest seen before now. This is a genuinely
interesting result specifically *because* her validation doesn't have an
obvious flaw -- it's a different kind of evidence than the Clarence/Zeening
cases (where a bad split fully explains the gap).

**Plausible contributing factors** (structural differences from this
project's models, not confirmed causes -- her code wasn't executed in this
environment, which uses hardcoded Mac paths):
- **Zero respondent-covariate interactions of any kind.** Her formula is
  `Choice ~ attrs + Price - 1`, no income/segment/age/etc. terms at all. This
  project's own progression found covariate interactions to be the single
  biggest source of legitimate, generalizable heterogeneity gain (mod6).
  Their complete absence here is the most obvious structural difference.
- **Only 6 of 19 attributes are factor-coded** (`NS/BU/FP/SC/MA/LB`, with
  rare levels collapsed to the modal); the other 13 enter as linear/
  continuous. This project confirmed early that factor-coding all attribute
  levels beats treating them as linear (mod1 -> mod2b) -- her spec only
  captures part of that gain.
- **Her xgboost component is the same binary-choice-with-renormalization
  architecture this project tested and found null** (a real ranking
  objective was needed to actually teach the within-task comparison
  structure) -- likely leaves real signal unused, capping the ensemble's
  ceiling regardless of the gap question.

None of these are proven to cause the specific 0.069 gap size, but they are
real, checkable differences from every model in this project's own lineage,
and they point the same direction: a model with the LEAST covariate-based
flexibility showing the LARGEST properly-validated gap is consistent with
(though doesn't prove) the confirmed income-shift finding being at least
part of the story -- a model that captures none of the transferable,
covariate-driven heterogeneity has less to fall back on when the population
shifts. Worth flagging to Imelda, and worth citing in the report's
generalization-gap section as independent evidence that the CV-to-public
gap on this dataset isn't just a symptom of any one team's validation
mistakes.

## 2026-07-28: Deep learning, full stacking, and LightGBM -- best-ever CV, still can't reach 1.186

Explicit push to clear public 1.186 (needed for a good module grade, per
teammate report of other groups' scores), not just 1.2. Three directions,
branch `codex-deep-stack-boost`, commit `8510b01`, merged into `zhenhao`.
Independently reviewed all four new scripts and cross-checked every number
against the raw generated CSVs in `data_processed/codex_deep_stack/` --
exact match throughout, including verifying the nested log-pool's inner/
outer fold structure has no leakage, the arithmetic blend's analytic BFGS
gradient is a correct softmax-parameterization derivation, and the deep
MLP's architecture was frozen from the single-split screen strictly before
the 5-fold CV loop.

**1. A real deep-learning framework is now available.** R `torch` 0.17.0
installed successfully -- the earlier "unavailable" finding was an
environment-setup gap, not a permanent limitation. A genuine 2-layer
(128/64 unit) dropout MLP was screened (5 configs, architecture frozen
before CV) and contains real signal: alone it scores 1.193140, and replacing
the shallow MLP with it in `ensemble_v11` gives a bootstrap-CI-excluding-zero
gain over plain v11 (+0.001948, CI [0.000386, 0.003513]). But it does **not**
clear the bar that actually matters -- improving over the ALREADY-SUBMITTED
shallow-MLP candidate. Both the incremental-add (+0.000620) and joint-blend
(+0.000759) comparisons have bootstrap CIs crossing zero
([-0.000310, 0.001552] and [-0.000188, 0.001714]).

**2. Full nested stacking across every diverse component -- learned
combiners still don't beat simple averaging.** With 6 (or 7, including the
new deep MLP) genuinely diverse cached OOF sources now available (mlogit,
2 xgboost variants, glmnet-Cox, shallow MLP, triple-interaction mlogit, deep
MLP), retested whether a properly nested, learned meta-model could extract
more than arithmetic blending -- this had only been tried once before, with
just 2 components, and found null. Same conclusion holds with far more
diversity: a nested ridge-regularized log-linear opinion pool (lambda
selected via inner folds, boundary-checked from both directions to confirm
0.01 is a genuine optimum, not an artifact of the tested range) reaches at
best 1.143155 (+0.000634, CI crossing zero) and *exactly ties* the current
best for the 7-component pool (-0.000002). A shallow xgboost meta-model is
decisively harmful in both pools (~1.152, clearly worse). Two attempts,
two component-diversity levels, same answer: this ensemble has reached
what a simple weighted average can extract: a learned combiner adds
nothing.

**3. The best CV number of the entire project -- but it doesn't clear the
bar.** The 8-component arithmetic blend (fold-cross-fitted weights: triple-
interaction mlogit 48%, deep MLP 13%, glmnet-Cox 13%, rank:ndcg xgboost 9%,
shallow MLP 8%, original mlogit 6%, retuned xgboost 2%, original xgboost
0.5%) reaches **1.142112** -- the best point estimate seen all session,
improving all 5 folds individually. High-precision bootstrap vs. the
current best: gain 0.001678, ordinary 95% CI **[0.0000466, 0.0032790] --
excludes zero, but only just.** The 99% CI and a 6-candidate
Bonferroni-adjusted CI both cross zero ([-0.000519, 0.003829]). Correctly
not submitted, per the project's own predeclared rule.

**4. LightGBM -- clean, fast negative.** Native categorical splitting
(different from xgboost's numeric treatment of attribute/covariate codes)
screened worse than the existing xgboost across all 3 regularized configs,
each receiving exactly zero blend weight -- correctly did not proceed to
CV, saving compute on an already-clear negative.

**The number that matters most for the 1.186 target.** Even taking the best
(statistically unconfirmed) result fully at face value, Codex calculated the
implied public-score movement: from 1.201 to roughly **1.199-1.200** -- not
1.186. This is worth sitting with: across the ENTIRE project's search --
ensembling, distribution-shift correction, choice-set rank features,
higher-order interactions (hypothesis- and data-driven), bagging in both
directions, Random Regret Minimization, a real deep-learning framework, and
now learned stacking across maximum available diversity -- no single
confirmed or unconfirmed gain has exceeded roughly 0.002 in CV terms.
Closing a 0.015 public gap would require something on the order of 10x any
single improvement found anywhere in this exhaustive, multi-technique
search. This doesn't prove 1.186 is impossible with a fundamentally
different approach, but it is strong evidence that it is not reachable via
further iteration on the modeling techniques already tried.

## 2026-07-28: CatBoost -- confirms, doesn't overturn, the LightGBM negative

One more follow-up (branch `codex-catboost`, commit `12676eb`, merged into
`zhenhao`): does CatBoost's native ordered-boosting categorical mechanism
(target statistics rather than pure splits) recover anything LightGBM's
split-based categorical treatment missed? Partly motivated by a prior year's
similar course project (methodology only, not copied -- their run dropped
respondent covariates and used a leakage-flawed split, so nothing there was
reusable, but it flagged CatBoost as worth testing properly).

Installation needed a workaround: this machine lacks Rtools, so a
from-source GitHub build would have been fragile; Codex instead used
CatBoost's official pre-built Windows release binary, which installed and
smoke-tested cleanly. Same feature set as xgboost/LightGBM
(`wide_feature_matrix()`, all attribute/price/covariate columns), with
attribute/price and categorical covariates declared as native CatBoost
categoricals.

**Result: clean, unambiguous negative, same conclusion as LightGBM.** 3 of 4
screened configs got exactly zero blend weight (component loss 1.212-1.222,
worse than every other tree-based component already logged this project).
The 4th technically selected a nonzero screen weight (2%, gain 0.0000364)
and was correctly advanced per the predeclared rule despite the negligible
size. In canonical 5-fold CV, with its tree count (681) frozen from the
screen *before* CV (no early-stopping using held-out fold labels -- verified
directly in the code), the fold-cross-fitted blend weight was **exactly
zero in all 5 folds**. The resulting blend is byte-identical to the current
best; the respondent bootstrap is a literal point mass at zero (SD 0, CI
[0,0]) -- the correct, deterministic consequence of every fold's selector
rejecting CatBoost outright, not a computational error.

**Interpretation.** Two independent tree-based frameworks with genuinely
different categorical-handling mechanisms (LightGBM's split-based treatment,
CatBoost's ordered target statistics) now agree: changing how the tree
learner handles categorical features is not the missing lever. The
project's established 1.142-1.144 CV ceiling holds. This closes the
tree-learner-diversity question cleanly -- consistent with, and reinforcing,
the broader conclusion that reaching public 1.186 is not achievable via
further modeling iteration on the techniques tried so far.

## 2026-07-28: Shared-utility exact-softmax models, scale heterogeneity, and yearind

Five directions from an external technical review, each fact-checked before
delegating rather than taken on faith (branch `codex-shared-utility`, commit
`3e1ca29`, merged into `zhenhao`). Two of the review's claims were checked and
one was dropped: a "Case-tail validation" idea (test directly -- no evidence
of a Case-order income trend in training data, correlation ~0.03, dropped
before it reached Codex) and a confirmed genuine gap (`yearind` has never
been tested as an interaction, verified by grep before delegating).

**1. A genuine shared-alternative-utility model on the EXACT choice
likelihood.** This closes a real, previously-unaddressed structural gap:
neither `multi:softprob` (no exchangeability constraint -- alternatives 1-3
get effectively separate learned splits despite being interchangeable
feature bundles) nor `rank:ndcg`/`rank:pairwise` (a shared per-alternative
scoring function, correctly, but trained on a ranking metric and only
calibrated to probabilities *after* fitting) directly optimizes the
four-way softmax cross-entropy the way `mlogit` does. Implemented via a
custom xgboost objective (one scoring function per alternative, `qid`
grouping, gradient = p_j - y_j, diagonally-dominant Hessian upper bound
matching xgboost's own multinomial approach) -- the analytic gradient was
verified against finite differences before trusting anything (max error
1.06e-10 against a 1e-7 tolerance). `clogitboost` (the R package suggested
as a lower-risk alternative) installed but failed structurally: its
componentwise spline learner needs >=4 unique x-values per term, and most
of this project's features (attribute levels, alt-position flags) are
binary/low-cardinality by nature -- not a fixable installation issue.

Cold-start (no base model), the shared-utility booster failed even the
single-split screen (component 1.218569 vs. v11's 1.160568) and correctly
never reached CV, per the established screen-first rule. This is a genuinely
informative negative: optimizing the *exact* choice likelihood is not by
itself sufficient to be competitive without m8trpg's extensive hand-built
interaction structure -- the loss function wasn't the bottleneck, the
features/specification were always doing the real work.

**2. The same objective as a residual on top of m8trpg** (m8trpg's utility
as a fixed `base_margin`, correction trained directly on the residual
softmax loss -- different from ensembling, since the correction sees the
base model's errors during its own fitting, not after). A small, real
effect: propagated through the full ensemble, 1.143789 -> 1.143714
(+0.0000759), bootstrap 95% CI [-0.0000839, +0.0002417] -- crosses zero.
Feature importance is dominated in every fold by `price_gap_min`/`price_gap_max`
-- this is a tiny refinement of the already-known price-context mechanism,
not a new source of signal.

**3. Continuous global scale heterogeneity** (a multiplicative `μ` on the
*entire* utility vector, not just the price sub-component like the earlier
discrete 2-class latent-class attempt) -- gradient-verified (max error
1.36e-10), tested at three ridge strengths. **Decisively harmful, not just
null**: all three have a 95% CI entirely below zero (least harmful,
ridge=0.01: [-0.002214, -0.000677]). A clean, confirmed negative.

**4. `yearind` interactions**: harmful (CV gain -0.001007 propagated, 4 of 5
folds worse). Closes the one genuinely untested covariate with a clean
answer: not useful.

**5. Test-like-respondent re-ranking** (no new training -- re-evaluating
existing candidates on the subset of training respondents most similar to
test, via the existing adversarial classifier): the 8-component arithmetic
blend and triple+MLP keep the same 1st/2nd ranking at every population
cutoff tested. On the top-30%-most-test-like slice specifically, the
8-component blend's edge over the submitted MLP blend actually strengthens
enough to exclude zero (CI [+0.000262, +0.007557]) -- corroborating evidence
its direction isn't an artifact of non-test-like respondents. Doesn't repair
the already-failed stricter multiplicity-adjusted interval, so the
no-submission decision on that candidate stands, but it's a genuinely
useful piece of corroborating evidence, not nothing.

## 2026-07-28: The questionnaire has ~299 recurring design versions -- confirmed directly, own analysis

The same external review proposed a specific, checkable hypothesis: that the
survey used a fixed pool of ~300 questionnaire "versions" (Sawtooth CBC's
documented default), assigned to respondents by sequential Case-number
cycling, which would explain the already-confirmed 98.5% test-design
recurrence in training. Tested this directly myself before sending anything
to Codex, since it's cheap and completely decisive either way
(`R/check_questionnaire_version_structure.R`).

**Method:** fingerprint every respondent's entire ordered 19-task sequence
(every attribute + price for all 4 alternatives, in task order -- pure
design information, no choices, so valid to compute across train+test
combined; 1398 total respondents).

**Result, split into a refutation and a confirmation:**

- **The specific sequential-cycling mechanism is refuted.** Tested every
  candidate period from 50 to 500 respondents: for none of them do
  same-(Case mod V) respondents actually share a fingerprint (purity
  exactly 0 at every tested period). Version assignment is not simple
  sequential rotation by Case number.
- **But the underlying structure is real, and stronger than the earlier
  design-fingerprint check suggested.** There are **exactly 299 unique
  full-sequence fingerprints among all 1398 train+test respondents** --
  matching Sawtooth's documented 300-version default almost exactly. 286 of
  299 versions recur (shared by more than one respondent, ~4.7 respondents
  per version on average), with train and test respondents mixed into the
  same version groups (Case-number gaps between same-fingerprint
  respondents range from 3 to 1131, essentially unstructured with respect
  to Case order -- consistent with random rather than sequential version
  assignment).

**Why this is stronger than the existing `design_cell_empirical_shrinkage`
null result, not a restatement of it.** That earlier attempt (2026-07-26)
grouped respondents by *individual task-position designs* (~296 groups,
~3.8 respondents each) -- 19 separate small-sample problems, one per task
position, and it correctly found too little data per cell to help. This
groups by the *entire 19-task sequence instead*: the same ~4-5 respondents
per version share **all 19** of their tasks, not just one, so a
version-level residual can pool roughly 4-5 respondents x 19 tasks ~= 80-95
data points per group, not ~3-4. The ~296-vs-299 near-match strongly
suggests the earlier per-task-position count was already an echo of this
same underlying ~299-version structure, just viewed one task at a time
without exploiting the full sequence.

**Implication.** This reopens the design-cell idea with a materially
different, better-powered version: a cross-fitted, heavily-shrunk
version-level residual/calibration (opt-out intercept, overall scale, or a
target-encoded residual by version, using only training-fold respondents
sharing a given version, applied to held-out respondents -- including test
-- who share that version). Cached as `data_processed/questionnaire_fingerprints.rds`
for reuse. This is now the single most promising untested lever, given it's
the only hypothesis from the recent external-review round that was
independently, empirically confirmed to exist in the data (not just
plausible) before any modeling was attempted on top of it.
