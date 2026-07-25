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
