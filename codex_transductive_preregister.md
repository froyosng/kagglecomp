# Pre-registration: transductive covariate-shift correction (codex_transductive track)

Date: 2026-07-29. Author: independent parallel worker ("codex_transductive"), one of
four isolated hypothesis tracks running against the same project state
(`ensemble_v11+MLP` as current best). This document is committed to git BEFORE any
of the evaluation code in this file's plan is executed for real, so the commit
timestamp is the pre-registration timestamp. Only smoke tests on tiny synthetic/
subset data (to catch bugs, not to peek at results) are allowed before this commit.

## 1. Restating the current best model, in my own words

The current best model, `ensemble_v11+MLP`, is a three-way blend built in two
nested stages, exactly as implemented in `R/submit_ensemble_v11.R` (mlogit + xgb
stage) and `R/codex_mlp_precision.R` (the MLP blend stage):

1. **m8trpg**: a conditional logit (`mlogit`, no ASCs, factor-coded attribute
   levels, factor-coded price levels 2-12) fit on the 1135 training respondents'
   21,565 choice tasks. On top of the base attributes it adds: alt2/alt3 position
   dummies; Price x {income, age, miles, night} and inside x {income, age, miles,
   night, gender, urbanicity, education} interactions (respondent covariates,
   standardized using TRAINING-only mean/sd); Price x segment and inside x segment
   (6-level vehicle segment) interactions; Price x task-position and inside x
   task-position (survey-fatigue) interactions; Price x region and inside x region,
   Price x parking-situation and inside x parking-situation interactions; and
   choice-set-context terms (`is_cheapest`, `is_dearest`, `price_gap_min`,
   `price_gap_max`) capturing an alternative's price rank within its own task.
2. **xgboost** (`multi:softprob`, nrounds=73): a gradient-boosted tree model on the
   wide-format attribute/price/covariate columns, fit independently of the logit.
3. `ensemble_v11 = 0.80 * m8trpg + 0.20 * xgboost` (fixed weight, chosen once by
   5-fold CV, never re-optimized per candidate).
4. A small single-hidden-layer softmax net (`nnet`, 5-seed average, 200 iterations)
   is blended on top: `final = (1 - w_fold) * ensemble_v11 + w_fold * MLP`, with
   `w_fold` in [0.13, 0.17] chosen by **fold-cross-fitted** weight selection (each
   fold's blend weight is chosen using only the other 4 folds, so the reported CV
   number has no weight-selection leakage).

Canonical CV log loss (pooled out-of-fold, seed-4821 respondent folds): **1.143789**
(reproduced exactly below, see `data_processed/codex_transductive/current_best_reconstruction.rds`
and the console output of `R/codex_transductive_verify_baseline.R`, which matches the
officially-logged value `1.14378944178118` to the 1e-10 hard-asserted tolerance already
used in `R/codex_mlp_precision.R`). Public leaderboard: 1.201.

## 2. Why my approach is mechanistically different from the two rejected corrections

Both previously-rejected shift corrections operated on the **likelihood weight**
attached to each existing training respondent's contribution to the fit or to the
evaluation:

- **(a) Evaluation-only test-likeness reweighting** (`R/codex_test_like_reranking.R`
  et al.): re-weighted the OOF *evaluation* using density-ratio importance weights so
  the reported loss mimics test's covariate distribution. This never touched what
  was fitted, only how it was scored, and it made the reported loss look worse, not
  better — because a genuine shift correction can never be validated by an
  evaluation-only reweighting of an unchanged fit (the fit itself never adapted to
  the shift).
- **(b) Importance-weighted-likelihood refit** (`R/codex_shift_common.R` /
  `R/codex_importance_refit.R`, the covariate-shift-refit branch): converted the
  adversarial classifier's output into a density ratio per respondent and used
  `ratio^alpha` as `mlogit` case weights, i.e. **down-weighted training respondents
  who look train-like and up-weighted the (few) training respondents who happen to
  look test-like**. This is the textbook Shimodaira (2000) importance-weighted MLE.
  It was decisively harmful: every alpha > 0 made the target-weighted loss worse in
  all 5 folds, confirmed by a respondent-bootstrap CI entirely on the harmful side
  ([-0.002614, -0.000422] at the gentlest setting tested). The diagnosed cause: the
  weighting discards effective sample size (908 -> ~521 effective respondents at
  alpha=1) faster than it buys targeting benefit, because there are too few
  genuinely test-like respondents in the training set to reweight *towards* without
  destroying precision.

Both (a) and (b) share one mechanism: **every training respondent already in the
data keeps their same covariate values; only the SCALAR WEIGHT multiplying their
log-likelihood contribution (or their contribution to a post-hoc evaluation average)
changes.** No respondent's designed matrix row is altered, and no new information
enters the fit beyond a re-derived scalar per existing respondent.

My assigned hypothesis is different in kind, not degree: **directly incorporate the
test covariate distribution into the design matrix or the training set itself,
holding every existing training respondent's likelihood weight at exactly 1**. Two
candidates, both avoiding respondent reweighting entirely:

- **Candidate A (moment-matching / distributional feature recoding):** change WHAT
  VALUE is fed into the continuous-covariate interaction terms (income, age,
  mileage, night), not how much each respondent's row counts. Every training
  respondent keeps weight 1; the model still sees exactly 908 (fold) / 1135 (full)
  full-weight respondents.
- **Candidate B (confident self-training):** add NEW rows (from test, pseudo-labeled)
  to the training set, again at weight 1, alongside the unchanged, undiminished
  original training respondents. This increases effective sample size rather than
  shrinking it — the opposite mechanism from (b)'s reweighting, which is exactly why
  it is worth testing even though (b) failed: (b) failed via a sample-size-destroying
  mechanism that Candidate B does not share.

Both candidates are pre-registered below and will be evaluated independently against
the current best via the same respondent-grouped-CV + respondent-bootstrap standard
this project has used throughout (see `cleaning_log.md` sections "External review
round", "Covariate-shift refit and attribute-rank experiments", and the MLP
candidate's precision audit).

## 3. Candidate A: quantile-mapped moment matching on m8trpg's continuous covariates

**Mechanism.** For each of the four continuous respondent covariates used in
m8trpg's Price/inside interaction terms — `incomea`, `agea`, `milesa`, `nighta` —
replace the existing standardization
`z = (x - mean_train) / sd_train`
with a **quantile-mapped recoding**:

```
x_matched  = Q_test( F_source(x) )
z_matched  = ( x_matched - mean_test ) / sd_test
```

where `F_source` is the empirical CDF (linearly-interpolated, using `(rank - 0.5)/n`
plotting positions, clamped to `[1/(2n), 1 - 1/(2n)]` to avoid 0/1 boundary blow-ups)
fit on the **respondent-level** values of that covariate in the data actually used to
fit this particular model (the fold's ~908 training respondents for a CV fold, or all
1135 for a final/production fit), and `Q_test` is the empirical quantile function
(type-7, R's default `quantile()`) of that covariate's respondent-level values in
`test.csv` (263 respondents) -- fixed, the same reference distribution regardless of
CV fold, since `test.csv`'s covariates are static and fully observable. `mean_test`/
`sd_test` are test.csv's own respondent-level mean/sd of that covariate (fixed,
non-fold-dependent), used only to rescale the matched value onto a numerically
convenient scale for `mlogit`'s optimizer.

This is a genuinely nonlinear, monotonic recoding (not an affine shift/rescale): I
confirmed algebraically before implementing that a pure affine change to the existing
`z` (i.e. same shape, different center/scale) is exactly reproducible by mlogit's own
MLE refit whenever the design otherwise omits free "plain Price" and "plain inside"
main-effect terms (which m8trpg does), meaning a naive re-centering/re-scaling
would either be a complete no-op (pure rescale) or a roundabout, constrained way of
adding two terms that could be added directly (pure recenter) -- neither is a
meaningful test of the hypothesis. Quantile mapping is different in kind: it changes
the *shape* of each covariate's within-training distribution to match test's shape,
which cannot be undone or reproduced by any linear reparameterization of the existing
interaction coefficients. This is the operationalization that actually matches the
brief's wording: "rescale/recenter ... so their EFFECTIVE distribution in the fitted
design matches test's known marginal covariate distribution."

**What does NOT change:** attribute factor terms, price-level factor terms, d2/d3,
segment/task/region/ppark interactions, price-gap/is_cheapest/is_dearest terms, the
xgboost component (reused unchanged from `data_processed/oof_ensemble_v10.rds`), the
MLP component and its fold blend weights (reused unchanged from
`data_processed/codex_behavioral_round/`), and the 0.80/0.20 mlogit/xgb blend weight
(reused unchanged, not re-optimized). Only the m8trpg mlogit component's 8
interaction columns (`P_income, P_age, P_miles, P_night, In_income, In_age, In_miles,
In_night`) are recomputed with `z_matched` in place of `z`. This isolates the
covariate-recoding treatment as the single changed ingredient versus the current
best, exactly mirroring how every other candidate in this project's history (price-
history, triple-interactions, rank features, etc.) was isolated against a frozen
`ensemble_v11` before being judged.

**Validation design (respondent-grouped, no leakage):**
- Use the canonical `fold_of_case` from `data_processed/oof_ensemble_v10.rds` (seed
  4821, 227/227/227/227/227).
- For each fold k: fit `F_source` using ONLY that fold's ~908 training respondents
  (never the held-out 227, never test.csv's labels -- test.csv has no labels to
  leak). Apply the resulting `x_matched`/`z_matched` recoding to BOTH the fold's
  training rows (for fitting) and the held-out fold's rows (for prediction) using
  the SAME fold-specific `F_source` -- i.e. exactly the same "fit transform on
  train-fold only, apply to held-out" discipline already used for the existing
  scaler (`ctr, scl`) in `R/cv_ensemble_v10.R`.
- Refit the full m8trpg formula (unchanged formula string, only the 8 interaction
  columns' input values differ) via ordinary unweighted `mlogit` MLE on the fold's
  908 respondents.
- Predict on the held-out 227 respondents; store as this candidate's OOF block for
  that fold.
- After all 5 folds: build `oof_mlogit_qmatch` (n=1135 x 4), then
  `v11_qmatch = 0.80 * oof_mlogit_qmatch + 0.20 * oof_xgb` (oof_xgb unchanged,
  reused), then re-apply the existing per-fold MLP weights unchanged:
  `crossfit_qmatch[fold k rows] = (1 - w_k) * v11_qmatch[fold k rows] + w_k * MLP[fold k rows]`.
- Uses test.csv's covariates only (never test labels, which do not exist), and never
  touches leaderboard feedback.

## 4. Candidate B: confident self-training on m8trpg

**Mechanism.** For each fold k:
1. Fit the **unmodified** m8trpg (same formula and covariate coding as the current
   best, i.e. NOT combined with Candidate A -- kept orthogonal so the two hypotheses
   remain separately attributable) on that fold's ~908 training respondents. This
   reuses exactly the same fitting code as the existing CV harness
   (`R/cv_ensemble_v10.R`); refitting per fold is required (rather than reusing the
   saved OOF) because I additionally need the fitted model OBJECT to score
   `test.csv`.
2. Apply that fold's fitted model to `test.csv` (4,997 rows / 263 respondents;
   covariates only, no labels exist for test in this competition).
3. Select **confident test rows**: `max_j(predicted_prob_j) > 0.85` (fixed threshold,
   taken directly from the assigned hypothesis text, not tuned or swept -- a single
   pre-registered threshold to avoid a multiplicity problem like the ones this
   project has repeatedly had to correct for with Bonferroni/stricter intervals).
   Pseudo-label = argmax alternative, one-hot encoded as if it were the true
   `Ch1..Ch4`.
4. Augment the fold's training long-format data with these confident pseudo-labeled
   test rows (full 4-alternative choice-task blocks, all attributes/prices already
   genuinely observed in `test.csv`; only the "chosen" flag is synthetic) at
   likelihood weight 1, identical treatment to genuine rows. The standardization
   scaler for this candidate is computed from the fold's 908 REAL training
   respondents ONLY (never including test respondents, pseudo-labeled or not) --
   this keeps Candidate B orthogonal to Candidate A's covariate-recoding mechanism;
   only the row-set changes, not the feature encoding.
5. Refit m8trpg on (908 real respondents' full panels) union (confident pseudo-
   labeled test tasks) via ordinary unweighted MLE.
6. Predict on the SAME held-out 227 REAL train respondents used for Candidate A/the
   current best (never used anywhere in steps 1-5).
7. Store as this candidate's OOF block for that fold.
8. Blend exactly as Candidate A: `v11_selftrain = 0.80 * oof_mlogit_selftrain + 0.20
   * oof_xgb` (unchanged xgb), then the unchanged per-fold MLP weights.

**Anti-circularity safeguard, stated explicitly:** the pseudo-labels are manufactured
from a model that never saw the held-out 227 respondents' true labels, and the
augmented refit is evaluated ONLY on those same held-out 227 respondents' true,
genuine `Ch1-4` values -- never on the pseudo-labeled test rows' own manufactured
labels, and never on the leaderboard. Concretely: I will NOT report or use
`log_loss(pseudo_label, refit_prediction_on_those_same_test_rows)` as any part of the
verdict -- that number is expected to look artificially near-perfect (the model was
selected/thresholded precisely because it was confident on those exact rows, then
refit to include them), and reporting it would be circular and worthless, per the
task brief. I will demonstrate this circularity concretely as a stress test (Section
6) rather than merely asserting it.

## 5. Bootstrap and promotion rule (identical standard for both candidates)

For each candidate C in {qmatch, selftrain}, independently:
- `case_gain[i] = mean over case i's rows of ( row_logloss(current_best) -
  row_logloss(candidate_C) )`, for each of the 1135 training respondents (paired,
  same folds, same rows -- respondent-level variation cancels in the difference,
  matching this project's established bootstrap practice).
- Respondent-clustered bootstrap: resample the 1135 cases with replacement, 100,000
  replicates (matching `R/codex_mlp_precision.R`'s `bootstrap_case_means`, same
  function reused verbatim), compute the mean gain each replicate.
- Report: point estimate, ordinary 95% CI (2.5%/97.5% quantiles), win rate (share of
  replicates with gain > 0), and -- to be maximally conservative given this
  project's own established practice of not trusting a single pre-registered ordinary
  CI in isolation -- 99% CI as a secondary check.
- **Promotion bar (pre-registered, matching the task brief exactly):** the ordinary
  95% CI must exclude zero on the positive side (`lower_95 > 0`). A positive point
  estimate alone is NOT sufficient. If both candidates clear this bar, both are
  reported; if one clears and one does not, the failing one is reported as a null
  result, not silently dropped. If neither clears, that is reported as the honest
  verdict for this entire track.
- Since there are exactly 2 pre-registered candidates (not a wide screen), I will
  also report what a Bonferroni-adjusted 95% CI (family size 2, alpha/2 = 0.025)
  looks like for whichever candidate clears the ordinary bar, for transparency
  consistent with this project's practice elsewhere -- but per the task's explicit
  promotion rule, the ORDINARY 95% CI excluding zero is the binding bar, not the
  stricter one.

## 6. Planned anti-circularity stress test (Candidate B)

Before trusting Candidate B's honest CV numbers, I will deliberately compute and
report the circular, wrong number as a labeled negative-control demonstration: for
one fold, evaluate the fold's self-trained refit's log loss ON the confident
pseudo-labeled test rows it was just trained to reproduce, and show it is close to
0 / near-perfect (as expected for a model literally fit on the label it will be
"tested" against) -- then contrast this explicitly against the actual, honest
held-out-train-respondent evaluation used for the real verdict, to make unmistakable
in the written findings which number is real and which is the circularity trap being
avoided.

## 7. Smoke test plan (before the real run)

Before running the full 5-fold refit for either candidate (each fold's `mlogit` fit
on ~140 parameters / ~18-21k rows takes real wall-clock time), I will:
1. Run both feature-construction pipelines (quantile-mapping and self-training
   augmentation) on a small subset (e.g. 2 folds, or a random 100-respondent subset)
   to confirm no errors, sane output shapes, `rowSums(pred) == 1`, no `NA`/`NaN`.
2. Confirm the quantile-mapping function is monotonic and its output for a
   deliberately-constructed synthetic covariate vector matches hand-computed
   expected values.
3. Confirm the confident-row selection threshold produces a non-trivial (but
   plausibly small) number of rows on a single fold before committing to the full
   run, and sanity-check a handful of the selected rows' pseudo-labels against the
   raw predicted-probability vector by hand.
4. Only after these checks pass do I run the full 5-fold CV + 100,000-replicate
   bootstrap for both candidates and write up `codex_transductive_findings.md`.

## 8. Outputs

- Code: `R/codex_transductive_qmatch.R` (Candidate A), `R/codex_transductive_selftrain.R`
  (Candidate B), plus shared helpers reused from the verified baseline reconstruction
  already committed (`R/codex_transductive_verify_baseline.R`).
- Data artifacts: `data_processed/codex_transductive/*.rds` / `*.csv` (OOF matrices,
  bootstrap draws, fold-level diagnostics), gitignored raw predictions but summary
  CSVs committed for reproducibility review.
- Findings: `codex_transductive_findings.md`, written only after the real run,
  reporting both candidates' verdicts honestly, including a null result if that is
  what is found, per the task brief.

This file will be committed before any of the Section 3-6 code is run against the
full data (only the isolated data-availability check in
`R/codex_transductive_datacheck.R`, which touches no model fitting or evaluation
logic and could not influence the pre-registered method, was run beforehand to
confirm feasibility -- e.g. that there are no NA values in the four covariates used
for quantile mapping in either train or test).
