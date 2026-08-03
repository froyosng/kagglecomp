# Segment-distribution shift: targeted opt-out-margin pruning -- pre-registration

Date: 2026-07-31. Route 2 (support/extrapolation audit) of the external
review's follow-up. A genuinely new structural fact, confirmed by direct
data inspection (not modeling): `segmentind`'s train/test distribution is
far more extreme than the already-known income shift.

## New structural fact (confirmed directly from `train.csv`/`test.csv`, not modeled)

Respondent-level segment counts:

| Segment | Train n (%) | Test n (%) |
|---|---|---|
| 1 | 128 (11.3%) | 57 (21.7%) |
| 2 | 383 (33.7%) | 9 (3.4%) |
| 3 | 49 (4.3%) | **84 (31.9%)** |
| 4 | 211 (18.6%) | 16 (6.1%) |
| 5 | 58 (5.1%) | **97 (36.9%)** |
| 6 | 306 (27.0%) | **0 (0.0%)** |

Segment 6 (27% of training respondents) has **zero** representation in
test. Segments 3 and 5, only 4.3% and 5.1% of training respondents
combined (107 of 1135), together make up **68.8% of test respondents**.
An adversarial-validation classifier confirms segment dominates every
other covariate as a train/test discriminator: segment alone reaches AUC
0.898 (in-sample; inflated somewhat by segment 6's quasi-complete
separation, since it has zero test rows to misclassify), vs. income alone
at AUC 0.654 (the previously-reported main driver, `cleaning_log.md`
2026-07-26). This has not been characterized as its own structural fact
before -- the earlier segment-sparsity note (`cleaning_log.md`, 2026-07-27,
"segment respondent counts checked first... two smallest are comparable in
size to the small K-means clusters flagged earlier as high-variance") was
about training-set sparsity as a reason for caution when *adding* new
segment-specific terms, not about the train/test distribution mismatch in
the terms *already deployed* in every submitted model since `mod8`.

## Coefficient audit (m8trpg refit on full training data)

| Term | Estimate | SE | p-value | Segment's test share |
|---|---|---|---|---|
| `P_seg2` | 0.0268 | 0.0120 | 0.026 | 3.4% |
| `P_seg3` | 0.1148 | 0.0171 | **<1e-10** | 31.9% |
| `P_seg4` | 0.0371 | 0.0130 | 0.004 | 6.1% |
| `P_seg5` | 0.1179 | 0.0164 | **<1e-12** | 36.9% |
| `P_seg6` | 0.0216 | 0.0124 | 0.082 | 0.0% |
| `In_seg2` | -0.2062 | 0.0822 | 0.012 | 3.4% |
| `In_seg3` | 0.0440 | 0.1429 | **0.758** | 31.9% |
| `In_seg4` | -0.1966 | 0.0900 | 0.029 | 6.1% |
| `In_seg5` | 0.1118 | 0.1341 | **0.404** | 36.9% |
| `In_seg6` | -0.3028 | 0.0857 | **0.0004** | 0.0% |

A striking, opposite-direction pattern: the **price-sensitivity** deviations
for segments 3/5 (`P_seg3`, `P_seg5`) are large, precise, and highly
significant -- there is no evidence these are unreliable. But the
**opt-out-margin** deviations for the same two segments (`In_seg3`,
`In_seg5`) are statistically indistinguishable from zero even with the
full 1,135-respondent training set (p=0.76, p=0.40) -- and these are
exactly the two segments whose weight in the loss function jumps from
~9% (in training, where CV is evaluated) to ~69% (in test, where the
actual score is computed). Conversely, the *other* three segments'
`In_seg` terms (2, 4, 6) are all individually significant, but govern
segments whose test share is *shrinking or vanishing* (3.4%, 6.1%, 0%).

## Hypothesis

`In_seg3` and `In_seg5` add opt-out-margin estimation noise that ordinary
respondent-grouped CV cannot penalize proportionately to its real cost,
because CV is drawn from the training segment distribution (~9% weight on
segments 3/5) while the actual scored test distribution weights the same
noisy terms at ~69%. Dropping these two specific, already-non-significant
terms should show a materially larger benefit when evaluated on a
test-like-reweighted subset of training respondents than on the ordinary
CV population -- a distinct, mechanically-grounded claim from a generic
"prune insignificant terms" pass (already tried once, `mod11`, and found
"essentially tied," a much weaker claim than what is tested here).

## Why this is not a duplicate

- Not `mod11`'s pruning (different terms: `P_income`, `P_night`,
  `In_income`, `In_age`, `In_miles`, `In_night`; segment terms were never
  touched, and that test never checked test-like-reweighted performance).
- Not the earlier segment-sparsity caution (about *adding* new
  segment-specific covariate slopes on top of the existing structure, not
  about *removing* an already-deployed, already-non-significant term).
- Not the already-rejected importance-weighted refit (that reweighted the
  *entire* likelihood by an estimated density ratio and decisively hurt,
  losing effective sample size project-wide). This targets exactly two
  already-weak parameters with a mechanical argument, not a global
  reweighting of the whole estimation.

## Minimum viable implementation and validation

- Model A (baseline): exact `m8trpg`, already cached
  (`oof_ensemble_v10.rds$oof_mlogit`, canonical 5-fold CV, seed 4821).
- Model B (candidate): identical formula with `In_seg3` and `In_seg5`
  removed, refit via the same canonical respondent-grouped 5-fold CV.
- Evaluate on **two** populations: (1) all training respondents (the
  ordinary comparison, expected to show at most a `mod11`-sized, likely
  negligible difference, since this is what CV directly optimizes for);
  (2) the top 30% of training respondents by test-likeness propensity
  (the project's own established diagnostic, reusing the same adversarial-
  validation classifier pattern from `codex_two_head_ensemble_v2.R`) --
  this is where the hypothesis predicts the real difference should show up.
- Respondent-clustered bootstrap (100,000 replicates) on both populations.

## Gates

Promotion requires the test-like-30% population's ordinary 95% CI to
exclude zero (the theoretically relevant population for this specific
hypothesis), not just the whole-population CI -- an explicit, pre-declared
deviation from this project's usual whole-population-first gate, justified
because the entire hypothesis is that whole-population CV cannot see this
effect. If promoted, propagate to the full ensemble and repeated CV before
any full-data build or submission.

## What would falsify this

The test-like-30% CI including zero, or the effect appearing equally weak
in both populations (would mean the mechanism is not really about the
segment-distribution mismatch), or the fitted `In_seg3`/`In_seg5`
coefficients in the pruned refit's dropped-term diagnostic turning out to
matter more than this audit suggests.
