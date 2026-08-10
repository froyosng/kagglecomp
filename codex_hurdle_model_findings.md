# Dedicated hurdle (opt-out / conditional-bundle) model -- findings

Pre-registration: `codex_hurdle_model_preregister.md` (committed before any
result was seen). Implementation: `R/codex_hurdle_model.R`.

## Result: clean, decisive reject

Canonical five-fold respondent-grouped CV (seed 4821): the nested
cross-fitted blend-weight search (`crossfit_blend()`, grid `seq(0, 0.40, by =
0.02)`) selected **weight = 0 in every one of the 5 outer folds**, so the
blended candidate is numerically identical to the exact frozen v14 prediction
(`candidate_logloss == baseline_logloss == 1.14353315`, `point_gain` =
1.9e-19, i.e. floating-point noise around an exact zero, not a real effect).
Repeated CV (the same 6 canonical seeds used throughout this project)
confirms this: **0 of 6 repeats positive**, pooled gain again ~2.8e-19, and
the test-like top-30%-respondent slice shows the same null result. Verdict:
**REJECT**, exactly as `R/codex_hurdle_model.R`'s own pre-registered gates
concluded.

Note on the escalation trigger: because the per-respondent gain is
deterministically ~0 (baseline and candidate are identical row-for-row), the
bootstrap has essentially no variance (`bootstrap_sd` ~4.5e-20), so its lower
95% bound came out as a tiny positive floating-point epsilon rather than a
true negative bound -- this technically satisfied the pre-registered
"canonical pass" condition (`point_gain > 0` and `lower_95 > 0`) and
triggered repeated-CV escalation on a result that was actually an exact null,
not a real near-miss. Harmless in this case (repeated CV correctly rejects
regardless, `positive_repeats = 0`), but worth flagging: any future exact-zero
result at the canonical stage should be treated as an immediate reject, not
escalated.

## Why this happened (independently verified from the raw saved artifacts)

Read `data_processed/codex_hurdle_model/canonical_result.rds` directly rather
than trusting the printed summary:

- `q`'s calibration is sane: mean predicted opt-out probability 0.30243 vs.
  the actual training opt-out rate 0.30230, essentially identical to v14's
  own mean q (0.30138). Not a degenerate or broken component.
- The **standalone hurdle prediction is uniformly worse than v14 in every
  single outer fold**, by 0.008-0.020 log loss (fold 1: 1.2115 vs. v14's
  1.2032; fold 4: 1.1365 vs. v14's 1.1168; etc.) -- not close in any fold,
  and the gap is fairly stable across folds (no fold where the hurdle
  component is competitive, let alone better).
- Its overall standalone log loss (1.159787) sits in a similar range to this
  project's other from-scratch alternative estimators (the `glmnet`
  stratified-Cox model from the 4-way ensemble round scored 1.164331 alone),
  so this is not an implementation bug producing nonsense numbers -- it is a
  real, moderately-competent standalone model that simply isn't different
  enough from the existing four components (mlogit, xgboost, shallow MLP,
  set-context) to earn *any* positive blend weight, unlike xgboost, the
  shallow MLP, and the set-context network, each individually weaker than
  the logit alone yet each still earning real ensemble weight because their
  errors are sufficiently uncorrelated with it.

This is a materially more decisive rejection than the two-head pooling
near-miss (89.6% win rate, CI barely crossing zero): here the fresh,
dedicated, heavily-regularized decomposition never clears zero weight at
all, in any of the 30 outer-fold fits (5 folds x 6 seeds) run.

## What this closes and what it does not

- Closes the "genuinely untested" gap identified in the coverage audit: a
  freshly-trained binary opt-out model (new task-difficulty/design features,
  offset-anchored to v14) plus a freshly-trained 3-way conditional bundle
  model (m8trpg's own feature set, re-estimated on the inside-only
  subsample) has now actually been built and tested, not just reasoned
  about. It does not clear the bar.
- Does **not** by itself rule out every possible hurdle-style decomposition
  -- only this specific feature set and estimator combination. In
  particular it does not test: richer task-difficulty features (design
  history, dominance/compromise structure via a different operationalization
  than "count of attributes varying"), a jointly-trained (not two-stage)
  hurdle objective, or a version that shares information between `q` and `r`
  beyond the offset. Per this project's standard, none of those are
  automatically promoted as "still open" without their own concrete
  implementation and test; they are noted only so a future session does not
  mistake this result for having closed the entire hurdle-architecture
  question.
- Corroborates, from a third independent angle (after the two-head pooling
  experiment and the OOF residual audit), that this dataset's four existing
  components have already captured essentially all the exploitable
  structure in the opt-out/conditional-bundle decomposition available from
  these covariates.

## Structural / numerical issues fixed before any result was seen (all
pre-registered as corrections, not post-hoc adjustments)

1. Dropped every `inside x covariate` term from `r`'s formula: once the
   opt-out alternative is removed, `inside` is constant (1) across the
   remaining alternatives, so those terms have no within-task variation and
   are unidentified in a conditional-choice likelihood.
2. Folded attribute `HU`'s level 2 into its reference level: this
   partial-profile design fixes exactly 9-of-19 active attributes per
   alternative (a known structural fact, `cleaning_log.md` finding #2), which
   becomes an exact linear identity once the opt-out row (the only row type
   that broke the "always 9" pattern) is removed -- the same mechanism as the
   original Price-factor collinearity, fixed the same way (drop one
   arbitrary column).
3. Even after both fixes, plain unpenalized `mlogit` MLE still hit an
   exactly-singular Newton-Raphson Hessian on this smaller, restricted
   training subsample (isolated via progressive term-block re-addition and a
   direct `qr()` full-rank check that ruled out any further exact
   collinearity) -- consistent with quasi-complete separation in a sparser
   interaction cell than m8trpg's full training set ever produces. Switched
   `r`'s estimator to `glmnet`'s stratified-Cox equivalence to the
   conditional-logit likelihood (ridge, `alpha = 0`), the same
   already-validated technique this project used for its regularized
   conditional-logit interaction search (`R/codex_glmnet_cox_ensemble.R`).
   This is numerically robust to the failure mode by construction and
   delivers the pre-registered "heavy regularization" directly.

## Not adopted; no submission made

`set_context_utility_network_v14` (1.143533 CV / 1.200 public) remains the
current best model, unchanged.
