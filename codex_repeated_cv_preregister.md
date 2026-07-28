# Pre-registration: repeated-CV confirmation and final candidate builds

This specification was fixed on branch `codex-repeat-cv`, based on
`zhenhao@482c469`, before running any additional fold assignment.

## Frozen candidates

No architecture or feature selection is reopened.

1. `history_prior_smoothed_features`: the already-selected `both_k3`
   specification from `R/codex_history_prior_smoothing.R`, propagated through
   the fixed submitted architecture
   `0.85 * (0.80 * mlogit + 0.20 * original_xgb) + 0.15 * shallow_mlp`.
2. `augmented8_arithmetic`: the exact eight components and fold-cross-fitted
   nonnegative arithmetic-weight optimizer from `R/codex_full_stacking.R`:
   m8trpg, original xgboost, rank:ndcg xgboost, retuned xgboost, glmnet-Cox,
   shallow MLP, triple-interaction mlogit, and deep MLP.

The comparison baseline for history remains its original fixed-15% submitted
architecture. The comparison baseline for the eight-component candidate
remains the original fold-cross-fitted v11+shallow-MLP candidate.

## Repeated folds

The canonical respondent-grouped five-fold result (seed 4821) is retained.
Exactly five additional balanced respondent-grouped five-fold assignments are
run, with seeds:

`1907, 2719, 6151, 8293, 104729`.

Every component is refit from scratch inside every new training fold using its
already-frozen hyperparameters. No canonical OOF prediction is relabeled or
reused under a new fold assignment. Within-model stochastic seeds are derived
only from the repeat and fold number and are not changed after results appear.

## Repeated-CV estimand and uncertainty

For every respondent and repeat, compute the respondent's mean row-loss gain
of candidate over its matching baseline. Average those six gains within
respondent first. The primary point estimate is the mean of those 1,135
respondent-level averages.

The primary 100,000-replicate bootstrap resamples the 1,135 respondents. It
does not treat the 30 folds or six predictions per respondent as independent
observations. Report:

- ordinary 95% and 99% intervals;
- the history candidate's retained cumulative family-13 interval;
- the eight-component candidate's retained family-6 interval;
- gain by repeat and by fold.

An existing candidate is promoted only if its repeated-CV point gain is
positive, the retained family-adjusted lower bound is above zero, and at least
five of six complete repeat estimates are positive. More splits reduce
fold-assignment/algorithm noise; they do not create new independent
respondents or erase the original search multiplicity.

## Eight-component submission

The full-data test submission is built regardless of the repeated-CV verdict
so it can be audited. Its weights are frozen now as the mean canonical
fold-cross-fitted weights from the existing saved result:

| Component | Weight |
|---|---:|
| m8trpg | 0.06246872652458 |
| original xgboost | 0.00474541986529 |
| rank:ndcg xgboost | 0.09330490494295 |
| retuned xgboost | 0.01555896281673 |
| glmnet-Cox | 0.13111978085328 |
| shallow MLP | 0.08008528400844 |
| triple-interaction mlogit | 0.47925772846576 |
| deep MLP | 0.13345919252297 |

The final output must preserve sample-submission order, contain finite strictly
positive probabilities, and sum to one per row. Compare it to the submitted
v11+MLP file overall and specifically for test respondent `No=22637`: maximum
absolute probability change, counts above 0.05/0.10/0.15, and the contribution
of each component to the maximum-change row.

## Joint history-plus-triple follow-up

One candidate only: add the already-frozen `both_k3` history terms to the
already-frozen `Price x z(income) x z(mileage)` m8trpg component, then replace
the plain triple component in the eight-component pool.

Screen gate: on the canonical seed-7402 respondent split, the joint mlogit must
improve over the plain triple mlogit. If it passes, run only canonical
respondent-grouped five-fold CV (seed 4821), cross-fit the same eight arithmetic
weights, and bootstrap against both the submitted current model and the plain
eight-component blend. No post-hoc removal of either history term is allowed.

## Explicitly excluded

Respondent-bootstrap bagging of m8trpg is not run. It is already a completed,
logged negative experiment: 15 bags across the canonical five folds worsened
the ensemble from 1.145094 to 1.145658, with every learning-curve point on the
harmful side.
