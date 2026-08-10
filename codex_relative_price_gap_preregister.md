# Dispersion-relative price-gap features -- pre-registration

Date: 2026-07-31. Ninth experiment of this session, chosen from the two
"still alive" buckets identified by tracing back every real historical
improvement in this project (bucket 3: new choice-set-structure
information; bucket 4: a genuinely different ensemble function class) --
deliberately not another variant of the two dead buckets (per-respondent
personalization; recalibration of an already-calibrated output) that
today's eight prior experiments all fell into.

## Hypothesis

The project's single biggest-ever incremental gain (`price_gap_min`/
`price_gap_max`, +0.0046 CV from 2 parameters) encodes the *absolute*
price-level distance from an alternative to the choice set's cheapest/
dearest option. It does not encode whether that distance is large or small
*relative to* the overall price dispersion of that specific choice set. A
gap of 2 price levels is a large, decisive signal in a task where the three
prices are otherwise clustered together (total spread 3), but a much
weaker one in a task where prices already span most of the 1-12 range
(total spread 10). Adding `price_gap_min` and `price_gap_max`, each
rescaled by the task's own price spread, will show a respondent-clustered
bootstrap 95% CI that excludes zero when added to the existing best logit.

## Why this is not a duplicate

- **Not the already-tested "relative price" feature** (`Price - task mean`,
  rejected as "near-zero effect, degenerate by construction -- a constant
  shift across alternatives cancels in the logit," `cleaning_log.md`
  2026-07-26): that was an additive, linear transform of `Price`, provably
  collinear with existing terms. `price_gap_min / price_spread` is a
  division by a *task-varying* denominator -- a genuinely nonlinear
  transform, not a linear shift, and not collinear with anything already in
  the model.
- **Not the already-tested attribute min/max rank flags** (null, reversed
  under CV): those applied a rank concept to categorical attribute codes
  without a stable cardinal ordering. Price is the one variable in this
  dataset with a confirmed, exploited cardinal ordering, and this
  experiment only rescales an already-validated price feature by another
  price-derived quantity.
- **Not the task-difficulty temperature/scale experiments** (all rejected
  today): those used `price_cv`/`price_spread` to rescale the *entire
  probability vector's confidence* (a link-function change). This
  experiment adds `price_gap_min`/`price_gap_max` divided by spread as
  ordinary *utility*-level covariates in the existing conditional logit --
  a feature-engineering change, not a probability-link change. Confirmed
  materially different failure/success mechanism: a link change can only
  rescale what the utility function already produces, while a new utility
  covariate can change the utility function's ranking of alternatives
  itself.
- **Not choice-set geometry** (near-miss, crosses zero): that measured
  attribute/price *similarity* between specific alternative pairs. This
  measures the *overall* price dispersion of the task, a task-level summary
  statistic, used only to rescale an already-alternative-varying feature
  (not as a new independent term, avoiding the identification issue the
  hurdle model's `In_*` terms hit).

## Minimum viable implementation

On top of the exact `m8trpg` formula (`submit_ensemble_v11.R`'s `fml`), add
exactly two terms:

- `price_gap_min_rel = price_gap_min / max(price_spread, 1)` (floored at 1
  price level to avoid dividing by zero on the rare task where all three
  inside prices tie; `price_gap_min` itself is already 0 in that case, so
  the floor only prevents `0/0`, it does not change any nonzero case).
- `price_gap_max_rel = price_gap_max / max(price_spread, 1)`.

Both are alternative-varying (numerator varies by alternative, denominator
is task-constant), so no `In_*`-style identification issue applies. No
other term is changed, added, or removed.

## Screening discipline

Screened first on the canonical single 80/20 split
(`data_processed/train_val_split.rds`) before any CV compute, matching this
project's established screen-first practice, given eight consecutive
rejections today already.

## Gates

1. **Screen**: single-split validation log loss must beat the exact
   `m8trpg` baseline on the identical split.
2. **Canonical CV pass**: `point_gain > 0`, `lower_95 > 0` on the canonical
   respondent-grouped 5-fold CV (seed 4821) -> escalate to repeated CV.
3. **Canonical near-miss**: `lower_95` in `[-0.00075, 0]` -> escalate to
   repeated CV.
4. Otherwise reject at the canonical stage.
5. **Repeated-CV promotion**: identical rule to every other candidate this
   session -- pooled `lower_95 > 0`, >=5/6 positive repeats, test-like
   top-30% gate.

## Leakage checklist

`price_spread`/`price_gap_min`/`price_gap_max` are computed purely from the
three inside alternatives' designed `Price` columns for that task -- no
outcome, no fitted model, no respondent-level information. Identical
computation at train and test time.

## What would falsify this

Failing the screen. At CV: an ordinary 95% CI that includes zero, or (if a
near-miss escalates) a pooled repeated-CV CI that includes zero or fewer
than 5/6 positive repeats.
