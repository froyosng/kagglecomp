# Choice-set-dependent attribute focusing/salience -- pre-registration

Date: 2026-07-31. Tenth experiment of this session. Proposed by the same
external second-opinion pass (ChatGPT), after being shown today's other
seven rejections plus the "bucket 3/bucket 4 still alive" framing. Audited
before implementation.

## Hypothesis

Respondents overweight attributes on which the displayed alternatives
differ substantially in a given task, and underweight attributes that
barely distinguish the options shown (Koszegi & Szeidl 2013, *QJE*, "A Model
of Focusing in Economic Choice"; Bordalo, Gennaioli & Shleifer, salience
theory of choice). This is especially plausible in a partial-profile design
where which attributes are even active, let alone distinguishing, changes
task to task. A one-parameter reweighting of the existing conditional
logit's own fitted per-attribute utility contributions -- upweighting
attributes with a large cross-alternative range in that specific task,
downweighting those with a small range, holding the total attribute-utility
"mass" per task fixed -- will show a validation log loss improvement over
the unweighted (`lambda=0`) model.

## Mechanism (exact formula)

Let `c_ntjm` be attribute `m`'s fitted utility contribution to inside
alternative `j` in task `t` for respondent `n` (i.e. the fitted `factor(m)`
coefficient for whichever level alternative `j` has on attribute `m`; 0 at
the reference level). For each task, over the 19 categorical attributes
(price excluded, per the falsification-relevant reasoning that price rank/
gap is already heavily exploited and the test should isolate whether
*non-price* attribute contrast adds anything):

```
d_ntm = max_{j in {1,2,3}} c_ntjm - min_{j in {1,2,3}} c_ntjm
z_ntm = (d_ntm - mean_m(d over training tasks)) / sd_m(d over training tasks)   [per-attribute standardization]
w_ntm(lambda) = exp(lambda * z_ntm) / [(1/19) * sum_r exp(lambda * z_ntr)]      [softmax-normalized across the 19 attributes, within task]
```

`sum_m w_ntm(lambda) = 19` exactly for any `lambda` (a built-in conservation
property: attribute-attention mass is redistributed, not created or
destroyed). At `lambda = 0`, `w_ntm = 1` for every attribute, exactly
recovering the unweighted model. The corrected utility is:

```
U_ntj(lambda) = U_ntj^(frozen) + sum_m (w_ntm(lambda) - 1) * c_ntjm
```

applied as a residual correction on top of the frozen, fold-fitted m8trpg
utility (`U^(frozen) = log(p^m8trpg)`), not a full joint re-estimation --
matching the brief's own MVP guidance ("use fold-fitted logit coefficients
... no full-data coefficients," i.e. `c_ntjm` comes from that fold's own
fitted model, never a leakage-prone full-data refit). The opt-out
alternative needs no special case: its attribute profile is all-zero, so
`c_nt4m = 0` for every attribute and the correction is automatically zero
for it.

## Why this is not a duplicate

- **Not choice-set geometry** (near-miss, additive similarity features):
  that model adds `V_j + f(similarity of j to competitors)` -- an
  *additional* utility term. This model *reweights the existing attribute
  decomposition itself*, changing how much each already-estimated
  coefficient counts, without adding a new additive term.
- **Not the set-context network or its variance-pooling extension**: those
  are learned, implicit representations inside a neural architecture. This
  is an explicit, one-parameter, interpretable reweighting of an already-
  fitted parametric logit's own coefficients.
- **Not RRM** (pairwise regret-based comparison) or the rejected
  alternative-link experiments (which rescale the *final probability*, not
  the per-attribute utility decomposition prior to summation).
- **Not the already-tested attribute min/max rank flags** (null, reversed
  under CV): those used attributes' raw *level codes* as if cardinally
  ordered, which is invalid for most attributes (confirmed in this
  project's own history). This uses each attribute's own *fitted
  coefficient* as the utility-scale quantity being compared across
  alternatives, side-stepping the raw-level-ordering problem entirely.

## Minimum viable implementation and screening discipline

Deliberately minimal, per the brief's own advice:

1. Fit m8trpg once on the canonical single-split training set
   (`data_processed/train_val_split.rds`), extract its fitted attribute
   coefficients.
2. Compute `d_ntm`/`z_ntm` (attributes only, opt-out excluded from the
   range computation, price excluded from the reweighted set) on the
   training tasks (for standardization) and validation tasks.
3. **Coarse one-dimensional grid search over `lambda >= 0`** (no continuous
   optimizer for this first pass) evaluating validation log loss directly.
4. Compare against the unweighted (`lambda=0`) validation log loss on the
   identical split -- this is exactly `m8trpg`'s own already-known screen
   number (1.159681), a free correctness cross-check.
5. **Placebo check**: randomly permute each task's vector of 19 attribute
   ranges among *other* training-fold tasks before computing `z_ntm`,
   refit/re-grid-search `lambda` the same way. A genuine focusing effect
   must beat this placebo clearly; if the permuted version does comparably
   well, the apparent gain is not attributable to genuine task-specific
   attribute contrast.

Only escalate to the canonical respondent-grouped 5-fold CV (fold-fitted
coefficients per outer fold, as the brief specifies) if the screen beats
`lambda=0` **and** clearly beats the placebo.

## Pre-registered falsification criteria (from the brief, adopted verbatim)

Reject immediately if any of:
- the optimal `lambda` (unconstrained) is `<= 0`, contrary to the proposed
  mechanism;
- the placebo (shuffled attribute-range assignment) performs comparably to
  the real version;
- gains do not concentrate in tasks where one or two attributes account for
  most of the cross-alternative utility dispersion;
- it improves the standalone logit but the effect vanishes once propagated
  into the full ensemble;
- a materially different `lambda` is preferred by different folds.

## Leakage checklist

`c_ntjm` uses only that fold's own training-fitted coefficients (never
full-data or test-set information); `z_ntm`'s standardization constants are
fit on training tasks only; the opt-out alternative and price are excluded
from the reweighted attribute set for the reasons stated above, not tuned
in or out based on validation performance.
