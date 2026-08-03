# Alternative-specific asymmetric probability link -- pre-registration

Date: 2026-07-31. Committed before any fold of this experiment is fit or
scored. Second experiment of this session, run after the hurdle model
(`codex_hurdle_model_preregister.md`/`codex_hurdle_model_findings.md`) was
independently pre-registered, implemented, and rejected on its own merits.

## Hypothesis

The project has exhaustively tested changes to the **utility function**
(which features enter, how attributes/price are coded, which interactions
exist) but has never tested a change to the **link function** -- the fixed
map from utility differences to choice probabilities, which every model in
this project (mlogit, xgboost's `multi:softprob`, every neural component)
implicitly assumes is ordinary softmax (equivalent to i.i.d. type-I extreme
value / Gumbel utility errors with a common, alternative-invariant scale and
an independence-across-alternatives copula). A small-parameter,
alternative-position-specific asymmetric transform of the frozen, honest,
cross-fitted v14 utilities -- estimated as a bounded residual correction with
very few global parameters, reverting to plain softmax under shrinkage --
will show a respondent-clustered bootstrap 95% CI that excludes zero when
blended against the exact frozen v14 prediction.

Literature motivation: the Marginal Distribution Model (MDM) family (Natarajan
et al. 2009; Mishra, Natarajan, Padmanabhan, Teo, Li 2014, *Management
Science*, "On Theoretical and Empirical Aspects of Marginal Distribution
Choice Models") replaces the softmax/Gumbel-independence assumption with a
distributionally-robust or moment-matched choice probability derived only
from each alternative's marginal utility distribution, without assuming
independence across alternatives -- and has been applied to conjoint-style
choice data as a genuine alternative to MNL/mixed logit. The full MDM
requires solving a semidefinite/convex program per choice set and is not a
"few global parameters" implementation; per this experiment's own
pre-registered scope (and the project's standing rule against starting with
a large flexible version cold), this experiment does **not** implement full
MDM. It implements the narrowest, cheapest, most falsifiable piece of the
same idea: relaxing the *shape* of the utility-to-probability map by a small
number of global parameters, while leaving the utility function (v14 itself)
completely untouched.

## Why this is not a duplicate

- **Not global utility-scale heterogeneity** (`R/codex_global_scale.R`,
  rejected, decisively harmful at every ridge strength): that experiment
  multiplies the *entire* utility vector by a single scalar that varies by
  respondent/task covariates (income, segment, task position) -- a scale
  change, applied identically to every alternative, which is exactly a
  softmax temperature parameterized by covariates. It never changes the
  *shape* of the link (still exactly softmax, just re-scaled).
- **Not the post-hoc temperature/shrinkage calibration**
  (`cleaning_log.md`, 2026-07-27, "clean no"): that swept a single *global*
  temperature (0.75-1.35) and shrinkage toward uniform/empirical shares,
  applied identically to every row and every alternative. Also a pure
  rescaling of the same softmax shape, not a shape change, and not
  alternative-specific.
- **Not the exact-softmax shared-utility experiments**
  (`R/codex_shared_utility_*`, rejected): those changed which *features*
  enter a shared utility function under an unchanged exact-softmax
  likelihood. This experiment holds the utility function completely fixed
  (v14's own frozen predictions) and changes only the function mapping those
  utilities to probabilities.
- **Not choice-set-geometry or task-difficulty features** (near-miss,
  crosses zero): those added *inputs* to a utility function under softmax.
  This experiment adds no new inputs at all -- only a small number of global
  shape parameters on the existing utilities.

## Minimum viable implementation (small number of global parameters)

Let `u_ij` be v14's frozen, honest, cross-fitted log-odds for row `i`,
alternative `j`: `u_ij = log(p^v14_ij)` (shift-invariant, so the arbitrary
additive constant from v14's own normalization does not matter). Define an
**alternative-position-specific power link**:

```
p_ij(theta) = softmax_j( (1 + theta_j) * u_ij )       for j = 1 (cheapest-ish
                                                        position), ... no --
```

Concretely, three candidate link families are pre-registered, in increasing
order of flexibility, and will be screened in this order (stop at the first
that clears the canonical screen; do not proceed to a more flexible family
unless the simpler one is rejected):

1. **Single global exponent** `p_i(theta) = softmax( (1+theta) * u_i )` --
   one parameter. Identical in form to a global temperature, kept only as an
   exact-replication sanity check (this is expected to reproduce the
   already-rejected post-hoc calibration null almost exactly, confirming the
   estimation pipeline is correct before trusting anything more flexible).
2. **Opt-out-specific asymmetric link** `p_i(theta) = softmax( u_i +
   theta * e_4 )`, where `e_4` is the indicator for the opt-out alternative
   -- i.e. a single learned *additive* shift specifically on the opt-out
   log-odds, applied uniformly to every task (not residual-boosting the
   whole utility, just testing whether the opt-out margin specifically is
   systematically mis-scaled relative to the inside bundles under the
   current softmax). Two parameters total when combined with (1).
3. **Semiparametric link with a strongly constrained variance/shape
   parameter**: a two-parameter Prentice-Gloeckler / generalized-extreme-
   value-family reweighting, `p_ij(theta) proportional to
   sign(u_ij) * |u_ij|^(1+theta_shape) * exp(theta_scale * u_ij)`, clipped
   to a bounded neighborhood of the identity (`theta_shape, theta_scale`
   both ridge-penalized toward 0, which exactly recovers plain softmax) --
   the closest single-digit-parameter approximation to letting the marginal
   utility distribution's shape (not just its scale) differ from the
   Gumbel/logistic assumption softmax presumes, without solving a full MDM
   program.

All three are estimated by nested cross-fitted maximum likelihood (respondent-
grouped folds, canonical seed 4821, same partition as every other component)
with an explicit ridge penalty on `theta` that shrinks the fit toward
`theta = 0` (exact softmax, i.e. exact v14) -- so the link can only revert to
identity, never diverge further from it than the data supports.

## Validation protocol (unchanged project standard)

- Same canonical respondent-grouped 5-fold partition, same 6 repeated-CV
  seeds, same respondent-clustered bootstrap (100,000 replicates), same
  `crossfit_blend()`-style honest weight/parameter selection pattern used by
  every other component this session.
- Zero respondent overlap between fitting and evaluation, asserted by hard
  `stopifnot`.
- A gradient check (analytic vs. finite-difference) for the custom link's
  log-likelihood gradient before any fold is fit, matching this project's
  standing practice for custom objectives (e.g. the shared-utility exact-
  softmax gradient check, the RRM gradient check).

## Pre-registered gates (fixed before any result is seen)

1. **Screen (family 1, cheap correctness check):** must reproduce the
   existing post-hoc-calibration null (i.e. `theta` shrinks to ~0, no gain)
   -- if family 1 finds a *real* gain, that would itself be a surprising
   contradiction of an already-logged result and triggers a full re-audit of
   the implementation before trusting it, not a promotion.
2. **Family 2/3 canonical pass:** `point_gain > 0` and `lower_95 > 0` ->
   escalate to repeated CV.
3. **Family 2/3 canonical near-miss:** `point_gain > 0` and `lower_95` in
   `[-0.00075, 0]` -> escalate to repeated CV.
4. Otherwise: reject that family, log, move to the next family only if a
   simpler family has not already been rejected for the same reason (a more
   flexible family is not tried merely because a simpler one failed, unless
   the failure mode specifically motivates more flexibility -- e.g. family 1
   nulling out does not by itself justify family 3, since family 3's shape
   parameter is a qualitatively different mechanism, not "more of the same
   scale change").
5. **Repeated-CV promotion:** identical rule to every other candidate this
   project has evaluated -- pooled `point_gain > 0`, pooled `lower_95 > 0`,
   >=5/6 repeat seeds positive, and the test-like top-30%-respondent slice's
   `point_gain >= 0` with `lower_95 >= -0.001`.
6. A promoted candidate still requires a full-data build with an MD5-lock +
   two-independent-run reproducibility check before any Kaggle submission.

## Leakage checklist

- `u_ij` sourced only from the already-verified out-of-fold v14 artifacts
  (`codex_set_context_network/canonical_result.rds` and the matching
  `repeat_result_<seed>.rds`), never refit on the evaluation row's own
  respondent.
- No test-set information of any kind enters fitting; `theta` is a handful
  of global scalars, not respondent- or task-indexed.
- Ridge penalty strength chosen by nested `cv.glmnet`-style inner CV
  restricted to the outer-training folds only.

## What would falsify this

Family 1 failing to reproduce the known null (implementation bug, must fix
before proceeding). Families 2/3: a canonical bootstrap 95% CI that includes
zero, OR a pooled repeated-CV CI that includes zero, OR fewer than 5/6
positive repeat seeds, OR a failed test-like-30% gate. A positive point
estimate alone does not promote any family.
