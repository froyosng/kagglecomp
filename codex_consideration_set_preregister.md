# Conjunctive price-screening consideration-set mixture -- pre-registration

Date: 2026-07-31. Sixth piece of this session's adversarial-modelling round,
proposed by an independent second-opinion literature pass (`chatgpt_fresh_
ideation_brief.md`), audited before implementation, screened before any CV
compute is spent.

## Hypothesis

Every model tried in this project so far, including the newly-rejected
alternative-link and hurdle-model experiments, assumes every inside bundle
remains fully eligible and that an unattractive attribute can always be
compensated for by other attractive ones (a standard compensatory
random-utility assumption). A genuinely different hypothesis: some
respondents apply a **noncompensatory price screen** first -- eliminating
any bundle priced above a personal threshold from consideration entirely,
regardless of its other attributes -- and only then choose compensatorily
among what remains (Gilbride & Allenby 2004, *Marketing Science*,
"A Choice Model with Conjunctive, Disjunctive, and Compensatory Screening
Rules"). A two-type population mixture (screeners vs. non-screeners, type
probability predicted from income) will show a respondent-clustered
bootstrap gain over the frozen v14 prediction.

## Why this is not a duplicate

Verified directly against this project's own exhaustive history (not
assumed): nothing tried so far removes an alternative from the choice set's
denominator. The two-head/hurdle experiments repartition *which model*
predicts the opt-out margin vs. the conditional bundle choice, but every
inside bundle remains fully compensatorily competitive within the
conditional head. Latent-class experiments (task-fatigue class, price-scale
class) change the *strength* of a taste parameter across classes, never an
alternative's eligibility. RRM changes how bundles are *compared*, never
excludes one. This is a structurally different mechanism: a discontinuity
(or near-discontinuity, via a soft sigmoid gate) in whether an alternative
enters the compensatory competition at all.

## Minimum viable implementation

Freezes v14's own frozen, honest, cross-fitted probabilities as `V_ntj =
log(p^v14_ntj)` (identical convention to this session's alt-link and
task-temperature experiments) -- no utility function is re-estimated.

- **Gate** (inside alternatives only; the opt-out is always "available"):
  `g_ntj = sigmoid((c_n - Price_ntj) / tau)`, `c_n = a + b * z(income_n)`,
  `tau` fixed at a small positive constant for this first pass (not
  estimated, per the brief's own MVP advice not to start with the most
  flexible version).
- **Screened branch**: `P^S_ntj = g_ntj * exp(V_ntj) / [exp(V_nt4) +
  sum_k g_ntk * exp(V_ntk)]` for `j=1,2,3`; `P^S_nt4` takes the remaining
  mass. Note `P^0_nt` (the no-screening branch) is simply v14's own
  prediction unchanged, since `softmax(log(p)) = p` exactly.
- **Mixing weight**: `pi_n = sigmoid(alpha0 + alpha1 * z(income_n))`,
  predicted from income alone for this first pass (segment interactions
  deferred, per the brief's own advice against starting maximally flexible).
- **Training objective** (only on training-fold respondents): the
  **panel** log-likelihood, using all ~19 tasks per respondent jointly
  under each branch (`logP0_n`, precomputed once since it does not depend
  on theta, and `logPS_n(theta)`), combined via `logsumexp(log(1-pi_n) +
  logP0_n, log(pi_n) + logPS_n(theta))` -- not an independent per-row
  mixture, so that a respondent's full 19-task sequence, not a single task,
  determines whether they are estimated to behave as a screener.
- **Held-out prediction** (the critical leakage-safety step, explicitly
  flagged by the brief itself): for any respondent not used to fit theta,
  the per-row predicted probability is the simple **prior**-weighted
  mixture `P_nt = (1 - pi_n(z_n)) P^0_nt + pi_n(z_n) P^S_nt(theta)`, using
  only `pi_n` as a function of that respondent's own covariates -- never a
  posterior updated from their own (held-out) choices. This is the standard
  and only valid way to generate out-of-sample predictions from a discrete
  mixture/latent-class model for a respondent whose class is never observed
  (identical in spirit to how this project's own earlier latent-class
  experiments correctly predicted class membership from covariates alone,
  not from held-out outcomes).

Five free parameters total: `(a, b, alpha0, alpha1)` plus a fixed `tau`.

## Screening discipline

Because this model requires custom non-convex panel-likelihood
optimization (multiple random restarts, per this project's own established
caution around latent-class multimodality), it is **screened first** on the
existing canonical single 80/20 split (`data_processed/train_val_split.rds`,
908 training / 227 validation respondents) using v14's already-computed OOF
predictions restricted to that split, before any commitment to the full
nested 5-fold + repeated-6-seed protocol. A correctness self-check is
required before trusting any fitted result: at `pi_n = 0` for every
respondent (achieved by a strongly negative `alpha0`, `alpha1 = 0`), the
model's predicted probabilities must exactly reproduce v14's own known
validation-split log loss.

## Gates

1. **Correctness check**: `pi_n=0` must reproduce v14's exact val loss
   (to numerical tolerance) before any fitted result is trusted.
2. **Screen**: the fitted model's validation log loss must beat v14's own
   validation log loss on the identical split; if not, reject and log,
   no CV compute spent.
3. If the screen passes, escalate to the canonical respondent-grouped
   5-fold CV (seed 4821) with nested per-fold refitting, respondent-
   clustered bootstrap (100,000 replicates), and the same near-miss/
   promotion gates used throughout this session (`lower_95 > 0` for a
   pass; `[-0.00075, 0]` for a near-miss escalating to the 6-seed repeated
   CV; promotion requires pooled `lower_95 > 0`, >=5/6 positive repeats,
   and the test-like top-30% gate).

## Leakage checklist

- `pi_n` and `c_n` are functions of `income` alone, standardized using the
  training fold's own mean/sd, never a function of any choice outcome.
- Held-out prediction never uses a posterior over an individual
  respondent's own held-out choices -- verified by construction (the
  prediction function takes only covariates as input, never truth).
- `tau` is fixed, not tuned against validation loss, removing one
  degree of freedom that could otherwise be leakage-adjacent (tuned
  against the very data used to judge the result).

## What would falsify this

Failing the correctness check (implementation bug, must fix first). At the
screen stage: validation log loss not beating v14's own. At the CV stage:
an ordinary 95% bootstrap CI that includes zero, a fitted threshold outside
the observed price range, type-mixture proportions collapsing entirely
toward zero screening mass, or (per the brief's own criterion) any
indication the apparent gain depends on conditioning predictions on a
respondent's own held-out outcomes rather than covariates alone.
