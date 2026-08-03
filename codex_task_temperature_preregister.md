# Task-content-conditioned local temperature -- pre-registration

Date: 2026-07-31. Committed before any fold of this experiment is fit or
scored. Third experiment of this session, run after the hurdle model and the
three alternative-link families were independently pre-registered,
implemented, and rejected on their own merits (`codex_hurdle_model_findings.md`,
`codex_alt_link_findings.md`).

## Hypothesis

Confidence (softmax temperature) should vary with observable **task
difficulty** -- how close the leading bundles' predicted attractiveness is,
how spread the prices are -- rather than being a single global constant or a
respondent-covariate-indexed constant (both already tested and rejected). A
bounded linear function of task-difficulty features, ridge-penalized toward
`beta = 0` (full reversion to v14's plain softmax under strong shrinkage),
applied to v14's own frozen utilities, will show a respondent-clustered
bootstrap 95% CI that excludes zero when blended against the exact frozen
v14 prediction.

## Why this is not a duplicate

- **Not global utility-scale heterogeneity by respondent/task covariates**
  (`R/codex_global_scale.R`, rejected): that model's scale was a function of
  `Task_c`, `z(income)`, and segment dummies -- respondent and survey-position
  covariates, never a property of the specific three bundles shown in that
  task.
- **Not the post-hoc single global temperature** (rejected) or **family 1 of
  the alternative-link experiment** (`codex_alt_link_findings.md`, rejected,
  CI entirely below zero): both are a single constant scale, not a function
  of anything.
- **Not family 3 of the alternative-link experiment** (shape exponent on the
  surprisal, rejected): that reshapes probabilities globally by a fixed
  power; it does not condition on task content either.
- **Not the choice-set-geometry experiment** (near-miss, crosses zero): that
  added similarity/crowding features as *additive utility* terms inside the
  existing mlogit utility function. This experiment adds no utility terms at
  all -- it only rescales the whole already-fitted probability vector, and
  only as a function of task difficulty, never respondent outcomes.

This experiment is prioritized lower than the hurdle model and the
alternative-link families (per the original coverage-audit ranking) because
it is the closest in spirit to already-rejected mechanisms; it is run anyway,
briefly and cheaply, specifically so this remains a tested closure rather
than an assumption, per the brief's explicit instruction to audit rather
than assume this question is closed.

## Minimum viable implementation

Reuses `R/codex_alt_link.R`'s exact `msurp = -log(p^v14)` representation and
softmax-rescaling mechanism, generalizing only the scalar temperature `b` to
a **linear function of task-difficulty features**, computed once per task
(the same value for all 4 alternatives in that task) and completely
model-free (design-only, not from any additionally-fitted model):

- `gap_top2`: the difference between the largest and second-largest of v14's
  own frozen conditional-bundle probabilities among the three inside
  alternatives (closeness of the leading options -- a small gap is a
  genuinely difficult task).
- `price_cv`: the coefficient of variation of the three inside alternatives'
  prices (already computed identically in the hurdle-model experiment).

```
b_i = beta_0 + beta_gap * z(gap_top2_i) + beta_pricecv * z(price_cv_i)
eta_ij = -exp(b_i) * msurp_ij
p_i = softmax_j(eta_i)
```

All of `(beta_0, beta_gap, beta_pricecv)` are estimated by nested
cross-fitted penalized MLE with an explicit ridge penalty toward `beta = 0`
(exact v14 softmax) -- so at strong shrinkage the model reverts exactly to
`b_i = 0` for every task, satisfying the pre-registration's "must revert to
temperature one under strong shrinkage" requirement. No respondent-outcome
information enters `b_i` at any point (both features are computable from the
test choice set's own designed content alone).

## Validation protocol and gates

Identical to `codex_alt_link_preregister.md`: canonical respondent-grouped
5-fold CV (seed 4821, same partition), nested penalty selection
(`penalty_grid = c(1, 0.1, 0.01, 0.001, 0)`), gradient-checked analytic
gradient before any real fold is fit, respondent-clustered bootstrap
(100,000 replicates), escalation to the same 6 repeated-CV seeds only on a
canonical pass or the same pre-registered near-miss band
(`lower_95 in [-0.00075, 0]`), promotion requiring pooled `lower_95 > 0`,
>=5/6 positive repeats, and the test-like top-30% gate.

## What would falsify this

A canonical bootstrap 95% CI that includes zero (or is entirely below zero,
as all four already-tested link/scale variants this session were), OR a
pooled repeated-CV CI that includes zero, OR fewer than 5/6 positive repeat
seeds, OR a failed test-like-30% gate.
