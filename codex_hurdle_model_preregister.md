# Dedicated hurdle (opt-out / conditional-bundle) model -- pre-registration

Date: 2026-07-31. Committed before any fold of this experiment is fit or scored.

## Hypothesis

A genuinely dedicated two-part ("hurdle") model --

1. a freshly-trained binary **opt-out propensity model `q(x)`** using task-difficulty /
   choice-set-design features never used by any existing component, plus a
   fixed-coefficient (unpenalized) offset equal to the frozen v14 opt-out margin, and
2. a freshly-trained **3-way conditional bundle-choice model `r(x)`** sharing
   parameters across alternatives 1-3, estimated only on rows where an inside
   bundle was chosen, using `m8trpg`'s exact already-validated feature formula
   with no new terms --

recombined as `P(Ch4) = q`, `P(Chj) = (1-q) * r_j` and then blended against the
exact frozen v14 OOF prediction, will show a respondent-clustered bootstrap 95%
CI that excludes zero.

## Why this is not a duplicate of anything already logged

- **Not nested logit** (rejected, `cleaning_log.md` 2026-07-26/Imelda's independent
  test): nested logit imposes a substitution-pattern restriction inside one
  jointly-estimated 4-way likelihood. Here two separately-estimated models with
  their own coefficients and their own feature set for the opt-out margin are
  fit.
- **Not the binary chosen/not-chosen xgboost renormalization** (rejected,
  `R/test_xgb_binary_choice.R`, 2026-07-26): that reused the same 4-way
  row-level attribute/price features per alternative and renormalized softly.
  Here `q` is a genuinely different target (task-level opt-out) with new
  task-difficulty covariates (`price_spread`, `price_cv`, `n_attrs_varying`)
  that appear in no existing component, and `r` is fit on a restricted
  (inside-only, 3-alternative) subsample with its own coefficients -- not a
  renormalized 4-way score.
- **Not the two-head pooling experiment** (rejected, `R/codex_two_head_ensemble_v2.R`,
  verified by direct code read): two-head pooling takes four *already fixed*,
  already fully-4-way-trained component probability matrices (mlogit, xgboost,
  shallow MLP, set-context) and fits only a convex-combination weight per head
  via `fit_convex_pool()`. No new information enters and no new parameters are
  estimated on raw features. This experiment trains fresh parameters on fresh
  features never seen by any existing component and estimates the
  conditional-bundle coefficients on a different (inside-only) sample -- a
  structurally different estimating equation. Confirmed via repo-wide grep:
  no script anywhere trains a fresh binary opt-out classifier or a conditional
  logit restricted to inside-only rows.
- **Not the componentwise exact-softmax residual boosting** (rejected,
  smallest-ever near-miss, +0.0000382 repeated-CV, dominated by
  `price_gap_min`/`price_gap_max` -- i.e. it re-discovered existing features in
  residual form). The new features here (`price_spread`, `price_cv`,
  `n_attrs_varying`) were never in that boosting's candidate pool.

## Feature set (frozen before fitting; identical definitions at train and test)

**`q` (opt-out head), one row per choice task (task-level, from the wide table):**

- `Task_c` = `(Task - 10) / 9` (existing fatigue proxy)
- `price_min`, `price_max`, `price_spread = price_max - price_min`,
  `price_mean`, `price_cv = price_spread / price_mean`, computed over the 3
  inside alternatives' `Price1..Price3`
- `n_attrs_varying`: count, over the 19 attribute columns, of attributes where
  the 3 inside alternatives do **not** all share an identical level -- a
  genuine, model-free choice-set-complexity proxy, distinct from the fixed
  per-alternative "9 of 19 active" structural constant already known to be
  invariant
- Standardized `incomea`, `agea`, `milesa`, `nighta`, `genderind`, `Urbind`,
  `educind` (identical variables to mod7/mod8's `In_*` terms -- no new
  covariate introduced, so no new sparsity risk)
- Factor dummies for `segmentind` (6 levels), `regionind` (5), `pparkind` (5)
  (identical to mod8's existing segment/region/ppark terms)
- **Offset** (fixed coefficient = 1, never penalized): `qlogis(q_v14)`, where
  `q_v14` is the frozen, honest, cross-fitted v14 OOF probability of `Ch4` for
  that row, taken from `data_processed/codex_set_context_network/canonical_result.rds`
  (canonical seed) or the matching `repeat_result_<seed>.rds` (repeated-CV
  seeds) -- the same verified artifacts and leakage-safety pattern
  `codex_two_head_ensemble_v2.R` already uses.
- Estimator: `glmnet(family = "binomial", alpha = 0)` (ridge), penalty chosen
  by `cv.glmnet` with a respondent-grouped `foldid` restricted to the 4
  outer-training folds only (never the held-out outer fold) -- nested,
  leakage-free lambda selection. Only the new feature coefficients are
  penalized; the offset is unpenalized and fixed at coefficient 1, so the
  model can only add a bounded correction on top of v14's implied opt-out
  logit, never rescale it (this is the structural difference from the
  already-rejected global temperature-scaling experiments).

**`r` (conditional bundle head), 3-way conditional logit, inside-only rows:**

- `m8trpg`'s formula (`submit_ensemble_v11.R`'s `fml`: 19 attributes as
  factors, Price as a 12-level factor, `d2`/`d3`, Price x
  {income,age,miles,night,segment,task,region,ppark}, `is_cheapest`,
  `is_dearest`, `price_gap_min`, `price_gap_max`), refit via `mlogit`/`dfidx`
  on the subset of choice tasks where `Ch4 == 0`, with alternative 4 removed
  from the choice set entirely (3 alternatives, not 4). No new term is added
  relative to `m8trpg`.
  **One structural correction, caught by the smoke test before any result was
  seen:** every `inside x covariate` term (`In_income`, `In_age`, ...,
  `In_ppark5`) is dropped. In the full 4-way model `inside` varies across
  alternatives (0 for the opt-out, 1 for alts 1-3), which is what makes those
  interactions identifiable. Once alt 4 is removed, `inside` is identically 1
  for every remaining alternative in every task, so those products no longer
  vary within a choice set -- conditional logit only identifies effects
  through within-task utility differences, so a term constant across a
  task's alternatives is unidentified (this is exactly what produced the
  smoke test's singular-Hessian error, diagnosed directly rather than papered
  over with a regularization hack). This is not a loss of information: the
  opt-out margin's heterogeneity that those terms captured is now the `q`
  head's job. Only the alternative-varying `Price x covariate` and design
  terms remain in `r`'s formula.
  **A second structural correction, also caught by the smoke test's
  `qr()`-rank diagnostic before any result was seen:** this partial-profile
  design fixes every alternative at exactly 9 of 19 active (non-reference)
  attributes (`cleaning_log.md` finding #2). With the opt-out alternative
  removed, every remaining row shares this same total, so the 19 attributes'
  one-hot dummy columns sum to an exact constant for every observation -- a
  genuine rank-1-deficient direction, the conditional-logit analogue of the
  original Price-factor collinearity (which was only avoided there because
  the opt-out's all-zero profile broke the "always 9" pattern). Fixed the
  same way the original collinearity was fixed: folding one attribute level
  (`HU` level 2, arbitrarily -- any of the 19 attributes' levels is
  interchangeable here) into its reference level. This does not change the
  fitted choice probabilities (the likelihood is exactly flat along that one
  direction); it only means `HU`'s own level-2 effect is no longer separately
  estimable in the `r` head.
  **A third, decisive correction, also caught by the smoke test before any
  result was seen:** even after both fixes above, plain unpenalized `mlogit`
  MLE still hit an exactly-singular Newton-Raphson Hessian on this smaller,
  restricted (inside-only, single-outer-fold) training subsample -- isolated
  by progressively re-adding term blocks, it recurred even for the
  attributes+price+`d2`/`d3` core alone, and a direct `qr()` rank check of
  that exact design matrix showed full column rank, ruling out a further
  exact linear dependency. This is consistent with quasi-complete separation
  in a sparser interaction cell than m8trpg's full training set ever
  produces, not a remaining design bug. **`r` is therefore estimated via
  `glmnet`'s stratified-Cox equivalence to the conditional-logit likelihood
  (`family = "cox"`, one stratum per choice task via `survival::stratifySurv`,
  ridge `alpha = 0`)** -- the same equivalence and package this project
  already validated for its regularized conditional-logit interaction search
  (`R/codex_glmnet_cox_ensemble.R`, `cleaning_log.md` 2026-07-25). This is a
  strictly more conservative estimator for the identical feature set (no
  formula term changed), numerically robust to the separation failure mode by
  construction, and it directly delivers the "heavy regularization" the
  hypothesis calls for rather than patching the MLE with further ad hoc term
  removal. Regularization strength (`lambda`) is chosen the same way as `q`'s:
  nested, respondent-grouped `cv.glmnet` restricted to the outer-training
  folds.

## Recombination and blending

- `P(Ch4) = q(x)`; `P(Chj) = (1 - q(x)) * r_j(x)` for `j = 1,2,3`, renormalized
  to guard floating-point drift.
- The hurdle model's OOF prediction is blended against the exact frozen v14
  OOF prediction via the project's established `crossfit_blend()` pattern
  (grid search over blend weight per outer fold, minimizing log loss on that
  fold's pooled training folds), verbatim from `codex_set_context_network_v2.R`.

## Validation protocol (unchanged project standard)

- Outer folds: the canonical respondent-grouped 5-fold partition
  (`oof_ensemble_v10.rds$fold_of_case`, seed 4821) -- the same partition used
  by every other component in the project.
- Nested lambda selection for `q` via `cv.glmnet` with `foldid` restricted to
  the 4 outer-training folds.
- Zero respondent overlap between outer-fit and outer-eval asserted by hard
  `stopifnot`.
- Respondent-clustered bootstrap (100,000 replicates) on the per-respondent
  mean gain (v14 loss minus candidate loss), following
  `bootstrap_summary()`/`respondent_gain()` verbatim from
  `codex_two_head_ensemble_v2.R`.

## Pre-registered gates (fixed before any result is seen)

1. **Screen**: the hurdle blend must beat frozen v14's canonical OOF loss
   (1.1435331472708) at all; if not, stop, log negative, do not proceed.
2. **Canonical pass**: `point_gain > 0` and `lower_95 > 0` -> escalate directly
   to repeated CV.
3. **Canonical near-miss** (same threshold `codex_set_context_network_v2.R`
   used): `point_gain > 0` and `lower_95` in `[-0.00075, 0]` -> escalate to
   repeated CV.
4. Otherwise: reject, log, stop -- no repeated CV.
5. **Repeated-CV promotion** (identical rule to every other candidate this
   project has evaluated, including the two-head experiment): pooled
   `point_gain > 0`, pooled `lower_95 > 0`, **and** >=5/6 repeat seeds
   positive, **and** the test-like top-30%-respondent slice's
   `point_gain >= 0` with `lower_95 >= -0.001`.
6. A promoted candidate still requires a full-data build with the project's
   MD5-lock + two-independent-run reproducibility check before any Kaggle
   submission -- no submission is made directly from this script.

## Leakage checklist

- `n_attrs_varying`, price statistics: derived only from the 3 inside
  alternatives' designed attribute/price columns (available at test time, no
  outcome used).
- `q_v14` offset: sourced only from already-verified out-of-fold artifacts,
  never refit on the evaluation row's own respondent.
- Respondent covariates: identical to those already used in mod7/mod8 (no new
  covariate, so no new sparsity risk reopened).
- `cv.glmnet`'s internal `foldid` for the ridge penalty never includes the
  outer-held-out fold.

## What would falsify this

A canonical bootstrap 95% CI that includes zero, OR (if a near-miss triggers
repeated CV) a pooled repeated-CV CI that includes zero, OR fewer than 5/6
positive repeat seeds, OR a failed test-like-30% gate. Per project standard, a
positive point estimate alone does not promote the candidate.
