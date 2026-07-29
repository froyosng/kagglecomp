# Pre-registration: smooth-spline covariate interactions (age, mileage, income)

Independent worker track (isolated worktree/branch, not `zhenhao`). Fixed
before any screen/CV fit is run, on top of `zhenhao` commit `dfa514d` (merged
into this branch only, never pushed back to `zhenhao`).

## What this project has already found (read in full before writing this)

**Current best model**, per `AGENTS.md`/`cleaning_log.md`/`submissions_log.csv`:
`ensemble_v11 + MLP` = `0.85 * (0.80 * mlogit_m8trpg + 0.20 * original_xgb) +
0.15 * shallow_mlp`. `mlogit_m8trpg` is a conditional logit (`mlogit`, no ASCs)
with all 19 non-price attributes factor-coded, Price as a 12-level factor
(`Pr_lvl2..12`), alt2/alt3 position dummies, task-fatigue (`P_task`/`In_task`),
segment/region/ppark interactions, and -- centrally for this experiment --
**linear** `Price x z(covariate)` and `inside x z(covariate)` interaction terms
for income, age, miles, night (`P_income`/`In_income`, `P_age`/`In_age`,
`P_miles`/`In_miles`, `P_night`/`In_night`), plus rank/magnitude price-context
terms (`is_cheapest`/`is_dearest`/`price_gap_min`/`price_gap_max`). Verified by
reconstructing this exact model from the project's own cached artifacts (see
"Verification" below): canonical 5-fold CV log loss reproduces to 9 decimal
places, both for `mlogit_m8trpg` alone (1.147021211) and for the full current
best blend (1.143686618, matching `codex_price_history_only.R`'s and
`codex_shared_utility_mlp`'s own `current_canonical` reconstruction bit-for-bit
-- this is the exact number every other recent pre-registered round in this
project has used as "the current best" for its own promotion decision, and is
what this round uses too, even though the headline table in `AGENTS.md` quotes
a very slightly different figure, 1.143789, from a fold-cross-fitted rather
than fixed MLP blend weight -- both refer to the same submitted model,
`submission_codex_mlp_v12_candidate.csv`, public LB 1.201).

**The specific already-rejected finding this hypothesis must NOT re-test**
(cleaning_log.md, "2026-07-26: Extending the price-context idea further"):
the project tried replacing the continuous `Price x {age, miles, night}`
interactions with **categorical/binned** versions (dummy variables per
`ageind`/`milesind`/`nightind` level, the same trick that worked for turning
Price itself from a linear slope into a 12-level factor). Result: mostly
negative, and dangerous in one case. All three binned covariates combined blew
validation log loss up from 1.1657 to 1.1934, driven by `nightind` levels 9-10
(only ~133 and ~114 rows, roughly 6 respondents each) producing a coefficient
of -0.815 -- classic quasi-separation from sparse discrete cells. Tested
individually: age alone (5 *balanced* levels, smallest cell ~2,070 rows) gave
a small genuine gain (1.16495 vs 1.16573); miles alone (9 levels, some thin
categories) was worse (1.16814); income (25 levels) was never even attempted
given its sparsity risk is worse than night's. The project's own conclusion:
"the continuous-to-categorical trick that worked so well for Price does not
generalize automatically -- it depends on per-cell sample size, must be
checked category-by-category."

**Why a natural cubic spline is a genuinely different functional-form test,
not a re-run of that finding:** the binned treatment's failure mode was
*discrete, arbitrary-width cells* with no guarantee of comparable sample size
per cell (nightind's top two levels had ~6 respondents each). `splines::ns()`
instead (a) places knots at *quantiles* of the training respondents'
covariate values, so every spline segment sees comparable data mass by
construction, and (b) is constrained to be *linear* beyond the boundary knots
(the defining property of a *natural* spline), so extreme values extrapolate
smoothly and boundedly rather than risking an unconstrained polynomial or an
isolated dummy coefficient blowing up. This is the same reasoning this project
already used to distinguish the (adopted) continuous triple-interaction
products from the (rejected) binned covariate interactions -- "a fundamentally
different risk profile, since there's no discrete cell to be sparse in" -- and
it has never been tested for a single-covariate smooth nonlinear form, only
for products of two z-scored linear terms.

## Candidate family (fixed now, 6 configs, no others)

For each of the three covariates named in the brief -- age (`agea`), mileage
(`milesa`), income (`incomea`) -- and for spline degrees of freedom in
`{3, 4}` (2 boundary knots + 1 interior knot at the median for df=3; 2
boundary knots + 2 interior knots at terciles for df=4, `splines::ns()`
defaults): replace **both** the linear `P_<covariate>` and `In_<covariate>`
terms with `df` spline-basis columns each, interacted the same way
(`P_<covariate>_ns{1..df} = Price_num * ns_basis_j`, `In_<covariate>_ns{1..df}
= inside * ns_basis_j`). Every other m8trpg term is untouched, including the
OTHER two covariates' linear interactions and `P_night`/`In_night` (night is
explicitly out of scope for this experiment per the brief -- the "sparse
night-driving" bin failure is cited as background, not re-tested).

| Candidate | Covariate | df | Params added | Params removed |
|---|---|---:|---:|---:|
| `age_df3` | age | 3 | 6 | 2 |
| `age_df4` | age | 4 | 8 | 2 |
| `miles_df3` | miles | 3 | 6 | 2 |
| `miles_df4` | miles | 4 | 8 | 2 |
| `income_df3` | income | 3 | 6 | 2 |
| `income_df4` | income | 4 | 8 | 2 |

Both interaction directions (Price and inside) are replaced together per
covariate, because the hypothesis is about that covariate's whole functional
form, not just one of its two roles in the model; this also keeps the family
small (6, not 12) and matches this project's general preference for compact
pre-registered families (`price_only`: 3; `shared_utility_mlp`: 3).

Spline knots are fit **only on the training fold's respondents** in every
context below (single-split training subset; each canonical CV fold's 908
training respondents; each repeated-CV fold's training respondents) via
`splines::ns(x, df)` on de-duplicated per-respondent covariate values, then
applied to validation/held-out data via `predict(ns_obj, newx)` -- the same
fold-safety discipline this project already uses for its z-score scaler
(`choice_scaler()`), just carried over to the spline basis. Implementation:
`R/codex_splines_common.R` (`fit_ns_basis`, `add_spline_terms`,
`spline_formula`, `fit_predict_spline`).

## Stage 1 -- Screen (single split, seed 7402, screen-then-freeze)

Reuses the project's existing single-split validation object,
`data_processed/train_val_split.rds` (908/227 respondent split, seed 7402),
unmodified. Verified this round's plumbing reproduces the project's own
cached m8trpg baseline screen number exactly: refitting `mlogit_m8trpg` on
`train_long_tr`/predicting on `train_long_val` gives 1.15968144721113,
matching the value hard-coded in `R/codex_triple_interactions.R`'s own
`stopifnot`. Screening baseline for this round is therefore this exact number,
computed by this round's own code from the same cached split.

Each of the 6 candidates is fit once (mlogit component alone, no ensemble
blend) on the training subset and scored on the held-out 227 respondents. Per
this project's established screen-then-freeze rule (`R/codex_triple_interactions.R`'s
`stage == "screen"`/`"cv"` split, `codex_shared_utility_mlp_preregister.md`):
**only candidates whose single-split validation log loss is strictly lower
than 1.15968144721113 proceed to canonical CV.** No candidate is modified
after seeing its screen number (no df re-tuning, no knot-placement changes);
a failed screen is a clean, reportable null for that candidate, not grounds
for a variant.

## Stage 2 -- Canonical 5-fold CV (only for candidates that pass Stage 1)

- Fold assignment: `data_processed/oof_ensemble_v10.rds$fold_of_case` (seed
  4821, verified 227/227/227/227/227 over the 1,135 train respondents) --
  reused directly, not regenerated.
- Every surviving candidate's mlogit component is refit from scratch inside
  each of the 5 folds (fresh z-score scaler AND fresh spline knots from that
  fold's 908 training respondents only; predicted on the other 227).
- `original_xgb` and `shallow_mlp` are **not** refit -- reused byte-for-byte
  from `data_processed/oof_ensemble_v10.rds$oof_xgb` and
  `data_processed/codex_behavioral_round/mlp_oof.rds$oof[["h08_d0.100"]]`,
  matching this project's established "only the mlogit component under test
  changes" pattern (identical fold assignment, identical xgboost/MLP specs,
  so this is exact reuse, not an approximation).
- Ensemble architecture (fixed, matches the current best exactly):
  `candidate = 0.85 * (0.80 * candidate_mlogit + 0.20 * original_xgb) + 0.15 *
  shallow_mlp`, compared against `baseline = 0.85 * (0.80 * mlogit_m8trpg +
  0.20 * original_xgb) + 0.15 * shallow_mlp` (`mlogit_m8trpg`'s OOF also reused
  directly from `oof_ensemble_v10.rds$oof_mlogit`, already verified above to
  reproduce 1.147021211 -- no need to refit the baseline mlogit, since it is
  already cached and confirmed correct). No blend-weight re-optimization.

## Estimand and bootstrap

Per respondent, the mean row-level log-loss gain of `candidate` over
`baseline` (positive = candidate better), pooled across that respondent's 19
tasks x 4 alternatives. Respondent-clustered paired bootstrap: resample the
1,135 respondents with replacement, 100,000 replicates, seed 4821 (matching
this project's established convention, `R/codex_repeat_cv_common.R`'s
`repeat_bootstrap_average_gain`). Report point gain, ordinary 95%/99% CI, and
win rate, for every surviving candidate.

## Promotion rule (binding, fixed before any CV result is seen)

**A candidate is promoted only if the ordinary 95% respondent-bootstrap CI
(canonical CV, or repeated CV if escalated per the rule below) excludes zero
gain vs. the current best.** A positive point estimate alone is not
sufficient -- this project has already correctly rejected multiple
positive-point-estimate candidates whose CI crossed zero (the eight-component
blend, the price-history near-miss, the shared-alternative-utility MLP), so
the same standard applies here with no exception.

## Escalation rule: when to run repeated CV

This project's own history shows that *even* a canonical-CV CI that excludes
zero has not been sufficient for confident adoption on its own (the
eight-component blend's canonical CI barely excluded zero,
`[0.0000466, 0.0032790]`, and reversed under repeated CV). Accordingly, for
every candidate that is not a **decisive** canonical-CV rejection, this round
escalates to repeated CV before making any promotion claim, using the same six
fold-seed assignments already established in this project
(`4821` canonical + `1907, 2719, 6151, 8293, 104729`), reusing the cached
`original_xgb`/`shallow_mlp`/baseline-`mlogit` components already saved per
seed/fold in `data_processed/codex_repeat_cv/checkpoints/seed_<seed>_fold_<fold>.rds`
and refitting only the candidate spline mlogit component for each of the 30
seed x fold combinations (5 already computed at Stage 2 for the canonical
seed, 25 new fits for the 5 additional seeds).

Concretely: a canonical-CV result is treated as a **decisive rejection**
(repeated CV skipped, reported as a clean single-stage null) only if the
ordinary 95% CI's upper bound is comfortably below zero (candidate clearly
worse, no ambiguity). Every other outcome -- CI excludes zero on the positive
side (by any margin), or CI crosses zero with a positive or near-zero point
estimate -- is treated as needing the repeated-CV check before any claim is
made, exactly mirroring how this project has actually treated every real
near-miss and every apparent single-split win so far. The final promotion
decision always uses the ordinary 95% CI from whichever stage
(canonical-only, if decisively rejected, or repeated) is the last one run for
that candidate.

For context only (not part of the binding promotion rule above, but reported
for comparability with the rest of the project's repeated-CV write-ups): for
any candidate that reaches repeated CV, also report (a) the count of the 6
repeat-level mean gains that are positive, and (b) the sign of the relevant
spline coefficients' aggregate direction across all 30 fold fits, as
descriptive diagnostics.

## Explicitly excluded from this round

No re-test of the binned/categorical covariate treatment (already closed, see
above). No touching `P_night`/`In_night` (out of scope per the brief). No new
df values beyond `{3, 4}`. No re-optimization of the 0.80/0.20 or 0.85/0.15
ensemble blend weights. No Kaggle submission, no test-label access, no
leaderboard probing -- this is a CV-only research track; any submission
decision is for the coordinating session to make after independently
reviewing this evidence.

## Verification checks run before this file was committed

Run directly from this round's own scripts against the project's cached
artifacts, all passing to the stated precision (see
`R/codex_splines_common.R` and the verification script referenced in
`codex_splines_findings.md`):
- `oof_ensemble_v10.rds$fold_of_case`: 227/227/227/227/227 over 1,135 cases.
- `oof_ensemble_v10.rds$oof_mlogit` canonical CV log loss: 1.147021211.
- `0.80 * oof_mlogit + 0.20 * oof_xgb` (ensemble_v11): 1.145094213.
- `0.85 * ensemble_v11 + 0.15 * shallow_mlp` (current best, fixed-blend
  reconstruction): 1.143686618 (matches `codex_price_history_only.R`'s
  hard-coded `1.143686618134879` to 9 decimals).
- Single-split (seed 7402) `mlogit_m8trpg` baseline: 1.15968144721113 (matches
  `R/codex_triple_interactions.R`'s hard-coded value).
- Smoke test of `R/codex_splines_common.R` on a 120-respondent (90 train / 30
  validation) toy subset: all 6 candidates fit without error, produce
  finite, row-normalized, NA-free predictions correctly aligned back to `No`,
  and the spline basis extrapolates without NA/Inf on held-out respondents
  outside the training knot range.
