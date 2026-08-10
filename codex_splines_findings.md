# Smooth-spline covariate interactions (age/mileage/income): one screen pass, fails repeated CV

Pre-registration: `codex_splines_preregister.md`, committed at `410e80d`,
verified via `git log --format="%h %ci %s"` to predate every result below.
Implementation: `R/codex_splines_common.R`, `R/codex_splines_screen.R`,
`R/codex_splines_cv.R`, `R/codex_splines_repeat_cv.R`. Raw output:
`data_processed/codex_splines/`.

## The idea, and what it deliberately does NOT re-test

Current best model (`ensemble_v11 + MLP`, CV 1.143686618 by this round's own
fixed-blend reconstruction / 1.143789 by the project's fold-cross-fitted
headline figure, public LB 1.201) uses **linear** `Price x z(covariate)` and
`inside x z(covariate)` interaction terms for income, age, miles, and night in
its `mlogit_m8trpg` component. This project already tried, and rejected,
replacing those same terms with **categorical/binned** versions
(`cleaning_log.md`, 2026-07-26): all three combined blew validation log loss
up from 1.1657 to 1.1934, driven by `nightind`'s sparsest levels (~6
respondents each) producing a coefficient of -0.815 -- textbook
quasi-separation. Age alone (5 balanced bins) gave a tiny genuine gain;
miles alone (thinner cells) was worse; income was never attempted.

This round tests a **materially different functional-form family**: natural
cubic splines (`splines::ns()`), which place knots at quantiles of the
training respondents' covariate values (comparable data mass per segment by
construction, unlike arbitrary bin edges) and are constrained to extrapolate
**linearly** beyond the boundary knots (no unconstrained-polynomial or
isolated-dummy blow-up risk). Night is out of scope per the brief; only age,
mileage, and income were touched, one covariate at a time, both its Price and
inside interaction terms replaced together, at df in {3, 4} -- 6 candidates,
no others.

## Stage 1: screen (single split, seed 7402)

Baseline (`mlogit_m8trpg` on the cached 908/227 split): **1.15968144721113**
(verified to reproduce `R/codex_triple_interactions.R`'s own hard-coded
value to 8 decimals).

| Candidate | Screen logloss | Delta vs. base | Max \|extra coef\| | Passes screen? |
|---|---:|---:|---:|:---:|
| `age_df3` | 1.163164 | +0.003483 | 0.287 | no |
| `miles_df3` | 1.158258 | **-0.001423** | 1.461 | **yes** |
| `income_df3` | 1.160806 | +0.001124 | 1.224 | no |
| `age_df4` | 1.162651 | +0.002969 | 0.349 | no |
| `miles_df4` | 1.162288 | +0.002606 | 2.340 | no |
| `income_df4` | 1.160777 | +0.001096 | 1.302 | no |

Only **`miles_df3`** beats the baseline on this single split and proceeds to
canonical CV, per the pre-registered screen-then-freeze rule. Notably, age's
spline (both df) makes the single-split screen *worse* even though the
project's earlier, now-superseded binned-age test showed a small gain on an
older, weaker baseline -- consistent with this project's broader pattern of
diminishing/vanishing returns as later terms (price-gap, price-as-factor,
task/region/ppark) already absorb more of the covariate signal. `miles_df4`
also fails despite `miles_df3` passing, suggesting the extra flexibility at
df=4 is overfitting the single screen split rather than reflecting real
additional curvature.

## Stage 2: canonical 5-fold CV (seed 4821), `miles_df3` only

Baseline `mlogit_m8trpg` OOF reused directly from `oof_ensemble_v10.rds`
(verified 1.147021211); `original_xgb`/`shallow_mlp` OOF reused unchanged.

| Fold | Baseline mlogit | Candidate mlogit | Gain |
|---:|---:|---:|---:|
| 1 | 1.208224 | 1.205836 | +0.002388 |
| 2 | 1.124692 | 1.127925 | -0.003233 |
| 3 | 1.145733 | 1.146332 | -0.000599 |
| 4 | 1.118863 | 1.115148 | +0.003715 |
| 5 | 1.137594 | 1.135025 | +0.002569 |
| **Pooled** | **1.147021** | **1.146053** | **+0.000968** |

3 of 5 folds positive (folds 1, 4, 5); folds 2 and 3 negative.

Propagated into the fixed ensemble blend (`0.85*(0.8*mlogit+0.2*xgb)+0.15*mlp`):
current best **1.143686618** -> candidate blend **1.142977974**, gain
**+0.000708644**. Respondent-clustered paired bootstrap (100,000 replicates,
seed 4821): **95% CI [-0.000648, +0.002074]** (99% CI [-0.001067, +0.002511]),
win rate 84.60%.

The canonical CI's upper bound is well above zero (not a decisive rejection),
so per the pre-registered escalation rule this candidate proceeds to repeated
CV before any promotion claim -- exactly the situation the rule anticipated
(a single-split screen pass plus a favorable but not-yet-decisive canonical
CV number has misled this project before, most recently the eight-component
blend, whose canonical CI also excluded zero and then reversed under repeated
CV).

## Stage 3: repeated CV (6 fold-seed assignments: canonical 4821 + 1907/2719/6151/8293/104729)

Reused the project's own cached `original_xgb`/`shallow_mlp`/baseline-`mlogit`
components per seed/fold from `data_processed/codex_repeat_cv/checkpoints/`
(verified: fold-1 validation respondent sets for all 5 additional seeds match
this round's own `repeat_fold_map()` reconstruction exactly before trusting
anything downstream). Refit only the `miles_df3` spline mlogit component --
25 new fits (5 seeds x 5 folds) on top of the 5 already computed at Stage 2.

| Repeat seed | Mean per-respondent gain |
|---:|---:|
| 4821 (canonical) | +0.000709 |
| 1907 | +0.000476 |
| 2719 | +0.000604 |
| 6151 | +0.000778 |
| 8293 | +0.000459 |
| 104729 | **-0.000914** |

5 of 6 repeats positive; the pooled per-respondent gain (averaging the 6
repeats within each of the 1,135 respondents first, then bootstrapping)
gives:

**Point gain +0.000352005. Ordinary 95% respondent-bootstrap CI
[-0.000976849, +0.001684114] -- crosses zero.** (99% CI [-0.001398,
+0.002094], win rate 69.67%, 100,000 replicates, seed 4821.)

Full per-seed, per-fold gains and spline-coefficient magnitudes:
`data_processed/codex_splines/splines_repeat_cv_folds.csv`. Coefficient
magnitudes across all 30 fold fits stayed in a stable, dense range
(~0.64-1.64 in the canonical folds; comparable across repeats) -- no
isolated blow-up like the rejected binned version's single -0.815 coefficient
from a ~6-respondent cell. This confirms the mechanism-level claim in the
pre-registration: the spline's quantile-based knots and linear-extrapolation
constraint genuinely avoid the categorical treatment's sparse-cell failure
mode. The candidate is simply not confirmably better, not unstable in the
way the rejected version was.

## Decision

Per the pre-registered binding promotion rule -- **the ordinary 95%
respondent-bootstrap CI must exclude zero gain vs. the current best** -- the
repeated-CV CI `[-0.000976849, +0.001684114]` crosses zero. **`miles_df3` is
not promoted.** (`data_processed/codex_splines/splines_repeat_cv_decision.csv`:
`promote = FALSE`.)

The other five candidates (`age_df3`, `age_df4`, `income_df3`, `income_df4`,
`miles_df4`) never reached CV at all: they failed the single-split screen
outright, a clean Stage-1 null for each.

## Interpretation

A genuinely smooth, materially different functional-form test of this
project's already-rejected binned-covariate finding -- and it is a genuinely
different result, not a re-run: unlike the categorical treatment, no
candidate here showed anything resembling quasi-separation or a runaway
coefficient, and one candidate (`miles_df3`) even passed the single-split
screen and looked promising through canonical CV (84.6% bootstrap win rate).
But that is exactly the profile this project's own repeated-CV discipline
exists to catch: a favorable canonical-CV draw that does not survive contact
with 5 additional independent fold assignments. The pooled point estimate
survives at roughly half its canonical size (+0.000708 -> +0.000352), and
5/6 repeats stay directionally positive, but the added respondent-level noise
from genuinely re-randomizing the fold assignment is enough to pull the CI
back across zero -- the same qualitative story as the eight-component blend
and the price-history near-miss earlier in this project's search.

Net conclusion for the report: smooth nonlinearity in age/mileage/income's
Price and inside-good interactions is **not** a source of exploitable
additional signal beyond what m8trpg's existing linear covariate
interactions, segment interactions, and price-context terms already capture
-- consistent with, and now extending, this project's broader finding that
the returns to additional covariate-interaction flexibility (categorical
bins, continuous triple products, latent-class price-sensitivity scales,
partial pooling, SHAP-guided interactions) have been repeatedly tested and
have repeatedly come back null or too small to distinguish from this
dataset's respondent-level noise floor once evaluated honestly.

**Not adopted; no submission made; no changes proposed to the current best
model.**

## Reproduction

```powershell
& 'C:\Program Files\R\R-4.6.0\bin\Rscript.exe' R/codex_splines_screen.R
& 'C:\Program Files\R\R-4.6.0\bin\Rscript.exe' R/codex_splines_cv.R
& 'C:\Program Files\R\R-4.6.0\bin\Rscript.exe' R/codex_splines_repeat_cv.R
```

All three scripts read only `csv files/train.csv`,
`data_processed/train_val_split.rds`, `data_processed/oof_ensemble_v10.rds`,
`data_processed/codex_behavioral_round/mlp_oof.rds`, and
`data_processed/codex_repeat_cv/checkpoints/*.rds` (all already present in
this project's local, gitignored `data_processed/`); no new caches beyond
`data_processed/codex_splines/` are required or created elsewhere.
