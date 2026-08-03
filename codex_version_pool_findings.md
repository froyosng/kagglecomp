# Neighbor-pooled questionnaire-version opt-out correction: findings

Branch: `codex-version-pool-smoothing`. Pre-registration committed at `b2d3cad`
(2026-07-29 12:17:15 +0800, see `codex_version_pool_preregister.md`) before any grid
search, smoke test, or bootstrap ran; this document reports what actually happened
when that pre-registered plan was executed unchanged.

## Verdict

**Reject. Not promoted. No submission file generated, no Kaggle submission made.**
The pooled respondent-clustered 95% bootstrap CI vs. the current best model
(`ensemble_v11+MLP`, canonical CV `1.143686618134879`) is **`[-0.000828, +0.000793]`**,
which crosses zero by a wide margin (point estimate `-0.0000231`, i.e. very slightly
worse than the baseline; win rate across 100,000 bootstrap replicates `47.4%`, i.e.
under half). The promotion bar (95% CI lower bound `> 0`) is not cleared, and it is not
close.

More importantly, a mechanism check (Section 4 below) shows the correction does not do
what it was designed to do: it does not rescue the zero/one-peer versions that sank the
original rejected estimator. If anything it makes those specific respondents worse off,
while providing a small, still-not-significant benefit to respondents whose versions
already had adequate peer support. This is a more informative null than "the CI crossed
zero" -- it is evidence the specific similarity metric tried here does not carry
transferable opt-out signal across questionnaire versions, not just that the sample is
too small to see it.

## 1. What was implemented (recap; full spec in `codex_version_pool_preregister.md`)

For version `v`, instead of the rejected estimator's isolated
`delta_v = -g_v/(h_v+lambda)`, this experiment pools `g`/`h` across a k-nearest-neighbor
Gaussian-kernel graph built from a 20-dimensional, design-only similarity feature (mean
level of each of the 19 attributes + Price, across a version's 19 tasks x 3 real
alternatives -- identical for every respondent sharing a version, by the fingerprint's
own construction, so no leakage risk and no fold-dependence):

```
G_v = sum_u w(v,u) * g_u,   H_v = sum_u w(v,u) * h_u,   delta_v = -G_v / (H_v + lambda)
```

`k = 0` (no neighbors, self-weight only) collapses this exactly to the rejected
estimator -- a deliberate built-in continuity/correctness check (see Section 3).
Selection of `(k, lambda)` from a pre-registered `5 x 8 = 40`-combination grid
(`k in {0,5,10,20,40}`, `lambda in {5,10,20,40,80,160,320,Inf}`) happens per outer fold
via genuine nested inner-OOF CV, with a leave-one-respondent-out exclusion at the
correction-estimation step itself (Section 4 of the preregister), exactly mirroring the
rejected experiment's own anti-leakage rigor.

**Reused, read-only, independently re-verified:** the already-audited nested
inner-OOF/outer-refit prediction matrices from the rejected experiment
(`data_processed/codex_version_shrinkage/outer{f}_{inner{g},final}_base_fit.rds`,
branch `codex-version-shrinkage` commit `356baf5`), instead of re-running ~20 expensive
frozen-architecture (mlogit `m8trpg` + xgboost + 5-seed MLP) refits. Before trusting
them, `R/codex_version_pool_inspect.R` and `R/codex_version_pool_inspect2.R`
independently re-confirmed, for all 5 outer folds: the outer-holdout respondent set
exactly equals `which(fold_of_case==f)`; zero case overlap between that holdout and any
inner/outer source or target; the 4 inner target sets partition the 908-respondent
outer-training set with no gaps or duplicates. All checks passed for every fold.

## 2. Smoke test (Section 7 of the preregister) -- passed

1. **Neighbor graph sanity.** `k=0` produces the exact 299x299 identity matrix. For
   `k in {5,10,20,40}`, every version gets exactly `k` off-diagonal neighbors with
   weights in `(0,1]`, non-increasing in distance, self-weight fixed at `1`. All checks
   passed (`R/codex_version_pool_smoketest.R`).
2. **`k=0` reproduces the rejected experiment.** Running the full nested pipeline with
   the grid restricted to `k=0` (i.e. exactly the rejected estimator, computed by
   entirely new code) reproduced the previously published numbers almost bit-for-bit:

   | Outer fold | Published lambda | Reproduced lambda | Published gain | Reproduced gain |
   |---:|---:|---:|---:|---:|
   | 1 | 160 | 160 | +0.0000865 | +0.0000865 |
   | 2 | Inf | Inf | 0 | 0 |
   | 3 | Inf | Inf | 0 | 0 |
   | 4 | 160 | 160 | -0.0000591 | -0.0000591 |
   | 5 | 160 | 160 | -0.0008852 | -0.0008852 |

   Pooled gain: published `-0.000172` vs. reproduced `-0.0001716` (max per-fold
   deviation `2.6e-08`, all lambda selections identical). This is strong independent
   confirmation that the nested structure, the leave-one-out exclusion, and the
   baseline reconstruction are implemented correctly before any `k>0` result is
   trusted. Output: `data_processed/codex_version_pool/smoketest_k0_*`.

## 3. Full grid result (`k in {0,5,10,20,40}`)

Per outer fold (`data_processed/codex_version_pool/full_fold_summary.csv`):

| Outer fold | k* | lambda* | Primary gain (vs. exact submitted OOF) | Secondary gain (fresh refit) |
|---:|---:|---:|---:|---:|
| 1 | 0 | 160 | +0.0000865 | +0.0000714 |
| 2 | 0 | Inf | 0 | 0 |
| 3 | **40** | **320** | **+0.0007422** | +0.0007151 |
| 4 | 0 | 160 | -0.0000591 | -0.0000421 |
| 5 | 0 | 160 | -0.0008852 | -0.0008746 |

Four of five folds selected `k=0` -- i.e. inner CV found no benefit from neighbor
pooling and fell back to the (already-rejected) isolated estimator, reproducing its
per-fold numbers exactly. Only fold 3 selected a non-trivial neighborhood (`k=40,
lambda=320`), moving that fold's gain from `0` (its `k=0` value) to `+0.000742`.

**Pooled outer result** (`data_processed/codex_version_pool/full_overall.csv`,
`full_bootstrap_primary.csv`):

- Primary baseline (exact submitted OOF): `1.143686618`
- Primary corrected: `1.143709722`
- **Point gain: `-0.0000231`**
- **95% respondent-bootstrap CI (100,000 replicates, seed 4821): `[-0.0008282, +0.0007927]`**
- 99% CI: `[-0.0010787, +0.0010571]`
- Win rate: `47.4%`
- Secondary (freshly-refit) baseline comparison: gain `-0.0000260`, 95% CI
  `[-0.0008300, +0.0007888]` -- consistent with the primary result.

**Comparison to the `k=0`-only (rejected-estimator) run inside this same pipeline:**

| | Point gain | 95% CI |
|---|---:|---:|
| `k=0` only (replicates rejected estimator) | -0.0001716 | [-0.0007996, +0.0004576] |
| Full `k` grid (this experiment) | -0.0000231 | [-0.0008282, +0.0007927] |

Allowing neighbor pooling moved the point estimate closer to zero (less bad) but made
the confidence interval *wider*, not narrower -- consistent with the extra `k`
dimension adding one fold's worth of larger, more volatile corrections (fold 3) rather
than adding robust, broadly-supported signal. Either way, both intervals cross zero by
a comparable margin; neither clears the promotion bar.

**Is fold 3's improvement a real dose-response in neighbor count, or an isolated
spike?** The inner-loss grid for fold 3 (`data_processed/codex_version_pool/full_grid_all.csv`)
shows: `k=5` and `k=20` (with their best lambda) give **zero** inner improvement over
baseline (same as `k=0`); only `k=10` (small, `+0.000104` inner gain) and `k=40`
(`+0.000260` inner gain) show anything. A genuine smoothing effect would be expected to
improve roughly monotonically as more neighbors are pooled in; instead the pattern is
erratic (0, then non-zero, then 0, then largest at the far end of the grid), which
looks more like one specific combination out of 40 fitting fold 3's inner data somewhat
by chance than a robust, broadly-supported neighborhood effect.

## 4. Mechanism check: does pooling rescue the respondents it was designed for?

The rejected estimator failed specifically because zero/one-peer versions'
`g_v`/`h_v` were dominated by a single respondent's own outcome. This experiment's
entire premise is that neighbor pooling should rescue exactly those cases. Splitting
the full-grid result's per-respondent gain (`data_processed/codex_version_pool/mechanism_check_by_respondent.csv`)
by each respondent's own-version training-peer count:

| Peer band | n respondents | Mean per-respondent log-loss gain |
|---|---:|---:|
| 0 peers | 53 | **-0.001325** |
| 1 peer | 189 | **-0.000919** |
| 2-3 peers | 510 | +0.000260 |
| 4+ peers | 383 | +0.000222 |

This is the opposite of the intended mechanism. The respondents whose versions had no
or one training peer -- the ones the rejected estimator could not help and the ones
this experiment specifically targets -- come out **worse**, on average, than doing
nothing. Respondents whose versions already had 2 or more peers (i.e. the cases where
the original, unpooled estimator was least likely to be dominated by a single
respondent) show a small positive average gain, but it is the population where the
correction was least needed in the first place, and the overall bootstrap already shows
this small benefit is not statistically distinguishable from zero.

Restricting to fold 3 alone (the one fold that selected `k>0`) shows the same pattern
even more sharply: 0-peer respondents there average **-0.00585**, 1-peer respondents
average **-0.00156**, while 2-3-peer and 4+-peer respondents average **+0.00150** and
**+0.00193** respectively. The other four folds (all selected `k=0`, pure replication of
the rejected estimator) show the expected near-zero/negative pattern for low-peer
respondents and small negative numbers elsewhere, consistent with the original
findings.

**Interpretation.** Borrowing `g`/`h` mass from design-similar neighbor versions does
not supply *useful* information about a low-peer version's opt-out propensity -- it
appears to import noise (or a systematic pattern uncorrelated with the true one) for
exactly the versions with the least reliable own-signal to begin with, while acting as
a mild, roughly neutral-to-positive extra regularizer for versions that already had
enough peers to estimate reasonably well on their own. This is direct diagnostic
evidence, not just an underpowered-CI shrug, that mean design-level attribute/price
exposure similarity between two different CBC questionnaire versions is not a
meaningful proxy for whether their respondents' opt-out behavior should resemble each
other.

## 5. Honest assessment (per the parent brief's explicit request)

This was pre-registered and flagged, correctly, as the weaker-motivated of the
remaining leads: 299 questionnaire versions are an arbitrary, fixed CBC design draw
from a survey tool, and there was no strong a priori reason one version's respondents'
opt-out behavior should be informative about a different version's, beyond whatever
mechanical similarity their shown attribute/price profiles happen to share. The
experiment gave that hypothesis a genuine, honestly-executed chance: a real
similarity metric, a real kernel smoother, a grid wide enough to include both
"pool a little" (`k=5,10`) and "pool a lot" (`k=40`), and a nested CV / leave-one-out
structure independently verified (smoke test) to reproduce the prior rejected result
exactly before trusting anything new. The result is a clean null, with a mechanistic
explanation (Section 4) for *why* it is null, not just *that* it is. This is reported
as-is; the modestly-less-negative point estimate on the full grid vs. the `k=0`-only
run is not being reframed as a partial win -- the CI is if anything wider, and the
per-respondent breakdown shows the effect is going the wrong direction for the
population it was meant to help.

**What was not tried, and remains open:** the covariate-composition similarity basis
(respondent segment/income/age/etc. composition of a version's assignees) mentioned as
an alternative in the parent brief was not tested -- it was set aside at
pre-registration time specifically because it would need per-fold recomputation
(a version's inner-training composition changes fold to fold), adding another
leakage-prone moving part, and the design-marginal basis was judged simpler and
equally principled a priori. Given this run's finding that the *design*-based
similarity is not informative, whether a *covariate*-composition-based similarity would
behave differently is a genuinely open question this experiment does not answer either
way. No version-specific scale, cheapest-inside shift, or full alternative-intercept
extension was attempted, consistent with the parent brief's scope.

## 6. Reproducibility

- `codex_version_pool_preregister.md` -- committed `b2d3cad`, before any run.
- `R/codex_version_pool_common.R` -- shared functions (similarity matrix, kNN kernel,
  pooled Newton statistics, leave-one-out exclusion, bootstrap).
- `R/codex_version_pool_driver.R` -- `run_version_pool_cv(k_grid, lambda_grid, ...)`,
  the single nested-CV code path used by both the smoke test and the full run.
- `R/codex_version_pool_smoketest.R` -- Section 2 above; must pass before the full run
  is trusted (it does).
- `R/codex_version_pool_cv.R` -- the full run (Section 3).
- `R/codex_version_pool_diagnostics.R` -- the fold-3 grid-sensitivity check.
- `R/codex_version_pool_mechanism_check.R` -- Section 4's peer-band breakdown.
- `R/codex_version_pool_inspect.R`, `R/codex_version_pool_inspect2.R` -- read-only
  artifact/leakage re-verification, run before pre-registration.
- Generated outputs: `data_processed/codex_version_pool/*.csv`, `*_result.rds`
  (gitignored; regenerate by re-running the scripts above after copying
  `csv files/`, `data_processed/oof_ensemble_v10.rds`,
  `data_processed/questionnaire_fingerprints.rds`,
  `data_processed/codex_behavioral_round/mlp_oof.rds`, and
  `data_processed/codex_version_shrinkage/outer*_{inner*,final}_base_fit.rds` from the
  primary worktree, as this branch's own commits do not carry raw/derived data).

No `AGENTS.md`, `cleaning_log.md`, or `submissions_log.csv` edits were made.
