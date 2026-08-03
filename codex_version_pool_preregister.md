# Pre-registration: neighbor-pooled (kernel-smoothed) questionnaire-version opt-out correction

Branch: `codex-version-pool-smoothing`. Isolated worktree, based on primary-worktree
`zhenhao` commit `dfa514d` (data/artifacts copied in; code history inherited from the
repo's shared commit graph, including `codex-overnight-queue`/`codex-version-shrinkage`
which this experiment builds directly on).

This document is committed **before** any grid search, bootstrap, or outer-fold
evaluation is run. The only things that ran before this commit were read-only
inspection scripts (`R/codex_version_pool_inspect.R`, `R/codex_version_pool_inspect2.R`)
that print artifact shapes and re-verify the rejected experiment's own partition/leakage
claims (all confirmed, see "Artifacts reused and re-verified" below) -- no candidate
correction, grid search, or bootstrap has been computed yet.

## 1. What this restates (to confirm the prior logs were actually read)

**Current best model ("ensemble_v11+MLP").** `0.85 * (0.80 * mlogit_m8trpg_OOF + 0.20 *
xgboost_OOF) + 0.15 * shallow_MLP_OOF`, where the MLP OOF is a 5-seed average of an
8-hidden-unit, decay-0.1 `nnet` fit (`codex_behavioral_round/mlp_oof.rds`,
candidate `h08_d0.100`), and the mlogit/xgboost OOF come from `oof_ensemble_v10.rds`.
Canonical respondent-grouped 5-fold CV (seed 4821, `fold_of_case`) log loss on the
exact submitted out-of-fold matrix: **1.143686618134879**. Public leaderboard: 1.201.
This is the model every comparison in this file is measured against.

**The rejected estimator (`version_newton_optout_correction`, tested on branch
`codex-version-shrinkage` commits `356baf5`/`44d8395`/`1aeea88`/`031260f`, folded into
`codex-overnight-queue` at `798662c`).** For each of the 299 questionnaire versions
(fingerprinted from a respondent's complete ordered 19-task design sequence, attributes
+ price only, never the choice outcome -- `data_processed/questionnaire_fingerprints.rds`),
using **inner-OOF base-model probabilities only**:

```
g_v = sum_{i in v}(p4_i - y4_i)        # opt-out gradient
h_v = sum_{i in v} p4_i * (1 - p4_i)   # opt-out curvature
delta_v = -g_v / (h_v + lambda)        # Newton step on alt-4's log-utility
```

`lambda` was chosen from `{5,10,20,40,80,160,320,Inf}` by nested inner CV, independently
per outer fold, with a genuine 4-inner-fold refit of the frozen architecture supplying
inner-OOF predictions for every outer-training respondent (never in-sample fits), plus a
further leave-one-respondent-out exclusion at the *correction-estimation* step itself (so
a respondent's own residual never contributes to the delta later used to correct that
same respondent). This is a strict, doubly-nested, audited design.

**Why it failed.** The pooled respondent-clustered bootstrap gain vs. the exact submitted
OOF was **negative** on both the exact-OOF and a freshly-refit outer pipeline
(`-0.000172`, 95% CI `[-0.000800, +0.000458]`; `-0.000169`, CI `[-0.000797,+0.000460]`),
and stayed negative/CI-crossing-zero even after excluding one-peer versions
(`-0.000068`, CI `[-0.000679,+0.000538]`). Root cause confirmed by diagnostics: each
version is shared by only ~3-5 respondents on average, and a held-out respondent has
on average only ~2.8-3.0 *training* peers **in their own version**; 7-15 of 227 held-out
respondents per fold had **zero** training peers (received no correction at all) and
36-41 had exactly **one** peer, and among versions with >=2 peers, one respondent
supplied more than half the summed absolute gradient in ~50-55% of cases (median
dominance 0.707 at exactly 2 peers) -- i.e. the "correction" for many versions was
really just one person's own outcome being reflected back, not a genuine shared signal,
which is exactly why it added variance without adding real information.

**How my approach differs.** The rejected method treated every version's `g_v, h_v` as
resting entirely on that version's own (tiny, often single-respondent) peer group,
shrunk only toward a *global* zero correction when peers were scarce. This experiment
instead lets a version borrow `g`/`h` mass from **other, design-similar versions** before
computing its Newton step -- a version with zero or one own-peer can still receive a
non-trivial, non-zero correction if its design neighbors have usable signal, rather than
falling back straight to zero. Setting the neighbor count `k=0` in the grid below
reproduces the rejected estimator *exactly*, byte-for-formula, and is included as a
built-in correctness/continuity check.

**Honesty flag, stated up front.** Per the parent brief, this is explicitly the
weaker-motivated of the remaining leads: 299 questionnaire versions are an arbitrary,
fixed CBC design draw from a survey tool, and there is no strong a priori reason one
version's respondents' opt-out behavior should be informative about a *different*
version's respondents, beyond whatever the two versions' shown attribute/price profiles
mechanically have in common. A null result here (correction fails to clear the
promotion bar) is a legitimate, expected, useful possible outcome and will be reported
as such -- it will not be reframed as a partial win.

## 2. Artifacts reused and re-verified (not re-derived)

To avoid re-running ~20 expensive frozen-architecture refits (mlogit `m8trpg` +
xgboost + 5-seed MLP, per inner/outer split) that a different worker already produced
and audited under the identical frozen architecture and identical nested design, this
experiment reuses, read-only:

- `data_processed/oof_ensemble_v10.rds` -- canonical `fold_of_case` (seed 4821, 227/227/
  227/227/227), `oof_mlogit`, `oof_xgb`, `oof_truth`.
- `data_processed/codex_behavioral_round/mlp_oof.rds` -- the `h08_d0.100` MLP OOF matrix.
- `data_processed/questionnaire_fingerprints.rds` -- 299-version fingerprints, `Case` 1-1398.
- `data_processed/codex_version_shrinkage/outer{1..5}_final_base_fit.rds` and
  `outer{f}_inner{g}_base_fit.rds` (`g` = the other 4 canonical folds) -- the cached
  nested inner-OOF and outer-refit predictions from the frozen `v11+MLP` architecture,
  produced by the rejected experiment (branch `codex-version-shrinkage`, commit
  `356baf5`), copied into this worktree's `data_processed/codex_version_shrinkage/`.

**Independent re-verification performed before trusting these** (`R/codex_version_pool_inspect.R`,
`R/codex_version_pool_inspect2.R`, both read-only, outputs not used as results):
for every outer fold `f in 1..5`, (a) `outer{f}_final_base_fit.rds$target_case` exactly
equals `which(fold_of_case == f)` (227 respondents), (b) zero case overlap between that
holdout set and every inner fit's source/target, (c) the union of the 4 inner fits'
target respondents exactly equals the 908-respondent outer-training set with no
duplicates. All five folds passed all three checks. Peer-count diagnostics recomputed
independently from `questionnaire_fingerprints.rds` and `fold_of_case` alone (not from
the cached deltas) reproduce the previously reported range (7-15 zero-peer, 36-41
one-peer held-out respondents per fold), confirming this worktree's copies are the same
data the rejected experiment described.

These cached artifacts supply the frozen-model probabilities (`p4`, hence `g_v`, `h_v`)
and the doubly-nested nothing-touches-the-holdout structure. **Nothing about the
correction estimator itself (similarity metric, neighbor graph, kernel smoother, k/lambda
grid, or the pooled Newton formula) is reused** -- that part is new and is fully
specified below.

## 3. Similarity metric (design-only, fixed before any fold split)

For each of the 299 versions, all respondents assigned to it share, by the fingerprint's
own construction, an **identical** ordered 19-task design (attributes + price, alt 4
excluded since it is always the constant all-zero opt-out). So one representative
respondent per version fully determines that version's design.

**Feature vector.** For version `v`, take one representative respondent's 19 tasks x 3
real alternatives (57 design cells) and compute, for each of the 20 numeric design
columns (`CC, GN, NS, BU, FA, LD, BZ, FC, FP, RP, PP, KA, SC, TS, NV, MA, LB, AF, HU,
Price`), the mean level across those 57 cells. This gives one 20-dimensional vector
`x_v` per version, built **only** from `train.csv`/`test.csv` design columns (never
`Ch1..Ch4`), so it is identical whether or not the version's respondents are in train or
test, and identical across every outer/inner split -- computed exactly once, globally,
with zero leakage risk.

**Standardization and distance.** Each of the 20 coordinates is z-scored using the mean
and SD across all 299 versions (fixed constants, not fold-dependent). Pairwise distance
is Euclidean in this standardized 20-d space.

**Why the design basis over a covariate-composition basis.** The brief offered two
options (design-marginal frequencies vs. respondent-covariate composition of a version's
assignees). Covariate composition would need to be recomputed per outer fold (a
version's inner-training respondent composition changes fold to fold), adding another
leakage-prone moving part for comparatively little added conceptual value. The design
basis is simpler, deterministic, identical across every fold, and requires no
recomputation inside the nested loop, so it is the pre-registered primary (and only)
similarity basis for this run.

## 4. Neighbor graph and kernel-smoothed Newton estimator

For version `v`, let `N_k(v)` be its `k` nearest **other** versions by the distance
above (`k` is a pre-registered grid value, see below). Define an adaptive-bandwidth
Gaussian kernel weight:

```
b_v            = distance from v to its k-th nearest neighbor      (k > 0 only)
w(v, u)        = exp(-0.5 * (d(v,u) / b_v)^2)   for u in N_k(v)
w(v, v)        = 1                                (self weight, always)
w(v, u)        = 0                                otherwise
```

`k = 0` means no neighbors at all (only the self weight survives), which makes every
formula below **collapse exactly to the rejected per-version Newton estimator** -- this
is a deliberate built-in identity check, not a separate method.

**Pooled sufficient statistics.** Where the rejected method used a version's own
`g_v = sum(p4-y4)`, `h_v = sum(p4(1-p4))` computed strictly from inner-OOF rows, this
method instead pools across the weighted neighborhood:

```
G_v = sum_{u in N_k(v) U {v}} w(v,u) * g_u
H_v = sum_{u in N_k(v) U {v}} w(v,u) * h_u
delta_v = -G_v / (H_v + lambda)          (delta_v = 0 if lambda = Inf, or if H_v == 0)
```

`g_u`/`h_u` for a neighbor version `u` are computed from whatever inner-OOF rows that
version happens to contribute in the current fold (zero if `u` has no respondents among
the current outer-training set, e.g. a test-only or held-out-only version) -- neighbor
weight is not conditioned on a version being "usable"; an empty neighbor simply
contributes zero mass to both numerator and denominator. The neighbor graph itself is
built once from all 299 versions and is not refit per fold.

**Leave-one-respondent-out at the estimation step (mirrors the rejected design's own
extra safeguard).** When estimating `delta` for a specific held-out-in-the-inner-loop
respondent `i` belonging to version `v`, that respondent's own `g`/`h` contribution to
`v`'s totals is first subtracted (`excluded_g_v = total_g_v - g_i`, likewise for `h`)
before pooling with neighbors -- so respondent `i` never receives a correction that used
their own residual, even indirectly through their own version's aggregate, exactly as
`leave_respondent_out_newton()` did in the rejected experiment. Neighbor versions'
totals are used as-is (respondent `i` does not belong to them, so no exclusion is
needed there). For the **final** application to a fully disjoint outer-holdout fold, no
such exclusion is needed (no outer-holdout respondent ever contributed to any inner-OOF
statistic), so the full pooled `G_v`/`H_v` from all outer-training respondents is used.

**Application.** Exactly as the rejected method: add `delta_v` to `log(p4)` for every row
whose respondent belongs to version `v`, and re-normalize (softmax-consistent
re-scaling of the 4-way probability vector), i.e. the same
`apply_optout_delta_vector()`-style transform.

## 5. Grid and nested-CV selection rule (exactly mirrors the rejected design's rigor)

- Outer loop: the canonical 5 respondent-grouped folds (`fold_of_case`, seed 4821).
- Inner loop, per outer fold: the frozen architecture's 4 inner-fold refits (already
  cached, see Section 2) supply genuine inner-OOF predictions for all 908
  outer-training respondents.
- Grid, evaluated jointly per outer fold: `k in {0, 5, 10, 20, 40}` x
  `lambda in {5, 10, 20, 40, 80, 160, 320, Inf}` = 40 combinations. `(k=0, lambda=Inf)`
  is the explicit no-correction baseline entry.
- Selection rule: for each `(k, lambda)`, compute the leave-one-respondent-out pooled
  correction (Section 4) for all 908 outer-training respondents' inner-OOF rows, apply
  it, and compute the pooled log loss over all 17,252 inner-OOF rows. Pick the `(k,
  lambda)` that **minimizes** this pooled inner loss for that outer fold (ties broken
  toward the no-correction entry). This mirrors the rejected experiment's own selection
  rule one-for-one, just over a 2-D grid instead of a 1-D one.
- Final per-fold estimate: recompute `G_v`/`H_v` (no leave-one-out, full outer-training
  pool) at the selected `(k, lambda)`, assign `delta_v` to every outer-holdout respondent
  by their version, and apply to that fold's outer-holdout predictions.

**Two outer baselines, one primary.** Primary: the exact fixed submitted OOF matrix
(`1.143686618134879`, "ensemble_v11+MLP", the number this whole project calls current
best). Secondary/robustness: the freshly-refit `outer{f}_final_base_fit.rds$target_pred`
pooled across folds (this differs slightly from the primary baseline because it uses
different xgboost/MLP random seeds -- reported for robustness only, exactly as the
rejected experiment's write-up did; the promotion decision below is made on the primary
comparison only).

## 6. Reporting and promotion rule (hard constraint, restated)

- Comparison: respondent-clustered paired bootstrap. Per-respondent mean gain in log
  loss (baseline row loss minus corrected row loss, averaged over that respondent's 19
  rows), resampled with replacement at the respondent level, 100,000 replicates, seed
  4821 (same seed/replicate count as the rejected experiment, for direct comparability).
- **Promotion bar:** the ordinary 95% respondent-bootstrap CI of the gain vs. the primary
  baseline must exclude zero (lower bound `> 0`). A positive point estimate alone is
  explicitly **not** sufficient, per the parent brief.
- If (and only if) the primary comparison clears the bar, the same bootstrap will also
  be run against the secondary (freshly-refit) baseline as a robustness check, and slice
  diagnostics (peer-count bands, opt-out vs. inside-choice rows, selected `k`/`lambda`
  per fold, sign consistency) will be reported in the same style as the rejected
  experiment's write-up.
- If the bar is not cleared, that is reported as a clean null result: no submission
  file will be generated, and the finding will be written up exactly as plainly as a
  positive result would have been.

## 7. Smoke test plan (before the real run)

Before trusting the full 5-outer x 40-grid evaluation:

1. Verify the neighbor graph mechanically: for a handful of versions, confirm `k`
   neighbors are returned, weights are in `(0, 1]`, decreasing in distance, and that
   `k = 0` produces an all-self (identity) weighting.
2. Run the full pooled-Newton pipeline with `k` fixed at `0` for all folds and confirm
   it reproduces the rejected experiment's own published numbers (per-fold selected
   `lambda`, per-fold outer gains, and the pooled `-0.000172` / CI
   `[-0.000800, +0.000458]` result) to within floating-point tolerance. This is the
   single most important check: if the `k=0` special case does not reproduce a result
   this codebase did not have to re-derive, the new machinery has a bug, and no `k>0`
   result will be trusted until it is fixed.
3. Only after (1) and (2) pass does the full `k in {0,5,10,20,40}` grid run.

## 8. Outputs

All generated files go to `data_processed/codex_version_pool/` (gitignored) and this
repo's `codex_version_pool_findings.md` (git-tracked). Scripts:
`R/codex_version_pool_common.R` (shared functions), `R/codex_version_pool_smoketest.R`
(Section 7), `R/codex_version_pool_cv.R` (the real run). No `AGENTS.md`,
`cleaning_log.md`, or `submissions_log.csv` edits are made by this branch.
