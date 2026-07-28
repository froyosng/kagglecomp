# Codex overnight modeling queue: findings

Branch: `codex-overnight-queue`, based on `zhenhao` commit `5a8d601`.

No Kaggle submission was made or generated. None of the candidates met the
predeclared evidence bar.

## Bottom line

| Experiment | Baseline CV | Candidate CV | Point gain | Respondent-bootstrap 95% CI | Decision |
|---|---:|---:|---:|---:|---|
| Version Newton opt-out shift | 1.143687 | 1.143858 | -0.000172 | [-0.000800, +0.000458] | Reject |
| Prior-smoothed design history | 1.143687 | 1.142951 | +0.000735 | [-0.000090, +0.001564] | Not confirmed |
| Preregistered 24-config deep MLP | 1.143687 | 1.143322 | +0.000365 | [-0.000424, +0.001168] | Not confirmed |

The history candidate is the most interesting clue: it improved four of five
folds and nearly cleared the ordinary interval. It still failed the
predeclared statistical bar, and its five-candidate family-wise interval was
`[-0.000351, +0.001837]`. It should be logged as an unconfirmed lead, not
treated as a submission candidate.

## 1. Version-level Newton opt-out correction

The implementation follows the refined utility-space specification exactly.
For version `v`, using inner-OOF base probabilities only,

```
g_v = sum(p4 - y4)
h_v = sum(p4 * (1 - p4))
delta_v = -g_v / (h_v + lambda)
```

The outer held-out probabilities are corrected by adding `delta_v` to the
opt-out utility and applying the corresponding softmax update. A single global
`lambda` was selected inside each outer fold from
`{5, 10, 20, 40, 80, 160, 320, Inf}`. Respondents, not rows, define every
split. Versions with no usable training peer receive zero correction.

The two relevant outer-CV evaluations agree:

- Exact fixed-15% submitted v11+MLP OOF: `1.143686618 -> 1.143858171`,
  gain `-0.000171553`.
- Freshly refitted outer-fold pipeline: `1.143597718 -> 1.143766787`,
  gain `-0.000169070`.

Only folds 1, 4, and 5 selected a finite penalty (`lambda=160`); folds 2 and 3
selected no correction. On the official OOF baseline, fold gains were
`+0.0000865, 0, 0, -0.0000591, -0.0008852`. Excluding versions with only one
usable peer did not rescue the result: point gain `-0.0000679`, 95% CI
`[-0.000679, +0.000538]`.

The small effective group size is visible in the diagnostics. A held-out
respondent had about 2.77-2.99 training peers on average; 7-15 respondents per
fold had zero peers and 36-41 had one peer. Some version gradients were
dominated by a single respondent even when at least two peers existed.

Verdict: reject. The inner selection often shrinks completely to zero, and the
out-of-fold correction is harmful overall. Per the brief, no version-specific
scale, cheapest-inside shift, or full alternative intercepts were attempted.
The earlier raw mean-residual formulation was also evaluated only as a
documented negative control and was harmful (`1.143687 -> 1.143930`).

Detailed implementation and diagnostics are in:

- `R/codex_version_shrinkage_common.R`
- `R/codex_version_shrinkage_cv.R`
- `R/codex_version_shrinkage_diagnostics.R`
- `R/codex_version_shrinkage_audit.R`
- `codex_version_shrinkage_findings.md`

## 2. Historical reference-price and attribute-exposure features

The existing `R/codex_design_history.R` had already tested unsmoothed,
respondent-only running price and exposure features, with Task 1 initialized
to zero. That round was negative or multiplicity-unconfirmed. This follow-up
tested the untried refinement in the latest brief: a fold-fitted population
prior for initialization and partial pooling.

For each respondent and task, using design information from tasks `1..t-1`
only:

- price feature = the alternative's current price minus a running reference
  price;
- attribute feature = the mean prior familiarity of the alternative's active
  attribute levels.

Both histories are initialized from the fitting fold's design distribution.
Prior strength `k` acts as pseudo-exposure: the population prior dominates
Task 1 and the respondent's own observable sequence gradually dominates later.
No previous or current choices enter either feature.

Five candidates were fixed before screening:

1. price only, `k=9`;
2. attribute familiarity only, `k=9`;
3. both, `k=3`;
4. both, `k=9`;
5. both, `k=27`.

Only `both_k3` improved the canonical single split, by `+0.0000222`, so only
that frozen candidate went to five-fold CV. Applied to the exact submitted
v11+MLP architecture, it produced:

- baseline: `1.143686618`;
- candidate: `1.142951450`;
- gain: `+0.000735168`;
- fold gains: `+0.000621, +0.001236, +0.001024, +0.001154, -0.000360`;
- ordinary respondent-bootstrap 95% CI:
  `[-0.000090, +0.001564]`;
- family-5 Bonferroni interval:
  `[-0.000351, +0.001837]`;
- conservative interval across this family plus the eight earlier history
  candidates:
  `[-0.000483, +0.001960]`.

The price-history coefficient was negative in all five folds
(`-0.120` to `-0.183`), which is behaviorally coherent and more stable than
the attribute-familiarity coefficient, whose sign changed across folds.

Verdict: promising but not confirmed. It passes the four-of-five fold check
but no uncertainty interval excludes zero. To preserve the fixed search, I did
not post-hoc test a price-only `k=3` variant after seeing the coefficient
stability.

Implementation: `R/codex_history_prior_smoothing.R`.

## 3. Preregistered wider deep-MLP search

The complete 24-configuration registry and decision rule were committed before
any new architecture was run (`3f479a7`; see
`codex_deep_mlp_wide_preregister.md`). It crossed:

- six layouts: `32-16`, `64-32`, `128-64`, `256-128`, `128-64-32`,
  `256-128-64`;
- four frozen optimizer/regularization recipes varying dropout, weight decay,
  learning rate, and epochs.

The old `128-64`, dropout `0.30`, weight decay `0.001` architecture was included
as an anchor. All models used the established `build_mlp_matrix()` features.
The screen averaged seeds 13101 and 13102 and selected on three-way arithmetic
blend loss across v11, shallow MLP, and deep MLP.

The frozen winner was:

- hidden layers: `256-128-64`;
- dropout: `0.45`;
- weight decay: `0.002`;
- learning rate: `0.001`;
- 12 epochs, batch size 256.

Its screen blend gain was `+0.002246`. In canonical five-fold CV, averaging
three seeds per fold:

- deep component alone: `1.190743`;
- exact v11+shallow baseline: `1.143686618`;
- fold-cross-fitted three-way blend: `1.143321960`;
- gain: `+0.000364659`;
- fold gains: `+0.000195, +0.000610, +0.001067, -0.000136, +0.000087`;
- ordinary respondent-bootstrap 95% CI:
  `[-0.000424, +0.001168]`;
- 99% CI: `[-0.000670, +0.001421]`;
- family-24 Bonferroni interval:
  `[-0.000857, +0.001618]`.

The deep network received a stable 9.2%-12.6% cross-fitted weight, so it is
genuinely different enough to enter the blend. The gain is nevertheless
smaller than its sampling uncertainty and does not survive the predeclared
architecture-search correction.

Verdict: not adopted. Seed-bagging was explicitly gated on an ordinary 95%
lower bound above zero plus four improving folds. The fold condition passed,
but the interval condition failed, so no bagging run was performed.

Implementation:

- `codex_deep_mlp_wide_preregister.md`
- `R/codex_deep_mlp_wide_search.R`
- small reusable-function extension in `R/codex_torch_deep_mlp.R`

## Reproduction and audit

The generated model artifacts are under
`data_processed/codex_overnight_queue/` and
`data_processed/codex_version_shrinkage/` (gitignored).

The consolidated audit independently reloads the saved prediction matrices,
recomputes all three losses and fold results, verifies the exact baseline and
row alignment, checks the 24-entry architecture registry and frozen selection,
and confirms that both new candidate intervals cross zero:

```powershell
& 'C:\Program Files\R\R-4.6.0\bin\Rscript.exe' `
  'R/codex_overnight_audit.R'
```

It passes, as do the two dedicated version-correction audits.

## Recommendation

Do not submit any result from this queue. The history prior is worth preserving
as a technically coherent clue, but its maximum supported claim is “small,
four-fold-consistent point improvement whose confidence interval still crosses
zero.” The deeper search confirms that neural diversity can receive nonzero
blend weight, but a 24-configuration search did not produce statistically
defensible evidence of improvement over the already-submitted v11+MLP model.
