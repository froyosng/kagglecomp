# Shared-utility / residual-model round

Branch: `codex-shared-utility`, based exactly on `zhenhao@64817f7`.

Decision: **no new model is adopted and no Kaggle submission was made.** The
only promising new fit, the residual exact-softmax correction, produced a
small positive CV point estimate but failed the respondent-bootstrap
uncertainty bar. The global-scale and `yearind` models were harmful. The
test-like re-ranking provides additional support for the already-existing
8-component arithmetic candidate, but does not repair its previously failed
99%/multiplicity-adjusted interval.

All comparisons hard-assert the known cached OOF scores before calculating a
new result:

- m8trpg: 1.147021211
- ensemble_v11: 1.145094213
- submitted shallow-MLP blend: 1.143789442
- triple+MLP: 1.143327611
- 8-component arithmetic blend: 1.142111910

No existing script, `AGENTS.md`, `cleaning_log.md`, or
`submissions_log.csv` was edited.

## 1. Exact shared-utility choice likelihood

### `clogitboost` gate

`clogitboost` 1.1 installed successfully from the current Windows binary. Its
built-in `travel` example fit and held-out `predict()` call both worked.
However, a one-iteration fit on the project's 69,008-row long-format training
screen ran for about 195 seconds and then failed at the first binary feature:

```text
smooth.spline: need at least four unique 'x' values
```

This is structural for the package's componentwise spline learner, not an
installation problem. The project's alternative flags and categorical
attribute dummies are essential and mostly have only two or three unique
values. Dropping all of them merely to make the package run would not be a
faithful shared-utility candidate, so I used the requested custom-objective
fallback.

### Custom XGBoost objective

The fallback uses one scoring function over long-format alternatives and the
exact four-way task likelihood:

```text
L_i = -sum_j y_ij log softmax_j(f(x_i1), ..., f(x_i4))
gradient_ij = p_ij - y_ij
```

Rows are sorted by task and alternative, `qid` is attached to the DMatrix, and
every objective call asserts exactly one chosen alternative in every
consecutive four-row task. The true softmax Hessian has cross-alternative
terms, which XGBoost cannot consume in its scalar-row custom objective. The
implementation therefore uses the standard diagonally dominant multinomial
upper bound `2 p(1-p)`, the same strategy documented for XGBoost's own
multinomial objective. `nthread=1` is forced for deterministic artifacts.

The analytic gradient was checked against central finite differences before
any fit:

| Check | Result |
|---|---:|
| Maximum absolute gradient error | 1.0554e-10 |
| Mean absolute gradient error | 3.8608e-11 |
| Tolerance | 1e-7 |

References:

- [XGBoost R custom-objective interface](https://xgboost.readthedocs.io/en/latest/r_docs/R-package/docs/reference/xgb.train.html)
- [XGBoost advanced custom-objective notes](https://xgboost.readthedocs.io/en/release_3.0.0/tutorials/advanced_custom_obj.html)
- [Shi and Yin, componentwise smoothing-spline conditional-logit boosting](https://doi.org/10.1016/j.jocm.2017.07.002)

### Cold-start shared-utility result

The exact shared model was screened at depths 2 and 3 with checkpoint learning
curves. Its best component score was **1.218569** (`depth=3`, 200 rounds),
versus 1.160568 for v11 on the same canonical screen. Every shared-model
checkpoint received exactly zero diagnostic blend weight. Per the established
screen-first rule, it did not advance to five-fold CV.

This closes the specific structural gap in the brief: optimizing the exact
four-way cross-entropy with a shared alternative-level function did not make a
cold-start tree model competitive.

## 2. Residual nonlinear utility correction

The same objective was then initialized with the fold-specific m8trpg utility
as `base_margin`. The tree therefore learned a correction while directly
optimizing the residual four-way choice loss, rather than being blended after
training.

The feature set contains own-alternative attribute codes and price,
price-gap/rank context, task position, and respondent covariates. It does not
recreate m8trpg's full hand-written interaction block. Three deliberately
small, strongly regularized structures were screened. The frozen winner was:

```text
max_depth=1, eta=0.03, min_child_weight=50,
lambda=30, nrounds=200, nthread=1
```

On the canonical single split, replacing m8trpg inside v11 improved 1.160568
to **1.159970**, a gain of 0.000598. That cleared the screen gate.

Canonical respondent-grouped five-fold CV:

| Fold | Base m8trpg | Residual m8trpg | Gain |
|---:|---:|---:|---:|
| 1 | 1.208224 | 1.208072 | +0.000151 |
| 2 | 1.124692 | 1.124356 | +0.000336 |
| 3 | 1.145733 | 1.145244 | +0.000488 |
| 4 | 1.118863 | 1.118232 | +0.000631 |
| 5 | 1.137594 | 1.138155 | -0.000561 |
| **Pooled** | **1.147021** | **1.146812** | **+0.000209** |

Propagation through the established ensemble:

| Comparison | Baseline | Candidate | Gain |
|---|---:|---:|---:|
| m8trpg | 1.147021 | 1.146812 | +0.000209 |
| v11 with corrected m8trpg | 1.145094 | 1.144952 | +0.000143 |
| submitted MLP blend with corrected v11 | 1.143789 | **1.143714** | **+0.000076** |

The high-precision 100,000-resample respondent bootstrap for the last
comparison was:

```text
point gain       +0.0000759
95% CI           [-0.0000839, +0.0002417]
99% CI           [-0.0001301, +0.0002978]
bootstrap win rate 81.9%
```

The interval crosses zero. A diagnostic cross-fitted arithmetic blend between
the old and corrected candidates did not rescue it (gain +0.0000258, 95% CI
[-0.0001122, +0.0001685]). **Not adopted.**

The correction's tree importance is dominated in every fold by
`price_gap_min` and `price_gap_max`, followed far behind by price and continuous
respondent covariates. That is consistent with a tiny refinement of the known
price-context mechanism, not a new large source of signal.

## 3. Continuous global utility-scale heterogeneity

Using honest OOF m8trpg probabilities as utilities (`u = log(p)`), I fit:

```text
log(mu_it) =
  gamma_0 + gamma_1 Task_c + gamma_2 z(income)
  + gamma_3...gamma_7 segment dummies

p_ijt = softmax_j(mu_it * u_ijt)
```

Each fold's `gamma` was estimated only on the other four folds. The analytic
gradient was independently finite-difference checked (maximum error
1.36e-10). Unpenalized, ridge 0.001, and ridge 0.01 versions were tested.

All were harmful:

| Ridge | Scaled m8trpg | Adjusted submitted blend | Gain vs current |
|---:|---:|---:|---:|
| 0 | 1.149249 | 1.145572 | -0.001783 |
| 0.001 | 1.149100 | 1.145500 | -0.001710 |
| 0.01 | 1.148533 | 1.145240 | -0.001451 |

Even the least harmful version had a 95% gain interval of
**[-0.002214, -0.000677]**, entirely below zero. This is a decisive negative,
not an uncertainty-limited null.

## 4. `yearind` price/inside interactions

Added `Price x yearind` and `inside x yearind` dummy interactions to m8trpg
(12 coefficients, year 1 reference). Respondent counts by level were
211/165/172/155/209/176/47. The smallest cell has 47 respondents, so this is
not analogous to the earlier six-respondent binned-covariate
quasi-separation.

The single-split screen was already harmful: 1.159681 to 1.162152
(-0.002470). Because this was the only never-tested seven-level covariate and
the fit was cheap, I still completed canonical five-fold CV:

- m8trpg: 1.147021 to **1.148737** (gain -0.001716; 4/5 folds worse)
- propagated submitted blend: 1.143789 to **1.144797** (gain -0.001007)
- respondent-bootstrap 95% gain interval:
  **[-0.002592, +0.000555]**

**Not adopted.**

## 5. Test-like-respondent re-ranking

I reproduced the existing adversarial-validation logistic classifier at
5-fold AUC **0.634130**. A full domain classifier then scored every training
respondent's probability of being test-like. No choice labels enter that
classifier. Existing canonical OOF predictions were re-evaluated on the top
20%, 25%, and 30% of training respondents by that score:

| Candidate | All train | Top 30% | Top 25% | Top 20% |
|---|---:|---:|---:|---:|
| ensemble_v11 | 1.145094 (4) | 1.164214 (3) | 1.174002 (3) | 1.179695 (4) |
| submitted MLP blend | 1.143789 (3) | 1.164348 (4) | 1.174010 (4) | 1.178549 (3) |
| triple+MLP | 1.143328 (2) | 1.163807 (2) | 1.173320 (2) | 1.178483 (2) |
| 8-component arithmetic | **1.142112 (1)** | **1.160396 (1)** | **1.169896 (1)** | **1.175999 (1)** |

Parentheses show rank within each slice. The two leading candidates do not
swap: the 8-component blend remains first and triple+MLP remains second at
every cutoff.

Relative to the submitted MLP blend, the 8-component candidate's gain is
0.001678 overall and 0.003951 on the top-30% slice. The top-30% ordinary
clustered 95% interval is **[+0.000262, +0.007557]**. The corresponding
top-25% and top-20% intervals cross zero as the respondent count shrinks.

Interpretation: this is useful corroborating evidence that the 8-component
candidate's direction is not driven only by the least test-like respondents.
It is not a new independent model result, and it does not make the existing
full-population 99%/Bonferroni intervals exclude zero. I therefore did not
change the prior no-submission decision.

## Reproduction

Install the optional package checked first:

```r
install.packages("clogitboost")
```

Run from the repository root:

```powershell
& 'C:\Program Files\R\R-4.6.0\bin\Rscript.exe' R/codex_shared_utility_screen.R
& 'C:\Program Files\R\R-4.6.0\bin\Rscript.exe' R/codex_residual_utility_cv.R
& 'C:\Program Files\R\R-4.6.0\bin\Rscript.exe' R/codex_global_scale.R
& 'C:\Program Files\R\R-4.6.0\bin\Rscript.exe' R/codex_year_interactions.R
& 'C:\Program Files\R\R-4.6.0\bin\Rscript.exe' R/codex_test_like_reranking.R
```

Generated diagnostics and OOF artifacts are under
`data_processed/codex_shared_utility/` (gitignored). Source files added:

- `R/codex_shared_utility_common.R`
- `R/codex_shared_utility_screen.R`
- `R/codex_residual_utility_cv.R`
- `R/codex_global_scale.R`
- `R/codex_year_interactions.R`
- `R/codex_test_like_reranking.R`
