# CatBoost ensemble follow-up

## Verdict

**Not adopted; no submission candidate generated.** CatBoost was installable
and ran correctly, but its tiny single-split signal did not generalize. In the
canonical respondent-grouped five-fold CV, CatBoost received exactly zero
fold-cross-fitted blend weight in all five folds. The current submitted
shallow-MLP ensemble therefore remains unchanged at CV log loss 1.143789442.

## Installation and smoke test

The R package was not initially installed. This machine has R 4.6.0 but no
Rtools, so building the full CatBoost C++ source through
`devtools::install_github()` was unnecessarily fragile. I instead followed
CatBoost's current official Windows guidance and installed the official
GitHub release binary:

```r
remotes::install_url(
  paste0(
    "https://github.com/catboost/catboost/releases/download/",
    "v1.2.10/catboost-R-windows-x86_64-1.2.10.tgz"
  ),
  INSTALL_opts = c("--no-multiarch", "--no-test-load")
)
```

CatBoost 1.2.10 then passed a real multiclass smoke test: an R
`catboost.Pool` was constructed, an eight-tree four-feature model trained,
and its probability output was finite and normalized to machine precision.
The package is installed in the user's R library and is not part of the Git
commit.

Sources: [CatBoost released R-package installation
instructions](https://catboost.ai/docs/en/installation/r-installation-binary-installation);
[Prokhorenkova et al. (2018), CatBoost: unbiased boosting with categorical
features](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html).

## Method

`R/codex_catboost_ensemble.R` mirrors the screen-first discipline in
`R/codex_lightgbm_ensemble.R`.

- Data were split only by respondent using the canonical single split
  (seed 7402) and canonical five-fold map (seed 4821).
- Inputs came directly from `wide_feature_matrix()` in
  `R/codex_modeling_common.R`: 95 established wide-format columns, including
  all alternative attribute/price columns and all established respondent
  covariates.
- The 80 attribute/price columns and 11 categorical respondent columns were
  declared as native CatBoost categorical features. The four continuous
  `*a` covariates remained numeric.
- `boosting_type = "Ordered"` and `one_hot_max_size = 2` ensured that
  non-binary categoricals used CatBoost's ordered target-statistic mechanism.
- Four modest configurations varied depth, learning rate, regularization,
  and categorical-combination complexity. Early stopping was used only at
  the screen stage.
- The screen hard-asserted the known current-model loss
  1.160411892984 before any comparison.
- Because one candidate technically selected nonzero screen weight, it
  proceeded to CV. Its early-stopped 681-tree count was frozen before CV;
  outer held-out labels therefore did not control stopping.
- CV blend weights were selected for each held-out fold using only the other
  four folds.

## Canonical single-split screen

| Configuration | Depth | CTR complexity | Trees | Component | Best weight | Blend | Gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordered_d5 | 5 | 1 | 504 | 1.214087344 | 0.00 | 1.160411893 | 0 |
| ordered_d6 | 6 | 1 | 610 | 1.212039738 | 0.00 | 1.160411893 | 0 |
| ordered_d7 | 7 | 1 | 534 | 1.222048179 | 0.00 | 1.160411893 | 0 |
| ordered_ctr2 | 6 | 2 | 681 | 1.212284220 | 0.02 | 1.160375517 | 0.000036376 |

The first three configurations reproduced LightGBM's qualitative result:
each received exactly zero blend weight. `ordered_ctr2` earned 2%, but its
gain was only 3.6e-05. I nevertheless advanced it because the stopping rule
was predeclared as nonzero screen weight, not a subjective minimum gain.

CatBoost was much slower than LightGBM. The selected screen fit took about
622 seconds. One discarded configuration recorded an isolated 17,522-second
wall-clock time while neighboring fits took 149-622 seconds; this runtime
irregularity did not affect the saved predictions or model-selection rule.

## Five-fold confirmation

The selected `ordered_ctr2` specification was refit with exactly 681 trees
in every outer fold.

| Fold | CatBoost log loss | CatBoost blend weight |
|---:|---:|---:|
| 1 | 1.230709443 | 0.00 |
| 2 | 1.206088824 | 0.00 |
| 3 | 1.193991846 | 0.00 |
| 4 | 1.189340704 | 0.00 |
| 5 | 1.198168179 | 0.00 |

Pooled results:

- CatBoost component: **1.203659799**
- current submitted model: **1.143789442**
- fold-cross-fitted CatBoost blend: **1.143789442**
- gain: **exactly 0**
- selected weights: **0, 0, 0, 0, 0**

The requested respondent bootstrap was run with 100,000 resamples and a
four-configuration family size. Because every selected weight was zero, the
candidate prediction matrix is literally identical to the baseline; the
gain distribution is consequently a point mass at zero. This is not an
uncertainty claim—it is the deterministic consequence of the honest
cross-fitted selector rejecting CatBoost in every fold.

## Interpretation

CatBoost's ordered categorical statistics did not recover useful ensemble
diversity from these features. Its standalone loss was worse than the
original wide xgboost, ranking-objective xgboost, and glmnet-Cox components
already logged by the project. More importantly, its errors added no value
to the current submitted ensemble once evaluated across new respondents.

This closes the CatBoost lead cleanly. The result supports the earlier
LightGBM finding rather than overturning it: changing the tree learner's
categorical machinery is not enough to escape the project's current
1.142-1.144 CV range.

## Reproducibility artifacts

- Script: `R/codex_catboost_ensemble.R`
- Screen results: `data_processed/codex_catboost/catboost_screen.csv`
- Saved screen models/predictions:
  `data_processed/codex_catboost/catboost_screen.rds`
- CV fold results: `data_processed/codex_catboost/catboost_cv_fits.csv`
- CV weights: `data_processed/codex_catboost/catboost_cv_weights.csv`
- Bootstrap summary:
  `data_processed/codex_catboost/catboost_cv_bootstrap.csv`
- Saved OOF predictions and bootstrap object:
  `data_processed/codex_catboost/catboost_oof.rds`

All generated `data_processed/` artifacts are gitignored but remain in the
working copy for independent review. `AGENTS.md`, `cleaning_log.md`,
`submissions_log.csv`, and all pre-existing R scripts were left untouched.
