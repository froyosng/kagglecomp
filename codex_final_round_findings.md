# Final modeling round: bagging, partial pooling, and SHAP interactions

Branch: `codex-bagging-pooling-shap`, based on `zhenhao@ba93e2c`.

Verdict: **no candidate is adopted and no Kaggle submission is recommended.**
The official `ensemble_v11` remains best at CV 1.145094 and public 1.202.

All comparisons reuse the saved canonical respondent-grouped folds
(`seed = 4821`) and hard-check the known m8trpg and ensemble_v11 OOF losses
before comparing a candidate. Existing scripts and the three project logs were
not modified.

## 1. Respondent-bootstrap bagging of m8trpg

Script: `R/codex_mlogit_bagging.R`

Each bootstrap fit samples training respondents with replacement and keeps all
19 tasks from a sampled respondent together. Every copied respondent receives a
synthetic `Case`, `No`, and `chid`; the script asserts before each fit that:

- every `chid` maps to exactly one synthetic respondent;
- every task has exactly four alternatives;
- every `(chid, alt)` key is unique; and
- every synthetic respondent has 19 complete tasks.

The single-split screen was slightly negative: the 15-fit average scored
1.160469 versus the m8trpg baseline's 1.159681 (gain -0.000788). Because this
was the highest-priority test and the screen difference was within the
project's paired-noise scale, I still completed the requested five-fold check.
All 75 bootstrap fits succeeded; no replacement fit was needed.

| OOF model | Log loss | Gain vs corresponding baseline |
|---|---:|---:|
| Single m8trpg | 1.147021 | -- |
| 15-bag m8trpg | 1.147764 | -0.000743 |
| Official ensemble_v11 | 1.145094 | -- |
| 80% bagged m8trpg + 20% original xgboost | 1.145658 | -0.000563 |
| 80% bagged m8trpg + 20% seed-bagged xgboost | 1.145660 | -0.000566 |

Every point on the 1-to-15-bag pooled learning curve is worse than the
corresponding single-fit baseline; 15 bags is the least harmful blend, so this
is not an arbitrary unlucky stopping count. Respondent-clustered bootstrap
results:

| Comparison | Point gain | 95% CI | Win rate |
|---|---:|---:|---:|
| Bagged vs single m8trpg | -0.000743 | [-0.002098, 0.000565] | 15.9% |
| Bagged-m8trpg blend vs ensemble_v11 | -0.000563 | [-0.001611, 0.000451] | 15.3% |
| Both components bagged vs ensemble_v11 | -0.000566 | [-0.001648, 0.000517] | 16.5% |

Interpretation: unlike xgboost seed-bagging, respondent-bootstrap refits inject
enough finite-sample coefficient variation/bias into m8trpg to outweigh any
variance reduction. The result is null statistically and negative in point
estimate, so it does not clear the submission bar.

## 2. Penalized partial pooling of segment slopes

Script: `R/codex_partial_pooling.R`

This test uses the stratified-Cox equivalence but keeps the **full m8trpg design
unpenalized**, rather than comparing against the earlier weaker Cox model. The
new penalized candidates are:

- five `Price x segment x z(mileage)` deviations;
- five `Price x segment x z(income)` deviations; and
- both sets jointly.

The existing global `Price x z(mileage)` / `Price x z(income)` and
`Price x segment` terms remain in the core. Thus shrinking a new deviation to
zero is genuine pooling toward the current common slope. Candidate columns are
fold-locally standardized, and alpha in `{0, 0.25, 0.5, 0.75, 1}` plus lambda
are selected by respondent-grouped inner CV.

All three candidate sets select alpha 0, but the selected penalty drives every
new coefficient effectively to zero (absolute magnitudes around `1e-40`).
Their held-out loss is identically 1.159721 versus 1.159681 for m8trpg; the
0.000039 difference is the numerical difference between the Cox/glmnet and
mlogit optimizers after the candidate terms vanish.

No partial-pooling candidate passed the screen, so none proceeded to outer CV.
The data-driven answer is full pooling: the nested procedure rejects the
segment deviations rather than finding an intermediate shrinkage level.

## 3. XGBoost SHAP interaction discovery

Script: `R/codex_shap_interactions.R`

To avoid using held-out labels to nominate their own terms, the reference
xgboost is fit only on the canonical single-split training respondents with the
original 73-round configuration and `nthread = 1`. Exact multiclass SHAP
interaction values are aggregated in bounded-memory batches over 100 complete
training respondents (1,900 tasks).

The script writes the ranking of all raw feature pairs, then restricts
translation to defensibly standardized respondent covariates. Categorical codes
such as region and segment are not treated as continuous just because xgboost
can split on them. Duplicate coarse/fine representations are collapsed by
keeping the strongest raw representation per conceptual pair. The previously
tested income-age, income-mileage, and income-night pairs are excluded after
cross-checking `submissions_log.csv`.

The eight selected untried pairs, in SHAP order, are:

1. age x gender
2. age x mileage
3. mileage x night driving
4. age x night driving
5. age x education
6. mileage x urbanicity
7. education x mileage
8. gender x income

Each pair adds both `Price x z(cov1) x z(cov2)` and
`inside x z(cov1) x z(cov2)` to m8trpg. Four pass the single-split screen:

| Pair | Screen loss | Screen gain |
|---|---:|---:|
| age x mileage | 1.157292 | +0.002389 |
| education x mileage | 1.158076 | +0.001606 |
| mileage x urbanicity | 1.158706 | +0.000975 |
| age x gender | 1.158742 | +0.000939 |
| age x night driving | 1.160090 | -0.000409 |
| age x education | 1.160591 | -0.000910 |
| gender x income | 1.161185 | -0.001504 |
| mileage x night driving | 1.162644 | -0.002962 |

The four screen winners were refit from scratch in all five canonical folds:

| Pair | Mlogit CV | Mlogit gain | Fixed 80/20 blend | Blend gain | Blend 95% CI |
|---|---:|---:|---:|---:|---:|
| age x gender | 1.147304 | -0.000283 | 1.145143 | -0.000048 | [-0.001815, 0.001614] |
| age x mileage | 1.147616 | -0.000595 | 1.145252 | -0.000158 | [-0.001512, 0.001135] |
| mileage x urbanicity | 1.147062 | -0.000040 | 1.144976 | +0.000118 | [-0.001228, 0.001431] |
| education x mileage | 1.147739 | -0.000718 | 1.145438 | -0.000343 | [-0.001403, 0.000602] |

The only positive pooled point estimate is mileage x urbanicity in the blend,
and its gain is just 0.000118 with a wide CI crossing zero. SHAP successfully
nominates real nonlinear structure for screening, but it does not identify a
generalizable improvement here.

## Reproduction

From the repository root:

```powershell
$env:CODEX_STAGE = "screen"
$env:CODEX_N_BAGS = "15"
Rscript R/codex_mlogit_bagging.R

$env:CODEX_STAGE = "cv"
Rscript R/codex_mlogit_bagging.R

$env:CODEX_STAGE = "screen"
Rscript R/codex_partial_pooling.R

$env:CODEX_STAGE = "discover"
$env:CODEX_TOP_PAIRS = "8"
Rscript R/codex_shap_interactions.R

$env:CODEX_STAGE = "screen"
Rscript R/codex_shap_interactions.R

$env:CODEX_STAGE = "cv"
Rscript R/codex_shap_interactions.R
```

Generated diagnostics and OOF matrices are written below
`data_processed/codex_final_round/` (gitignored). No Kaggle submission is
created by any script.
