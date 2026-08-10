# Codex modeling experiments (2026-07-27)

## Bottom line

None of the three requested ideas produces a statistically secure replacement
for `ensemble_v11` (official CV 1.145094, public 1.202). All three are
legitimate and at least mildly useful:

- A genuine `rank:ndcg` xgboost is much stronger than the old xgboost alone
  (cross-fitted CV 1.163610 versus 1.178668), but its errors overlap more with
  m8trpg. Its two-way blend is only 1.144904.
- Retuning makes the multiclass xgboost slightly stronger and, more
  importantly, a slightly better blend partner. The selected two-way blend is
  1.144610 using the global OOF weight search, or 1.144716 when the blend
  weight is learned on four folds and applied to the fifth.
- The recreated glmnet-Cox model scores 1.164331 in fully nested OOF CV and
  receives a stable 11%-13% ensemble weight. With m8trpg and retuned xgboost,
  its meta-cross-fitted three-way score is 1.144474.

The best combined candidate uses m8trpg, the ranker, retuned multiclass
xgboost, and Cox. Its score is:

- **1.144029** with weights optimized on all OOF predictions;
- **1.144363** when weights are learned on four folds and applied to the fifth.

The latter is the honest number to use. Its estimated improvement over the
fixed 0.80/0.20 `ensemble_v11` predictions is **0.000742**. In 1,000 paired
respondent bootstraps it wins 93.3% of samples, but the 95% interval for the
gain is **[-0.000214, 0.001775]**, which includes zero. The bootstrap also holds
the already-selected model pool fixed, so it does not include model-selection
uncertainty and should not be read as stronger evidence than it is.

**Recommendation:** keep `ensemble_v11` as the official incumbent. If the team
wants to spend one submission slot on a variance-reduction candidate, the
four-component blend is the most defensible option found here, but it is only a
borderline CV gain. The expected gain is much smaller than the 0.002 needed to
move 1.202 below 1.200, and much smaller than the established public-sample SD
of about 0.0225. There is no honest basis to promise a sub-1.200 public score.

No Kaggle submission was made.

## Validation protocol

All experiments use the existing project protocol:

- single 80/20 respondent screen, seed 7402;
- canonical five-fold respondent assignment read from
  `data_processed/oof_ensemble_v10.rds` (seed 4821);
- pooled OOF log loss, never a mean of fold scores;
- m8trpg OOF predictions reused from the same artifact, preserving exact row
  alignment;
- ensemble-weight meta-validation: learn weights using four OOF folds and
  apply them to the fifth;
- paired uncertainty resampled by respondent, not by row.

The new xgboost hyperparameters were chosen by the single split before their
five-fold confirmation. Cox lambda selection was nested inside every outer
fold. The ranker's scalar probability calibration was also cross-fitted: its
temperature was learned on four folds' raw margins and applied to the fifth.

## 1. Genuine ranking-objective xgboost

Script: `R/codex_rank_xgb.R`

This is not the previously rejected binary-chosen model. Each choice task is a
four-row xgboost query (`qid = choice task`), and the loss compares
alternatives within that query. Features are the current alternative's 19
attributes, price, alternative/task indicators, price rank/gap context, and
respondent covariates. Raw ranking margins are converted to four-way
probabilities with a within-task softmax and a cross-fitted scalar temperature.

The single-split screen covered both `rank:pairwise` and `rank:ndcg`, learning
rates 0.05/0.10, depths 3/4/6, and 75/150/300 rounds (36 configurations).

| Objective | eta | depth | rounds | Single split | Five-fold CV | Best two-way blend |
|---|---:|---:|---:|---:|---:|---:|
| `rank:ndcg` | 0.10 | 4 | 300 | 1.193073 | **1.163610** | **1.144904** (76% m8trpg) |
| `rank:pairwise` | 0.10 | 4 | 300 | 1.193775 | 1.168657 | 1.145235 (80% m8trpg) |

The ranker is a real standalone improvement over multiclass xgboost, but it
does not replace m8trpg and gives only a 0.00019 global-blend improvement over
the official 1.145094 ensemble. This distinction is important: a stronger
component is not automatically a better ensemble component if it makes more
similar errors.

## 2. Cox as a third component

Script: `R/codex_glmnet_cox_ensemble.R`

The original interaction-selection implementation was not present in Git, so
the logged specification was reconstructed exactly:

- 63 unpenalized core columns: all non-reference levels of the 19 attributes,
  continuous Price, and alternative-2/3 dummies;
- 195 penalized candidates: Price/inside by seven covariates, Price/inside by
  segment, every attribute by segment, and every attribute by
  income/age/mileage/night driving;
- `glmnet(family = "cox", alpha = 1)` using choice task as the stratum;
- respondent-grouped inner folds for lambda selection.

The reconstruction passes the historical check. On the canonical single split,
`lambda.min` scores **1.194574** with 16 interactions, close to the logged
1.19542 with 13. `lambda.1se` scores 1.211697 and is clearly too sparse.

For honest component predictions, each outer fold runs a new inner
respondent-grouped `cv.glmnet`. The resulting OOF scores are:

| Cox rule | OOF log loss |
|---|---:|
| `lambda.min` | **1.164331** |
| `lambda.1se` | 1.178006 |

The selected interaction count at `lambda.min` is 14, 11, 25, 12, and 11
across the five folds. Cox is weaker alone than the ranker but different enough
to receive positive ensemble weight.

| Three-way pool | Global OOF | Weight-cross-fitted |
|---|---:|---:|
| m8trpg + original xgboost + Cox | 1.144635 | 1.145024 |
| m8trpg + ranker + Cox | 1.144433 | 1.144698 |
| m8trpg + retuned xgboost + Cox | **1.144202** | **1.144474** |

The requested original-xgboost/Cox three-way experiment is therefore mildly
positive, but its meta-CV gain over the comparable two-way pool (1.145312) is
only 0.000288.

## 3. Multiclass xgboost retune

Script: `R/codex_xgb_retune.R`

The single-split screen evaluated the exact old baseline plus 23 seeded,
space-filling configurations covering:

- `max_depth` 2-6;
- `min_child_weight` 1/3/8/15;
- `subsample` and `colsample_bytree` 0.65/0.8/1.0;
- `reg_alpha` 0/0.25/1/3;
- `reg_lambda` 0.5/1/5/15;
- `gamma` 0/0.1/0.5;
- `eta` 0.03/0.05/0.08/0.10, with a matched number of rounds.

The old setting reproduces at 1.202169 on the current runtime. The top three
screened configurations were then refit on all five canonical folds:

| eta | depth | child | subsample | colsample | alpha | lambda | gamma | rounds | XGB CV | Best blend |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 6 | 8 | 0.65 | 0.80 | 0 | 5 | 0 | 243 | 1.176029 | **1.144610** (80% m8trpg) |
| 0.05 | 6 | 1 | 0.65 | 0.80 | 3 | 5 | 0.1 | 146 | 1.176219 | 1.144613 (80%) |
| 0.08 | 3 | 1 | 0.65 | 0.65 | 3 | 15 | 0.1 | 137 | **1.175878** | 1.145034 (82%) |

The third setting is best alone, but the first is the better blend partner.
This again confirms that diversity, not standalone score alone, determines
ensemble value.

## Combined ensemble and uncertainty

Script: `R/codex_ensemble_diagnostics.R`

The script searches arithmetic simplex weights, then performs a meta-level
five-fold check in which weights are relearned without the held-out fold.

| Component pool | Global OOF | Weight-cross-fitted |
|---|---:|---:|
| m8trpg + original xgboost | 1.145063 | 1.145312 |
| m8trpg + ranker | 1.144904 | 1.145024 |
| m8trpg + retuned xgboost | 1.144610 | 1.144716 |
| m8trpg + retuned xgboost + Cox | 1.144202 | 1.144474 |
| m8trpg + ranker + retuned xgboost | 1.144324 | 1.144507 |
| **m8trpg + ranker + retuned xgboost + Cox** | **1.144029** | **1.144363** |
| Same four + original xgboost | 1.144037 | 1.144399 |

The original xgboost receives only 0.4% in the five-tree-family pool, so it is
fully superseded when the ranker and retuned multiclass model are available.
The preferred global deployment weights are:

- m8trpg: **0.6793**
- `rank:ndcg`: **0.0964**
- retuned multiclass xgboost: **0.1153**
- Cox `lambda.min`: **0.1090**

Fold-learned weights are reasonably stable: m8trpg 0.638-0.716, ranker
0.069-0.127, retuned xgboost 0.093-0.140, and Cox 0.074-0.148. That stability
supports a real diversity mechanism, although the total loss gain remains
small.

## Reproduction

Run from the repository root:

```powershell
$env:CODEX_STAGE='screen'
Rscript R/codex_rank_xgb.R
$env:CODEX_STAGE='cv'
Rscript R/codex_rank_xgb.R

$env:CODEX_STAGE='screen'
Rscript R/codex_xgb_retune.R
$env:CODEX_STAGE='cv'
Rscript R/codex_xgb_retune.R

$env:CODEX_STAGE='screen'
Rscript R/codex_glmnet_cox_ensemble.R
$env:CODEX_STAGE='cv'
Rscript R/codex_glmnet_cox_ensemble.R

Remove-Item Env:CODEX_STAGE
Rscript R/codex_ensemble_diagnostics.R
```

Intermediate predictions and CSV summaries are written under
`data_processed/codex/`, which is intentionally gitignored. The Cox CV is the
slow step (about 12 minutes on this machine) because it performs five nested
penalty searches.
