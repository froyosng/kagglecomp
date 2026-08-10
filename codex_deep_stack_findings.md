# Deep-learning, full-stacking, and LightGBM round

## Decision

No candidate clears the project's stated multiplicity-aware submission bar.
No submission file was generated and nothing was submitted to Kaggle.

The round nevertheless produced two useful results:

1. R torch is installable and a real two-layer dropout network contains
   independent signal, but it is not significantly better than the submitted
   shallow-MLP ensemble.
2. An eight-component arithmetic blend reaches the best CV point estimate in
   the project, 1.142112, and improves all five folds. Its ordinary 95%
   bootstrap interval barely excludes zero, but its 99% and six-candidate
   multiplicity-adjusted intervals cross zero. It therefore does not meet the
   predeclared threshold.

Neither result is remotely large enough to explain a public score difference
from 1.201 to 1.186.

## Environment discovery

The earlier statement that a deep-learning framework was unavailable was an
environmental limitation, not a permanent one.

- `torch` 0.17.0 installed successfully from the CRAN Windows binary.
- The CPU libtorch 2.8.0 runtime installed successfully and passed tensor and
  training smoke tests. CUDA is unavailable, so all training used CPU.
- `lightgbm` 4.7.0 installed successfully from the CRAN Windows binary.
- CatBoost was not pursued after LightGBM installed cleanly and all three
  LightGBM configurations failed the required screen.

The package binaries and native runtime live in the user's R library and are
not committed to the repository.

## 1. Deep torch network

### Screen and frozen specification

`R/codex_torch_deep_mlp.R` reuses the exact 306-column feature construction
from `R/codex_mlp_ensemble.R`. Five configurations were screened on the
canonical seed-7402 respondent split.

The initial 60-epoch networks overfit sharply: training loss fell as low as
0.32 while validation component loss rose to 1.69--2.16. Training curves
motivated two explicitly counted, more strongly regularized early-stopped
configurations.

The frozen CV candidate was:

- hidden layers: 128 and 64 units;
- ReLU activations;
- dropout: 0.30;
- Adam learning rate: 0.001;
- L2 weight decay: 0.001;
- 12 epochs;
- batch size: 256;
- two seeds during screening and three seeds per outer CV fold.

Its screen component loss was 1.223495. Blended into the submitted model, the
screen gain was 0.000680.

### Canonical five-fold CV

| Model | CV log loss | Gain |
|---|---:|---:|
| ensemble_v11 | 1.145094 | -- |
| Submitted shallow-MLP model | 1.143789 | -- |
| Deep component alone | 1.193140 | -- |
| v11 with deep MLP replacing shallow MLP | 1.143146 | +0.001948 vs. v11 |
| Submitted model plus deep MLP | 1.143169 | +0.000620 vs. submitted |
| Joint arithmetic v11/shallow/deep blend | **1.143030** | +0.000759 vs. submitted |

The replacement deep weight is stable at 0.16--0.19 across folds. In the
joint blend, the deep network receives 0.117--0.159 weight and the shallow
MLP retains 0.050--0.094.

High-precision 100,000-draw respondent bootstrap:

| Comparison | Point gain | Ordinary 95% CI | Bonferroni CI |
|---|---:|---:|---:|
| Deep replacement vs. v11 | 0.001948 | [0.000386, 0.003513] | [-0.000099, 0.004017] |
| Submitted + deep vs. submitted | 0.000620 | [-0.000310, 0.001552] | [-0.000600, 0.001845] |
| Joint shallow/deep blend vs. submitted | 0.000759 | [-0.000188, 0.001714] | [-0.000486, 0.002018] |

The deep framework independently reproduces the neural-diversity mechanism,
but it does not clear the comparison that matters: improvement over the
already-submitted shallow-MLP model.

## 2. Full cached-OOF stacking

`R/codex_full_stacking.R` hard-checks and aligns these seven requested
components:

1. m8trpg mlogit;
2. original xgboost;
3. rank:ndcg xgboost;
4. retuned xgboost;
5. glmnet-Cox;
6. shallow MLP; and
7. triple-interaction mlogit.

An augmented eight-component pool also includes the new deep torch MLP.

Three combination methods were evaluated:

- fold-cross-fitted arithmetic simplex weights, as a required sanity check;
- a conditional-logit/logarithmic pool with ridge selected through inner
  folds; and
- a shallow multiclass xgboost meta-model with its configuration selected
  through inner folds.

The ridge search initially selected its largest tested value, 0.01, in every
fold. A dedicated refinement tested 0.03, 0.10, and 0.30. All were worse in
every outer fold's inner selection, confirming that 0.01 is the real optimum
rather than an unresolved boundary.

### Results

| Pool and method | CV log loss | Gain vs. submitted |
|---|---:|---:|
| Requested 7, arithmetic | 1.142878 | +0.000911 |
| Requested 7, ridge log-pool | 1.143792 | -0.000002 |
| Requested 7, meta-xgboost | 1.152838 | -0.009049 |
| Augmented 8, arithmetic | **1.142112** | **+0.001678** |
| Augmented 8, ridge log-pool | 1.143155 | +0.000634 |
| Augmented 8, meta-xgboost | 1.152255 | -0.008466 |

The requested learned stacker does not win. The seven-component ridge
log-pool is exactly tied with the submitted model, the augmented log-pool is
too uncertain, and both GBM stackers are decisively harmful. Inner CV chooses
the depth-2 meta-xgboost in every fold, so the negative result is not caused
by inconsistent configuration selection.

The arithmetic sanity check is more interesting. The augmented blend improves
the submitted model in every fold:

| Fold | Gain |
|---:|---:|
| 1 | +0.000353 |
| 2 | +0.002538 |
| 3 | +0.003545 |
| 4 | +0.001209 |
| 5 | +0.000743 |

Its mean fold-cross-fitted weights are:

| Component | Mean weight |
|---|---:|
| Triple-interaction mlogit | 0.4793 |
| Deep MLP | 0.1335 |
| glmnet-Cox | 0.1311 |
| rank:ndcg xgboost | 0.0933 |
| Shallow MLP | 0.0801 |
| Original mlogit | 0.0625 |
| Retuned xgboost | 0.0156 |
| Original xgboost | 0.0047 |

### High-precision bootstrap of the arithmetic result

For the augmented arithmetic blend versus the submitted model:

- point gain: 0.001677532;
- bootstrap SD: 0.000824262;
- win rate: 97.824%;
- ordinary 95% CI: **[0.0000466, 0.0032790]**;
- 99% CI: [-0.0004625, 0.0037796];
- six-candidate Bonferroni CI: [-0.0005191, 0.0038290].

The ordinary interval excludes zero by only 0.000047 and the required
multiplicity-aware interval crosses zero. Direct comparisons against the
previous five-family arithmetic blend and the deep joint blend also cross
zero. This is the best point estimate of the round, but it is not a
submission-grade confirmation under the requested rule.

## 3. LightGBM

`R/codex_lightgbm_ensemble.R` uses native categorical splits for the 19
attribute levels by alternative, Price levels, and categorical respondent
covariates. This is materially different from the existing xgboost treatment
of integer codes as numeric inputs.

Three regularized configurations were screened:

| Configuration | Component loss | Best weight with submitted model |
|---|---:|---:|
| 15 leaves, depth 5 | 1.204818 | 0.00 |
| 31 leaves, depth 6 | 1.207183 | 0.00 |
| Conservative 15 leaves, depth 4 | 1.209857 | 0.00 |

All three receive exactly zero blend weight. Following the project's
screen-first rule, LightGBM did not proceed to five-fold CV.

## Reproducibility

Tracked scripts:

- `R/codex_torch_deep_mlp.R`
- `R/codex_full_stacking.R`
- `R/codex_full_stacking_precision.R`
- `R/codex_lightgbm_ensemble.R`

Generated OOF predictions, checkpoints, bootstrap draws, and result CSVs are
under `data_processed/codex_deep_stack/` and remain gitignored.

The branch started from `zhenhao` commit `1c23513`. `AGENTS.md`,
`cleaning_log.md`, `submissions_log.csv`, existing R scripts, and existing
submission files were not modified.
