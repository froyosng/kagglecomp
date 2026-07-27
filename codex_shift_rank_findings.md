# Covariate-shift refit and attribute-rank findings

Branch: `codex-shift-ranks`, based on `zhenhao` at `93723cc`.

## Verdict

Neither proposed lead improves the current model under the project's established
validation protocol. No Kaggle submission is warranted.

- Importance-weighted refitting is a clear negative. Increasing the density-ratio
  weight strength monotonically worsens both ordinary and target-weighted loss.
  The result persists in canonical five-fold respondent-grouped CV and after
  re-optimizing the mlogit/xgboost blend.
- Per-attribute min/max flags showed a small single-split gain, but it reversed in
  five-fold CV. Its respondent-bootstrap confidence interval crosses zero.
- The current `ensemble_v11` remains the submission choice.

## 1. Importance-weighted refit

### Method

The experiment implements the proposed covariate-shift correction as an actual
weighted likelihood refit, not merely a reweighted evaluation.

For every outer split:

1. Fit a respondent-level logistic domain classifier to distinguish the
   choice-training respondents from all test respondents. The predictors are the
   same 15 covariates used in `R/adversarial_validation.R`.
2. Convert the test propensity to a density ratio:
   `p(test | x) / (1 - p(test | x)) * n_source / n_target`.
3. Cap the ratio at 20, normalize it to mean one, and fit m8trpg with
   `ratio^alpha` respondent weights for `alpha = 0, 0.25, 0.5, 0.75, 1`.
4. Evaluate every alpha against the same held-out target-risk objective, using
   full-strength density-ratio weights estimated without the held-out choice
   respondents. This makes the comparison internally consistent: only the fit
   weights vary across alpha.

This follows the weighted-likelihood covariate-shift idea in
[Shimodaira (2000)](https://doi.org/10.1016/S0378-3758(00)00115-4). It relies on
the usual covariate-shift assumption that the conditional choice mechanism is
stable after conditioning on the observed covariates.

### Cheap screen

The canonical seed-7402 respondent split was used first. `alpha = 0` exactly
reproduces the saved m8trpg validation loss, 1.159681.

| Alpha | Ordinary loss | Target-weighted loss | Fit ESS (of 908 respondents) |
|---:|---:|---:|---:|
| 0.00 | **1.159681** | **1.178559** | 908.0 |
| 0.25 | 1.160221 | 1.179503 | 887.0 |
| 0.50 | 1.161069 | 1.180813 | 816.1 |
| 0.75 | 1.162406 | 1.182748 | 688.9 |
| 1.00 | 1.164606 | 1.185727 | 520.8 |

The same monotonic ordering holds when evaluation ratios are capped at 5 or 10.
The screen therefore prefers no refit weighting even under the proposed
target-weighted objective.

### Canonical five-fold confirmation

The complete five-fold respondent-grouped CV re-estimates the domain classifier
inside each fold, using seed 4821's saved fold assignment. Alpha zero reproduces
the saved m8trpg OOF prediction exactly.

| Alpha | Ordinary m8trpg | Target-weighted m8trpg | Target delta vs. alpha 0 | Best mlogit weight with xgb | Ordinary blend | Target-weighted blend | Target blend delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | **1.147021** | **1.159787** | 0 | 0.77 | **1.145222** | **1.155453** | 0 |
| 0.25 | 1.147527 | 1.161203 | +0.001416 | 0.76 | 1.145595 | 1.156342 | +0.000890 |
| 0.50 | 1.148593 | 1.163516 | +0.003729 | 0.74 | 1.146288 | 1.157621 | +0.002169 |
| 0.75 | 1.150453 | 1.166597 | +0.006810 | 0.71 | 1.147350 | 1.159116 | +0.003663 |
| 1.00 | 1.153587 | 1.170789 | +0.011002 | 0.68 | 1.148788 | 1.160844 | +0.005392 |

Positive deltas mean worse loss. Even the gentlest nonzero refit is worse in
all five individual folds.

The paired respondent bootstrap defines gain as baseline loss minus candidate
loss. For alpha 0.25:

- m8trpg gain: -0.001416; 95% CI [-0.002614, -0.000422];
- reweighted-refit blend gain: -0.000890; 95% CI
  [-0.001398, -0.000276].

Thus, the confidence intervals do not merely cross zero; they exclude zero in
the wrong direction. Full weighting is worse still. The target-optimal blend
weights above are selected on the pooled OOF target loss and are therefore only
a diagnostic, but this possible optimism cannot rescue a result that is already
negative.

### Interpretation and limits

The covariate shift is real, but weighting is not automatically beneficial. Here
it reduces the effective sample size sharply while asking a finite, parametric
choice model to fit relatively sparse test-like regions. At full strength, the
mean training-fold respondent ESS falls from 908 to about 512; the mean
held-out evaluation ESS is about 121 respondents. The variance and coefficient
instability cost dominates any benefit from targeting the shifted income mix.

This does not disprove covariate shift or show that the public test distribution
matches training. It says the particular observable density-ratio correction,
under its identifying assumption and with defensible stabilization, makes this
model worse on an honest proxy for target risk.

## 2. Per-attribute choice-set min/max features

### Method

For each of the 19 attribute level codes, add two indicators for each inside
alternative:

- its level is the minimum among alternatives 1--3;
- its level is the maximum among alternatives 1--3.

Ties retain multiple flags. This adds 38 terms to the full m8trpg formula. The
inside-only definition avoids treating the opt-out's structural zero as a real
attribute level.

For completeness, an all-four-alternative version was screened too. It is
computationally singular because alternative 4 is structurally zero for every
attribute, recreating the project's known inside/opt-out identification problem.
It was discarded rather than regularized into a different model.

### Results

The inside-only version passed the cheap screen:

| Model | Seed-7402 validation loss | Delta vs. m8trpg |
|---|---:|---:|
| m8trpg | 1.159681 | -- |
| m8trpg + 38 attribute min/max flags | **1.158879** | -0.000803 |

The apparent gain did not replicate:

| Model | Five-fold CV |
|---|---:|
| m8trpg | **1.147021** |
| m8trpg + attribute min/max flags | 1.147275 |

The point gain is -0.000254, with respondent-bootstrap 95% CI
[-0.001138, 0.000572] and only 26.7% positive bootstrap draws. Four of the five
folds are worse.

The price analogy has an important conceptual limit: price codes are ordered and
their direction has a stable economic meaning, whereas many attribute codes are
categorical labels rather than cardinal or consistently ordered quality scores.
Min/max comparisons on those labels can add arbitrary splits, which is
consistent with the failure to generalize.

## Reproduction

All generated result files and OOF matrices are under the gitignored
`data_processed/codex_shift/` directory.

```powershell
$env:CODEX_STAGE = "screen"
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" R\codex_importance_refit.R

$env:CODEX_STAGE = "cv"
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" R\codex_importance_refit.R

$env:CODEX_STAGE = "screen"
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" R\codex_attribute_ranks.R

$env:CODEX_STAGE = "cv"
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" R\codex_attribute_ranks.R
```

New scripts:

- `R/codex_shift_common.R`
- `R/codex_importance_refit.R`
- `R/codex_attribute_ranks.R`

No existing `R/` script, `AGENTS.md`, `cleaning_log.md`, or
`submissions_log.csv` was modified.
