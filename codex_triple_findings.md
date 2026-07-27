# Continuous higher-order interaction findings

Branch: `codex-triple-products`, based on `zhenhao` at `4204e79`.

## Verdict

No candidate clears the project's submission bar.

The best result is one additional term,
`Price × standardized incomea × standardized milesa`. It improves the
canonical fixed 80/20 m8trpg+xgboost OOF blend from 1.145094 to 1.144599, a
point gain of 0.000495. The respondent-clustered bootstrap 95% confidence
interval is `[-0.000689, 0.001660]`, which crosses zero. It is therefore an
interesting but unconfirmed effect, not a submission candidate.

Segment-specific mileage slopes are decisively harmful. All other age, night,
and combined specifications fail at the screen stage.

## Candidate definitions

All continuous covariates are standardized using the applicable choice-training
respondents only. No continuous covariate is binned.

The following three pairs were tested:

- income × age;
- income × annual mileage;
- income × night-driving share.

For each pair, the price triple
`Price × z(covariate 1) × z(covariate 2)` and the non-price analogue
`inside × z(covariate 1) × z(covariate 2)` were screened individually and
together. A six-term model containing all three price/inside pairs was also
screened.

The segment family adds deviations from the existing global continuous price
slope:

`Price × z(covariate) × I(segment = s)`, for segments 2--6.

Income, age, mileage, and night-driving slopes were screened separately, plus
one 20-term combined model.

## Segment support

The six training segment counts are 128, 383, 49, 211, 58, and 306
respondents. The two smallest segments are not empty or constant on the four
continuous covariates, but they are only about as large as the small persona
clusters previously flagged as high variance. Segment-specific results were
therefore treated cautiously even though continuous slopes do not have the
same sparse-bin quasi-separation mechanism as the earlier categorical model.

The seed-7402 screen-training counts are 98, 302, 35, 183, 46, and 244.

## Single-split screen

The baseline is exactly the canonical m8trpg validation loss, 1.159681.
Negative deltas are improvements.

| Candidate | Added terms | Validation loss | Delta |
|---|---:|---:|---:|
| Price + inside: income × age | 2 | 1.166595 | +0.006913 |
| Price + inside: income × mileage | 2 | **1.157562** | **-0.002119** |
| Price + inside: income × night | 2 | 1.161400 | +0.001718 |
| All three price + inside pairs | 6 | 1.164964 | +0.005282 |
| Price only: income × age | 1 | 1.166929 | +0.007247 |
| Inside only: income × age | 1 | 1.163104 | +0.003423 |
| Price only: income × mileage | 1 | **1.157576** | **-0.002106** |
| Inside only: income × mileage | 1 | **1.158883** | **-0.000799** |
| Price only: income × night | 1 | 1.161449 | +0.001767 |
| Inside only: income × night | 1 | 1.160195 | +0.000514 |
| Segment-specific income price slopes | 5 | 1.166445 | +0.006764 |
| Segment-specific age price slopes | 5 | 1.164854 | +0.005172 |
| Segment-specific mileage price slopes | 5 | **1.158403** | **-0.001278** |
| Segment-specific night price slopes | 5 | 1.162025 | +0.002344 |
| All segment-specific slopes | 20 | 1.171609 | +0.011928 |

Every screen-positive candidate advanced to canonical five-fold CV. No
screen-negative candidate was promoted.

## Canonical five-fold CV

CV uses the saved respondent folds from seed 4821. Each candidate is refit from
scratch within every fold, and predictions are matched back by row identifier.
The script asserts that the saved m8trpg OOF loss is exactly 1.147021 and that
the fixed 80/20 ensemble loss is exactly 1.145094 before comparisons proceed.

| Candidate | m8trpg-family loss | Gain vs. m8trpg | Fixed 80/20 blend | Gain vs. v11 |
|---|---:|---:|---:|---:|
| Price + inside: income × mileage | 1.146849 | +0.000172 | 1.144642 | +0.000452 |
| Price only: income × mileage | **1.146782** | **+0.000239** | **1.144599** | **+0.000495** |
| Inside only: income × mileage | 1.147213 | -0.000192 | 1.145041 | +0.000053 |
| Segment-specific mileage slopes | 1.151168 | -0.004147 | 1.147735 | -0.002640 |

The price-only term is directionally stable: its coefficient is negative in
all five folds, ranging from -0.0541 to -0.0270. Its predictive gain is not
stable enough: three folds improve and two worsen.

The segment model suffers a large fold-4 regression and is worse in four of
five folds. Its coefficients remain numerically moderate, so this is not a
repeat of the earlier coefficient explosion; the pattern is consistent with
high-variance overfitting from estimating several subgroup slopes.

## Respondent-clustered bootstrap

The bootstrap resamples the 1,135 respondents, not individual choice rows, for
2,000 replicates. Gain is baseline loss minus candidate loss.

| Candidate | Comparison | Point gain | 95% CI | Win rate |
|---|---|---:|---:|---:|
| Price + inside: income × mileage | m8trpg | +0.000172 | [-0.001579, 0.001810] | 59.3% |
| Price + inside: income × mileage | fixed 80/20 blend | +0.000452 | [-0.000738, 0.001615] | 78.3% |
| Price only: income × mileage | m8trpg | +0.000239 | [-0.001464, 0.001875] | 62.2% |
| Price only: income × mileage | fixed 80/20 blend | **+0.000495** | **[-0.000689, 0.001660]** | 80.7% |
| Inside only: income × mileage | fixed 80/20 blend | +0.000053 | [-0.000598, 0.000719] | 56.8% |
| Segment-specific mileage slopes | fixed 80/20 blend | -0.002640 | [-0.005229, -0.000563] | 0.3% |

As a diagnostic only, re-optimizing the mlogit blend weight on the same pooled
OOF predictions selects 82% for the price-only candidate and gives 1.144571.
Its gain is 0.000523 with CI `[-0.000723, 0.001708]`. Because the weight was
selected on the evaluated OOF data, the fixed 80/20 comparison above is the
primary, less optimistic result. Neither clears zero.

## Interpretation

There may be a weak interaction in which the effect of mileage on price
sensitivity changes with income. The negative coefficient is stable across
folds, and both the mlogit and ensemble point estimates move in the desired
direction. However, the predictive improvement is several times smaller than
its sampling uncertainty and is below the gain needed to confidently move a
1.202 public score under 1.200.

The correct decision under the pre-specified bar is to retain `ensemble_v11`
and not spend a Kaggle submission slot. This is a marginal, unconfirmed clue
that could be revisited only if independent evidence appears; it is not a
validated improvement.

## Reproduction

Generated CSV and OOF artifacts are under the gitignored
`data_processed/codex_triples/` directory.

```powershell
$env:CODEX_STAGE = "screen"
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" R\codex_triple_interactions.R

$env:CODEX_STAGE = "cv"
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" R\codex_triple_interactions.R
```

New files:

- `R/codex_triple_common.R`
- `R/codex_triple_interactions.R`
- `codex_triple_findings.md`

No existing R script, `AGENTS.md`, `cleaning_log.md`, or
`submissions_log.csv` was modified.
