# MLP seed-bagging follow-up

## Decision

Increasing the frozen MLP candidate from 5 to 20 initialization seeds makes
the MLP component substantially better, but it does **not** establish that the
resulting ensemble is better than the existing 5-seed candidate.

The 20-seed ensemble improves the point estimate against `ensemble_v11`
(gain 0.001956 versus 0.001305 for 5 seeds), but:

- its respondent-bootstrap 95% interval is wider, not tighter;
- its 99% and Bonferroni-adjusted intervals still cross zero; and
- the direct 20-seed-versus-5-seed comparison has a 95% interval that crosses
  zero.

Therefore the requested trigger for regenerating a candidate submission was
not met. No new submission CSV was created, the existing
`submission_codex_mlp_v12_candidate.csv` was not modified, and nothing was
submitted to Kaggle.

## Frozen specification and reproducibility checks

`R/codex_mlp_seed_bagging.R` changes only the number of MLP initializations.
It retains:

- 8 hidden units;
- decay 0.1;
- the 200-iteration cap;
- the canonical respondent-grouped five folds;
- the original data preparation and feature matrix;
- the same 0.00--0.30 blend-weight grid; and
- the same fold-cross-fitted weight-selection procedure.

For fold `k`, seeds are
`4821 + 100 * k + 0:19`. Thus the first five seeds in every fold are exactly
the seeds used by the existing candidate. All 100 fits are separately
checkpointed.

Before evaluating the extension, the script hard-asserts the saved mlogit
baseline and `ensemble_v11` losses. The cumulative five-seed OOF prediction
reproduces the existing saved MLP OOF prediction with maximum absolute
difference `2.22e-16`. It also reproduces:

- MLP component loss: 1.190543350; and
- fold-cross-fitted blend loss: 1.143789442.

All 100 fits reached the deliberately frozen 200-iteration cap (the same early
stopping behavior as the original candidate). One recorded elapsed time
includes a temporary system suspension; checkpointing allowed the run to
resume without changing any fit.

## Learning curve

The table reports honest fold-cross-fitted blend losses and weights, not a
same-OOF globally optimized blend.

| Seeds | MLP loss | Global weight | Cross-fitted blend loss | Gain vs. v11 |
|---:|---:|---:|---:|---:|
| 1 | 1.375457 | 0.05 | 1.144599 | 0.000495 |
| 5 | 1.190543 | 0.15 | 1.143789 | 0.001305 |
| 10 | 1.174632 | 0.20 | 1.143193 | 0.001902 |
| 15 | 1.170208 | 0.22 | **1.143073** | **0.002021** |
| 20 | **1.168530** | 0.22 | 1.143139 | 0.001956 |

The component improves nearly monotonically as seeds are added. Ensemble
returns flatten around 15 seeds: the point-best cumulative ensemble is at 15,
then fluctuates slightly through 20. Since 15 is visible only after inspecting
the same learning curve, it is a post-hoc checkpoint rather than a separately
validated candidate and is not promoted.

The 20-seed fold-cross-fitted MLP weights are 0.21, 0.26, 0.23, 0.22, and
0.20. The corresponding 5-seed weights are 0.14, 0.17, 0.15, 0.15, and 0.13.

## High-precision respondent bootstrap

`R/codex_mlp_seed_precision.R` uses 100,000 shared respondent-clustered
bootstrap samples. Positive gain means the model named first is better.

| Comparison | Point gain | Bootstrap SD | 95% interval | 99% interval | Bonferroni interval | Win rate |
|---|---:|---:|---:|---:|---:|---:|
| 20-seed MLP vs. 5-seed MLP | 0.022013 | 0.001855 | [0.018394, 0.025656] | [0.017273, 0.026820] | [0.016920, 0.027184] | 100.00% |
| 5-seed blend vs. v11 | 0.001305 | 0.000617 | [0.000092, 0.002513] | [-0.000276, 0.002897] | [-0.000396, 0.003007] | 98.24% |
| 20-seed blend vs. v11 | 0.001956 | 0.000853 | [0.000287, 0.003626] | [-0.000243, 0.004160] | [-0.000402, 0.004331] | 98.94% |
| 20-seed blend vs. 5-seed blend | 0.000651 | 0.000390 | [-0.000122, 0.001416] | [-0.000349, 0.001650] | [-0.000423, 0.001728] | 95.20% |

The Bonferroni interval uses the same nine-comparison family convention as the
original precision analysis.

More seeds unambiguously improve the MLP component. They do not tighten the
ensemble interval because the stronger component receives materially more
weight. That increases the ensemble's exposure to respondent-level variation:
the 95% interval width versus v11 grows from 0.002421 at 5 seeds to 0.003339 at
20 seeds. The lower 95% bound improves, but the stricter intervals remain
inconclusive.

## Files

- `R/codex_mlp_seed_bagging.R`: resumable 20-seed CV run, exact five-seed
  reproduction, cumulative learning curve, and fold-cross-fitted predictions.
- `R/codex_mlp_seed_precision.R`: shared 100,000-draw respondent bootstrap.
- Generated artifacts are under
  `data_processed/codex_mlp_seed_bagging/` and remain gitignored.

Official project logs were treated as read-only and were not edited.
