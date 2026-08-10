# Questionnaire-version opt-out corrections: findings

## Verdict

**Not adopted; no candidate submission file generated and no Kaggle submission
made.** The requested empirical-Bayes mean-residual correction looked promising
on the cheap respondent-held-out screen, then worsened both a freshly refit
frozen pipeline and the exact saved OOF prediction corresponding to the
submitted 15%-MLP blend under nested five-fold CV. A more principled
Newton/curvature sensitivity version also worsened both baselines. Both
respondent-clustered confidence intervals cross zero.

The current `ensemble_v11 + 0.15 MLP` submission remains the best submission.
The positive screen was not discarded or reinterpreted: it is reported below as
a concrete example of a promising single split failing honest nested CV.

## Requested mean-residual method

The run starts from `zhenhao` commit `5a8d601` and reuses
`data_processed/questionnaire_fingerprints.rds` directly. It does not reconstruct
the 299 questionnaire fingerprints.

For the primary requested experiment, version \(v\) receives one additive
opt-out utility shift:

\[
\delta_v =
\frac{n_v}{n_v+\alpha}
\operatorname{mean}_{i\in v}(y_{i4}-p_{i4}),
\qquad
\alpha \in \{20,40,60,80,100,\infty\}.
\]

Here \(n_v\) is the number of training tasks supporting the version. The shift
is applied by adding \(\delta_v\) to `log(p4)` and re-softmaxing the four
probabilities. `Inf` means full shrinkage/no correction.

The canonical screen used the seed-7402 respondent split. For nested five-fold
CV, each outer fold used the other four canonical folds as inner validation
folds; alpha was selected on pooled inner-held-out loss. Final deltas used only
outer-training respondents' calibration residuals (computed after fitting the
base architecture on that same training partition), and were applied to the
untouched outer fold. The outer comparison uses the exact fixed-15% MLP OOF
baseline, asserted at `1.143686618`.

## Newton sensitivity method

Because a probability residual is not naturally measured in utility units, a
secondary sensitivity check used the conditional-logit gradient and curvature.
For version \(v\):

\[
g_v = \sum_i (p_{i4}-y_{i4}), \qquad
h_v = \sum_i p_{i4}(1-p_{i4}), \qquad
\delta_v = -\frac{g_v}{h_v+\lambda}.
\]

A finite-difference audit reproduces the analytic gradient to
`1.9e-11` and Hessian to `8.2e-06`.

Leakage control is stricter than ordinary outer CV:

1. Hold out one canonical respondent fold.
2. Fit the frozen submitted architecture four times to generate inner-OOF
   predictions for every outer-training respondent.
3. Select one global lambda from
   `{5, 10, 20, 40, 80, 160, 320, Inf}`. Each outer-training respondent is
   scored with version sufficient statistics that exclude that respondent's
   entire 19-task block.
4. Estimate final version deltas using all outer-training respondents'
   inner-OOF predictions.
5. Refit the full frozen pipeline on all outer-training respondents, predict
   the untouched outer fold, and apply the already-estimated deltas.

The frozen pipeline is the actual submitted structure: 85% `ensemble_v11` plus
15% of the five-seed, 8-hidden-unit `nnet` MLP. For additional alignment
protection, the same deltas are also applied to the exact cached fixed-15% OOF
baseline, whose asserted log loss is `1.143686618`.

## Main results

### Requested mean-residual correction

The cheap screen selected alpha 20 and improved `1.161055905` to
`1.159665592`, a gain of `0.001390313`. It covered 218 of 227 held-out
respondents; the other nine received zero correction.

That result did not survive nested CV:

| Outer fold | Selected alpha | Inner gain | Exact-baseline outer gain |
|---:|---:|---:|---:|
| 1 | 100 | +0.000223 | -0.000117 |
| 2 | Inf | 0 | 0 |
| 3 | Inf | 0 | 0 |
| 4 | 100 | +0.000065 | -0.000269 |
| 5 | 80 | +0.000425 | -0.000832 |

| Comparison | Baseline | Corrected | Gain | Respondent-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Freshly refit frozen pipeline | 1.143598 | 1.143839 | -0.000242 | [-0.000893, +0.000409] |
| Exact fixed-15% submitted OOF | 1.143687 | 1.143930 | -0.000243 | [-0.000896, +0.000407] |

All three outer folds where inner CV selected a finite correction worsened. The
other two correctly chose the uncorrected baseline.

### Newton/curvature sensitivity

Nested lambda selections were:

| Outer fold | Selected lambda | Inner gain | Exact-baseline outer gain |
|---:|---:|---:|---:|
| 1 | 160 | +0.000320 | +0.000086 |
| 2 | Inf | 0 | 0 |
| 3 | Inf | 0 | 0 |
| 4 | 160 | +0.000143 | -0.000059 |
| 5 | 160 | +0.000357 | -0.000885 |

`Inf` means the inner procedure correctly selected the uncorrected baseline.
Fold 5 is the decisive failure: an apparently positive inner result reversed
materially on untouched respondents.

| Comparison | Baseline | Corrected | Gain | Respondent-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Freshly refit frozen pipeline | 1.143598 | 1.143767 | -0.000169 | [-0.000797, +0.000460] |
| Exact fixed-15% submitted OOF | 1.143687 | 1.143858 | -0.000172 | [-0.000800, +0.000458] |
| Fresh, exclude one-peer versions | 1.143598 | 1.143663 | -0.000066 | [-0.000675, +0.000540] |
| Exact OOF, exclude one-peer versions | 1.143687 | 1.143755 | -0.000068 | [-0.000679, +0.000538] |

All intervals use 100,000 respondent-clustered bootstrap resamples. Positive
gain means lower log loss; all point estimates are negative.

## Diagnostics

- The Newton correction improves only fold 1. Folds 2-3 are unchanged because
  lambda is infinite; folds 4-5 worsen.
- Removing corrections supported by only one training peer helps folds 1 and 4,
  but fold 5 still worsens by about `0.00092`. The pooled result remains
  negative.
- The correction barely helps tasks whose chosen outcome is an inside
  alternative (`+0.000028`) but harms opt-out choices (`-0.000633`). The
  one-peer exclusion reduces those figures to `+0.000015` and `-0.000259`,
  respectively.
- It worsens all three task-position bands. Harm is largest for Tasks 7-13
  (`-0.000274`; `-0.000162` with one-peer exclusion).
- The most test-like propensity tercile improves (`+0.000165`, or `+0.000455`
  with one-peer exclusion), while the low and middle terciles worsen. This is
  exploratory slice evidence only: it was not a pre-qualified target-only
  validation outcome, the overall clustered interval fails, and no test labels
  exist to confirm transfer.
- Mean absolute decile-level opt-out calibration error is `0.010293` before,
  `0.010301` after, and `0.010317` after the one-peer exclusion: no calibration
  improvement.
- Only `57.4%` of versions with repeated outer-fold estimates have a unanimous
  nonzero raw Newton direction. The median sign-consistency fraction is 1 only
  because five highly-overlapping outer training sets make this a coarse
  diagnostic; the unanimous rate is the more informative statistic.
- Held-out respondents average about 2.8-3.0 usable training peers. Among
  versions with at least two peers, one respondent supplies more than half of
  the summed absolute gradient in roughly 50-55% of cases. With exactly two
  peers, median dominance is `0.707`. This confirms the small-effective-sample
  concern even though each version contributes many task rows.

## Audit and reproducibility

- `R/codex_version_shrinkage_common.R`: saved-version loader, frozen base-model
  fitter, mean-residual and Newton updates, lambda evaluation, bootstrap.
- `R/codex_version_mean_residual_screen.R`: canonical single-split screen.
- `R/codex_version_mean_residual_eval.R`: cached-fit nested mean-residual CV.
- `R/codex_version_mean_residual_audit.R`: independent reconstruction of the
  requested correction.
- `R/codex_version_shrinkage_cv.R`: nested canonical Newton five-fold run.
- `R/codex_version_shrinkage_diagnostics.R`: calibration, outcome/task/
  propensity slices, sign consistency, peer counts, and dominance.
- `R/codex_version_shrinkage_audit.R`: independent reconstruction from saved
  base OOF artifacts.

The audit verifies all 17,252 inner-OOF rows per outer fold are present; every
inner model excludes both its inner target respondents and the outer holdout;
every final outer fit contains 908 source and 227 target respondents with no
overlap; the primary deltas exactly reproduce
`n/(n+alpha) * mean(y4-p4)`; the sensitivity deltas exactly reproduce
`-g/(h+lambda)`; and all corrected matrices reproduce the saved results
bit-for-bit.

Generated results live under
`data_processed/codex_version_shrinkage/` (gitignored), especially
`mean_residual_cv_result.rds`, `mean_residual_outer_results.csv`,
`mean_residual_bootstrap.csv`, `mean_residual_audit.csv`,
`newton_cv_result.rds`, `newton_outer_results.csv`, `newton_bootstrap.csv`,
`newton_loss_slices.csv`, and `newton_audit.csv`.

Because the opt-out correction fails every submission gate, the
version-specific whole-utility scale was not attempted, no cheapest-inside
follow-up was attempted, and no full set of version-specific alternative
intercepts was attempted.
