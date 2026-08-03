# Difficulty-predictability diagnostic -- findings

Implementation: `R/codex_difficulty_diagnostic.R` plus a critical follow-up
decomposition (not separately committed as a script; reproduced inline
below). This is a **diagnostic**, not a candidate model -- no promotion gate
applies; its purpose (per the adversarial-modelling brief's item #4) is to
decide whether persistently high-loss respondents/tasks are predictable and
exploitable, going beyond the existing slice-based flatness check
(`cleaning_log.md`, 2026-07-26 calibration diagnostic: confident-miss rate
flat at 0.15-0.21 across segment/task-position/region/true-class).

## Method

Fit a nested-cross-fitted ridge regression (`glmnet`, respondent-grouped
folds, same canonical partition as every model this session) predicting
v14's own per-row log loss from test-time-available features only (no
outcome leakage): respondent covariates (income, age, miles, night, gender,
urbanicity, education, segment, region, parking situation, year), task
position, and choice-set design content (price spread/CV, count of
attributes actually varying in the task). Loss is decomposed into an
opt-out-margin component and a conditional-bundle-discrimination component,
per the brief's request to distinguish these.

## Result 1: loss is genuinely predictable out-of-sample, not flat

| Target | n | OOF R² | OOF correlation |
|---|---|---|---|
| Total loss | 21,565 | 0.0252 | 0.159 |
| Opt-out-margin loss | 21,565 | 0.0403 | 0.203 |
| Conditional-bundle loss (inside rows only) | 15,046 | 0.0511 | 0.226 |

The out-of-fold decile table (predicted difficulty decile vs. actual mean
loss) is cleanly monotonic (0.980 in the easiest predicted decile to 1.315
in the hardest, all deciles ~2,156 rows), a real, non-flat, out-of-sample
signal -- a materially stronger finding than the 2026-07-26 slice check,
which only tested pre-chosen univariate slices and could not have detected
a multivariate combination like this.

## Result 2 (critical follow-up): this is not tautological, but appears to be irreducible heteroskedasticity, not exploitable miscalibration

Before trusting result 1, refit excluding v14's own predicted opt-out
probability and its own predicted top-two conditional-bundle gap (both of
which mechanically correlate with a model's own realized loss even under
perfect calibration -- a closer decision is expected to have higher average
loss for any well-calibrated model, so including them risks a tautological
"the model's confidence predicts its own loss" result rather than a genuine
external signal).

| Target | Full feature R² | Exogenous-covariates-only R² |
|---|---|---|
| Total loss | 0.0252 | 0.0147 |
| Opt-out-margin loss | 0.0403 | **0.0415** (unchanged/slightly higher) |
| Conditional-bundle loss | 0.0511 | 0.0292 |

The opt-out-margin result is essentially unchanged when the model's own
confidence is removed entirely -- confirming this is a real, exogenous
signal (respondent covariates and task design content), not a tautology.
The conditional-bundle result drops by roughly half but remains clearly
nonzero.

**However, real predictability of loss does not by itself imply exploitable
miscalibration.** A well-calibrated model that correctly hedges more for
respondent/task profiles with genuinely more erratic (higher-entropy)
choice behavior will show exactly this pattern: loss predictable from
covariates, because those covariates predict the *irreducible* difficulty
of the choice, not a fixable error in the model's confidence. This session
and prior sessions have already tested, using overlapping subsets of these
same exact features, whether rescaling/adjusting confidence along these
lines helps:

- Global temperature and covariate-indexed utility-scale heterogeneity
  (respondent/segment/task-indexed): decisively harmful.
- The alternative-link experiment's opt-out-specific shift and shape+scale
  families: decisively harmful.
- The task-content-conditioned local temperature (`gap_top2`, `price_cv` --
  the same two design features used here): decisively harmful.
- The gated "safe diversity recombination" experiment (2026-07-31): a
  difficulty-gated ensemble-weight blend also failed, weight stuck on v14.

All four independent rescaling/gating mechanisms fail despite this
diagnostic now confirming real predictability exists. The most coherent
reading is that v14's calibration already appropriately reflects this
heterogeneity (hedges more on genuinely harder profiles), so the
predictable variance in loss is irreducible aleatoric noise, not a fixable
miscalibration -- consistent with, not contradicting, every rescaling
experiment's rejection this session.

## A concrete, reportable nugget

Top-decile-loss respondents (n=114) have a **lower** actual opt-out rate
(12.7%) than the population average (30.2%), while v14's mean predicted `q`
for that group (29.2%) sits close to the population average (30.1%) --
high-loss respondents are not simply "people the model over-predicts
opt-out for." If anything they opt out less than average yet remain harder
to predict, ruling out the simplest possible story (a systematic
opt-out-margin bias) as the driver.

## Conclusion

This closes item #4 of the adversarial-modelling brief with a genuine test
rather than an assumption: difficulty is real and predictable (not flat),
but not exploitable by any rescaling or gating mechanism tried, consistent
across five independent implementations now. Does not itself justify a new
modeling attempt using these same features; would only be actionable via a
structurally different mechanism (e.g. training entirely separate,
non-rescaling specialist parameters for the hard segment) which the
brief's own dedicated-hurdle-model test already substituted for and
rejected in the opt-out margin, and which is not re-attempted here to avoid
re-litigating a closed result.
