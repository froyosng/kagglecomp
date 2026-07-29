# Findings: Bayesian hierarchical mixed logit (bayesm), 2026-07-29

Full pre-registration: `codex_bayes_mixed_preregister.md` (committed `d6caf4f`,
2026-07-29 12:22:30 +0800, verified via `git log` to predate every result
below). Raw artifacts backing every number in this document are in
`data_processed/codex_bayes_mixed/` (gitignored, kept locally; regenerate via
`R/codex_bayes_mixed_smoketest.R` then `R/codex_bayes_mixed_cv.R`).

## Verdict, up front

**Not promoted. The pre-registered promotion bar is not met, and the sign is
the wrong way: blending in this Bayesian mixed logit makes the current best
model very slightly WORSE, not better.**

- Standalone Bayesian mixed logit, canonical 5-fold CV: **1.195160** (worse
  than every conditional-logit specification in the project since `mod2b`,
  1.2186 -- this simple 23-term random-coefficients model sits closer to the
  continuous `mod1` (1.236) than to `m8trpg` (1.147021), as expected for a
  model with no observed-heterogeneity interaction terms at all).
- Current-best baseline (`ensemble_v11 + MLP`), reconstructed from raw cached
  OOF and independently re-verified here: **1.143789** (exact match to the
  documented number).
- Fold-cross-fitted candidate blend (current best + Bayesian mixed logit,
  weight chosen per fold from the other 4 folds only): **1.144403** --
  **worse** than the baseline by 0.000614.
- Respondent-clustered paired bootstrap (100,000 replicates, seed 4821):
  point gain **-0.0006138**, 95% CI **[-0.0012461, +0.0000324]**, win rate
  **3.16%**.
- **Promotion rule (95% CI lower bound > 0): FAILS.** Not only does the CI
  fail to exclude zero on the positive side, it is almost entirely on the
  *negative* side -- only 32 one-thousandths of a percentage point of the
  interval's upper edge pokes above zero, and only 3.16% of bootstrap
  replicates favor the candidate at all. This is a clean, unambiguous null/
  slightly-harmful result, not a close call.

This is, honestly, close to the outcome flagged as most likely going in: this
was pre-registered as the weakest-motivated of four parallel hypotheses,
conceptually adjacent to the already-rejected frequentist mixed logit, and it
reproduces that rejection under a genuinely different (Bayesian MCMC,
explicit hierarchical priors) estimation philosophy rather than escaping it.

## 1. What was actually tested (recap)

A hierarchical Bayesian mixed multinomial logit, `beta_i ~ N(mu, Sigma)`
(`ncomp = 1`, a single population Normal, not a mixture), fit via
`bayesm::rhierMnlRwMixture` (Gibbs sampling for `(mu, Sigma)` + per-respondent
random-walk Metropolis for `beta_i`) -- a genuine MCMC estimator with explicit
proper priors, in contrast to the already-rejected `mlogit`-family models'
simulated maximum likelihood. 23 random coefficients per respondent: 19
standardized attribute levels + standardized Price + `ASC2`/`ASC3`/`ASC4`
(continuous coding, so no ASC-identification conflict; `ASC4` is this
experiment's stand-in for the brief's "inside-good" random term). `rstan`/
`brms`/`cmdstanr` were confirmed impractical in this environment (no Rtools/
C++ toolchain -- `pkgbuild::has_build_tools()` returns `FALSE`) and `bayesm`
was substituted, as pre-registered, with the deviation disclosed there: bayesm
cannot mix fixed and random coefficients in one sampler, so all 23 terms are
random rather than just Price/inside-good, a strict superset of what was
asked for computational-tractability reasons.

## 2. Population-level posterior-predictive mechanism: verification recap

This is the specific correctness risk this track was flagged as most likely
to get wrong, so it was smoke-tested (`R/codex_bayes_mixed_smoketest.R`) on a
60-respondent subset (45 fit / 15 held out) *before* the real run, per the
pre-registration. All four checks passed (`data_processed/codex_bayes_mixed/
smoketest_summary.csv`):

| Check | Result |
|---|---|
| Predictions are valid row-stochastic probability matrices | PASS (all rows sum to 1 to 1e-9; all entries > 0) |
| `population_predict()`'s body never references `betadraw` | PASS (confirmed via `grepl()` on `deparse(body(...))`) |
| Re-running the held-out prediction with a different RNG seed changes the result by a genuine Monte-Carlo amount | PASS: mean abs diff 0.0198, correlation between the two seeds' predictions 0.9836 (different but from the same population distribution, as expected for fresh draws integrating over `N(mu_r, Sigma_r)`) |
| Deliberately WRONG comparison (refit with the "held-out" respondents included, use their own fitted individual posterior-mean `beta_i`) is deterministic across reruns, and visibly different from the population-marginal prediction | PASS: individual-posterior prediction is bit-for-bit identical across reruns (max diff 0.0), and differs from the population-marginal prediction by a mean absolute 0.2658 (max 0.887) across the 4 choice probabilities -- a large, unambiguous difference confirming the two mechanisms are not accidentally the same |

The real 5-fold run's held-out prediction code path (`population_predict()` in
`R/codex_bayes_mixed_common.R`) is exactly the function verified here: for
every one of a fold's held-out respondents (who never appear in that fold's
`lgtdata`, and therefore have no `betadraw` at all) and every one of the 1,500
pooled post-burn-in draws, a **fresh** 23-dimensional coefficient vector is
drawn from `N(mu_r, Sigma_r)` via `bayesm::rmixture()`, independently per
(respondent, draw); the resulting 19-task choice probabilities are averaged
over all 1,500 draws. No respondent-specific shrinkage estimate that only
exists for training respondents is ever used for a held-out prediction.

## 3. MCMC fitting: per-fold timing and convergence

Two independent chains per fold (seeds `4821+100k` and `9001+100k`), `R=8000`
raw draws each, `keep=8` (1000 retained draws/chain), first 250 retained draws
(=2000 raw) discarded as burn-in, 750 draws/chain pooled = 1500 draws/fold
used for everything downstream.

| Fold | Fit time (both chains, s) | Predict time (227 held-out, s) |
|---|---|---|
| 1 | 259.0 | 10.2 |
| 2 | 256.3 | 10.0 |
| 3 | 295.3 | 10.1 |
| 4 | 253.3 | 10.0 |
| 5 | 316.0 | 14.7 |

Total wall clock: ~24 minutes for the full 5-fold, 2-chain, 8000-draw run --
comfortably inside the feasibility estimate from the pre-registration's timing
check.

**Between-chain convergence** (informal check, since bayesm's Gibbs/MH output
doesn't directly support a formal multi-chain R-hat): comparing each fold's
two independently-seeded chains' post-burn-in posterior means for the
headline terms (`data_processed/codex_bayes_mixed/convergence_summary.csv`):

| Fold | mu_Price (chain1 / chain2) | mu_ASC4 (chain1 / chain2) | sigma_Price (var, chain1 / chain2) | sigma_ASC4 (var, chain1 / chain2) |
|---|---|---|---|---|
| 1 | -3.439 / -3.461 | -8.644 / -8.771 | 9.657 / 9.777 | 28.724 / 28.212 |
| 2 | -3.366 / -3.363 | -8.284 / -8.257 | 10.047 / 10.090 | 27.399 / 26.964 |
| 3 | -3.355 / -3.385 | -8.399 / -8.481 | 9.498 / 9.746 | 24.778 / 25.514 |
| 4 | -3.342 / -3.343 | -8.048 / -8.015 | 9.820 / 9.783 | 29.095 / 28.361 |
| 5 | -3.475 / -3.408 | -8.795 / -8.630 | 10.879 / 10.066 | 25.519 / 24.504 |

Between-chain absolute differences are small relative to the parameter
magnitudes throughout: `mu_Price` differs by 0.001-0.067 (on a mean of
~-3.3 to -3.5, i.e. <2% relative), `mu_ASC4` by 0.03-0.17 (on ~-8 to -8.8,
<2%), and the two variance terms (noisier, as expected for a covariance
component estimated from 750 draws/chain) by up to ~8% relative for
`sigma_Price` and ~4% for `sigma_ASC4`. No chain pair disagrees in sign or by
a large multiple -- reasonable evidence the sampler mixed adequately at this
draw count, though this is an informal check, not a formal convergence
guarantee.

**Population-level interpretation (the headline diagnostic the brief asked
for):** `mu_Price` is consistently negative (~-3.3 to -3.5, as expected --
higher price reduces utility) and `mu_ASC4` is consistently strongly negative
(~-8.0 to -8.8), consistent with the opt-out's known lower average appeal.
Both variance terms are large relative to their means (`sigma_Price`
implies a between-respondent SD of ~3.0-3.3 on the standardized-Price scale;
`sigma_ASC4` implies an SD of ~5.0-5.4) -- i.e. the model does find
substantial estimated between-respondent heterogeneity in both price
sensitivity and opt-out propensity, consistent with this project's earlier,
independently-confirmed finding (`latent_class_price_scale`,
2026-07-27) of a real, stable price-sensitivity split (~0.5x to ~1.85-2.0x
normal sensitivity). Finding the heterogeneity is not the same as the
heterogeneity being *usable* for new respondents, though -- see Section 5.

## 4. Baseline reconstruction (independent re-verification)

Per the pre-registration, the current-best baseline was reconstructed from
raw cached artifacts copied (read-only) from the primary worktree, not
re-derived by a different recipe:

- `data_processed/oof_ensemble_v10.rds$oof_mlogit` (m8trpg OOF): log loss
  **1.147021** -- matches the documented `mlogit_m8trpg_standalone` CV number
  exactly.
- `data_processed/oof_ensemble_v10.rds$oof_xgb` (original_xgb OOF): log loss
  **1.178668** -- matches the documented `xgboost_multiclass_v1`/component
  number.
- `0.8*oof_mlogit + 0.2*oof_xgb`: log loss **1.145094** -- matches
  `ensemble_v11` exactly.
- Re-running the exact fold-cross-fitted MLP-weight search
  (`data_processed/codex_behavioral_round/mlp_oof.rds`, config `h08_d0.100`)
  reproduces **1.143789** -- matches `ensemble_v11 + MLP`, the current best,
  exactly (per-fold MLP weights recovered: 0.14, 0.17, 0.15, 0.15, 0.13,
  `data_processed/codex_bayes_mixed/baseline_per_fold_mlp_weight.csv`).

All three `stopifnot()` checks in `reconstruct_current_best()` passed
(script exit code 0), so this experiment's comparison target is confirmed to
be the actual, currently-deployed best model, not an approximation of it.

## 5. Candidate construction and the promotion test

`candidate_OOF` = fold-cross-fitted blend of `current_best_OOF` and the
Bayesian mixed logit's OOF (grid search blend weight `w in seq(0,0.30,0.01)`
per fold, using only the other 4 folds; `data_processed/codex_bayes_mixed/
candidate_blend_weights.csv`):

| Fold | Bayesian-component blend weight |
|---|---|
| 1 | 0.00 |
| 2 | 0.07 |
| 3 | 0.05 |
| 4 | 0.10 |
| 5 | 0.06 |

Fold 1's cross-fitted search picked **zero weight** for the Bayesian
component outright (the other 4 folds' training data found no blend weight
that beat pure `current_best`); the other folds picked small (5-10%)
weights. This is already a soft signal that the component is not reliably
additive value.

**Respondent-clustered paired bootstrap** (100,000 replicates, seed 4821,
`data_processed/codex_bayes_mixed/bootstrap_summary.csv`):

| Statistic | Value |
|---|---|
| Point gain (current best - candidate, positive = candidate better) | -0.0006138 |
| Bootstrap mean | -0.0006159 |
| Bootstrap SD | 0.0003260 |
| 95% CI | **[-0.0012461, +0.0000324]** |
| Win rate (candidate better) | **3.16%** |

**Promotion rule result: FAILS.** The task's bar requires the ordinary 95%
CI's lower bound to exclude zero on the *positive* side. Here the interval
doesn't even contain a meaningfully positive region -- it sits almost
entirely below zero, and its upper edge (+0.0000324) is two orders of
magnitude smaller than the lower edge's distance from zero. Blending this
Bayesian mixed logit into the current best is, if anything, a small,
fairly confidently-estimated **harm**, not an unconfirmed gain.

## 6. Interpretation: why the Bayesian re-estimation didn't rescue the idea

The project's own accumulated evidence (cited in the pre-registration,
Section 0) already identifies *why* mixed logit fails to transfer here: test
(and, in this CV design, held-out) respondents are people the model has never
seen, so a respondent-*specific* random-effect estimate cannot exist for them
at prediction time -- full stop, regardless of estimation method. Switching
from frequentist simulated ML to a genuine Bayesian hierarchical model with
proper priors changes *how the population distribution is estimated and how
carefully individual estimates are shrunk*, but a properly-implemented
population-marginal prediction for a new respondent is, structurally, not
that different from what a comparably-specified fixed-effects model already
predicts (the population mean, plus some extra predictive variance from
integrating over `Sigma` and posterior uncertainty in `mu`). The extra
between-respondent variance this model estimates (Section 3) is real and
consistent with prior findings in this project, but a new/held-out
respondent's prediction only benefits from *knowing that variance exists* to
the extent it changes the (properly-integrated) predictive mean and
calibration -- and this project's own prior diagnostic work already showed
the existing fixed-effects ensemble's calibration is essentially optimal
(`cleaning_log.md`, calibration/post-hoc-shrinkage checks). There was limited
room for a population-marginal correction to help, and the standalone
model's much simpler fixed part (no segment/covariate/task/region
interactions at all, unlike `m8trpg`) means most of its predictive
information is coarser than the incumbent to begin with -- consistent with
its weak 1.195160 standalone score and the small, mostly-negative blend
weights it received.

## 7. Honest limitations and disclosed deviations

- **Estimation-tool substitution (disclosed and justified in the
  pre-registration, Section 2):** `rstan`/`brms`/`cmdstanr` were confirmed
  impractical (no C++ toolchain in this environment) and `bayesm::
  rhierMnlRwMixture` was used instead. This is a genuine MCMC hierarchical
  Bayes estimator with explicit priors -- a real change in estimation
  philosophy, matching the spirit of the assignment -- but it is a different
  software implementation from what was originally suggested, and the two
  are not guaranteed to behave identically for every possible model spec.
- **Fixed/random split not implemented (disclosed and justified in the
  pre-registration, Section 3):** the brief asked for random slopes
  specifically on Price and inside-good, with everything else fixed at the
  population level (i.e. matching `m8trpg`'s rich fixed-effects backbone).
  `bayesm`'s sampler makes every `X` column random by construction, so this
  experiment used a simpler, fully-random 23-term specification (19
  continuous attributes + Price + 3 ASCs) instead of `m8trpg`'s ~100-term
  fixed backbone plus 2 random slopes on top. This means the standalone
  1.195160 number is not a fair apples-to-apples comparison to `m8trpg`'s
  1.147021 on model richness alone -- some of the gap is "simpler fixed
  part," not just "Bayesian vs. frequentist." The blend-and-bootstrap
  comparison against the *current best ensemble* (Section 5) is the
  decision-relevant test and does not depend on this caveat, since it
  measures whether adding this component as-is helps the actual deployed
  system, which is what the promotion rule cares about.
- **Convergence diagnostics are informal.** Two chains' posterior-mean
  comparison (Section 3) is a reasonable but not rigorous convergence check;
  no formal split-R-hat or effective-sample-size calculation was computed
  (bayesm's raw output doesn't directly support this without extra
  post-processing that wasn't pursued given the clearly negative headline
  result).
- **A genuinely mixed fixed/random specification on the full `m8trpg`
  backbone was not attempted** (would require either a custom sampler or a
  workaround to force near-zero variance on specific dimensions in
  `bayesm`'s joint covariance -- assessed as high implementation risk for a
  hypothesis already flagged as the weakest-motivated of four, and not
  attempted given the clearly negative result from the simpler, safer
  specification actually run). If a future session wants to push this
  further, that is the one materially different variant left untested.

## 8. Bottom line

This experiment did what it was designed to do: it re-tested the mixed-logit
idea under a genuinely different, proper Bayesian estimation philosophy
(MCMC with explicit hierarchical priors, verified population-level posterior-
predictive scoring for new respondents, confirmed not to leak any
individual-respondent shrinkage into held-out predictions) rather than
frequentist simulated ML, and it reproduces the earlier finding: mixed-logit
random effects do not help prediction for genuinely new respondents on this
dataset, regardless of estimation method. The pre-registered promotion bar
(95% respondent-bootstrap CI excludes zero) is not met -- the CI in fact sits
almost entirely on the harmful side. **Recommendation: do not promote; the
current best (`ensemble_v11 + MLP`, CV 1.143789 / public 1.201) is
unchanged.** This is consistent with, and adds one more independently-
verified data point to, this project's broader conclusion that the dataset's
test respondents being entirely disjoint from train is a structural fact
that no amount of respondent-level personalization -- frequentist or
Bayesian -- can work around.
