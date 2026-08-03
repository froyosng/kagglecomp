# Pre-registration: Bayesian hierarchical mixed logit (bayesm), 2026-07-29

## 0. Restating the context (to confirm the logs were actually read)

**Current best model.** `ensemble_v11 + MLP`: `0.85*(0.80*mlogit_m8trpg + 0.20*original_xgb)
+ 0.15*shallow_mlp`, where `mlogit_m8trpg` is a fixed-effects conditional logit
(19 attributes as factors, Price recoded as a 12-level factor, alt2/alt3 position
dummies, `is_cheapest`/`is_dearest`/`price_gap_min`/`price_gap_max` choice-set-context
terms, task-fatigue terms, and Price/inside-good interacted with segment, region,
parking situation, and several respondent covariates), `original_xgb` is a
`multi:softprob` xgboost on the wide-format data, and `shallow_mlp` is a single-
hidden-layer `nnet` (8 units, decay 0.1, 5 seeds) blended in at a fold-cross-fitted
~15% weight. This scores **1.143789** on the canonical 5-fold respondent-grouped CV
(seed 4821, pooled OOF) and **1.201** on the public leaderboard. I independently
reproduced this number from the raw cached OOF artifacts before writing this
document (see Section 5 below) -- confirmed exact match to 8 decimal places
(`crossfit_blend_logloss = 1.143789`, `crossfit_blend_gain = 0.001304771` from
`data_processed/codex_behavioral_round/mlp_oof.rds`, reconciled against
`data_processed/oof_ensemble_v10.rds`'s `oof_mlogit`/`oof_xgb`).

**The rejected frequentist mixed logit(s).** Two specifications were tried and
rejected, both fit via `mlogit`'s panel/simulated-maximum-likelihood machinery
(`mlogit(..., panel = TRUE)`, Halton draws, `rpar` random-parameter declarations):

1. `mod4` (full mixed logit): independent normal random parameters on **all 20**
   continuous attributes (R=100 Halton draws). Training log-likelihood improved
   hugely (-16,449 vs. mod1's -20,432) but *validation* log loss got **worse**
   (1.247 vs. mod1's 1.236) -- classic overfitting: 20 free parameters per
   respondent estimated from only 19 choice observations/respondent memorizes
   training-respondent idiosyncrasies that provably cannot transfer, since test
   respondents (Case 1136-1398) are entirely disjoint from train (Case 1-1135).
2. `mod5`/`mlogit_v10_mod8_random_price` (price-only random, later re-tried on top
   of the full `mod8`/`m8trpg` fixed spec): a single random-normal Price parameter.
   Essentially matched the fixed-effects model (1.235 vs. mod1's 1.236; later,
   +0.0003 on top of the fully-featured m8trpg) -- once observed heterogeneity
   (covariate/segment interactions) is already in the fixed part, there is very
   little *unobserved* price heterogeneity left for a random parameter to capture.

Both were rejected for the same underlying structural reason documented
repeatedly in `AGENTS.md`: **test respondents are a completely disjoint set of
263 new people (Case 1136-1398) who were never observed during fitting**, so a
respondent-*specific* random-effect estimate (the `rpar`/individual conditional
posterior mean `mlogit` reports) has no way to attach to a new person at
prediction time. The paper trail is explicit that this is a structural fact about
the data, not a fixable estimation detail: "we can never personalize to a
specific test respondent, so respondent-specific random effects (mixed logit)
cannot transfer."

## 1. Why this experiment is different (estimation philosophy, not model structure)

The brief for this track is explicit that the frequentist-vs-Bayesian distinction,
not a new model structure, is the thing actually being tested here, and that this
is the weakest-motivated of four parallel hypotheses precisely because it is
conceptually close to an already-rejected idea. Restating that distinction
concretely, since "different estimation philosophy" is otherwise just a slogan:

- **What does NOT change:** the fundamental model is still a random-coefficients
  (mixed) conditional/multinomial logit -- a population distribution of
  respondent-level taste parameters, exactly like `mod4`/`mod5`.
- **What changes:** frequentist `mlogit` maximizes a *simulated* likelihood
  (Monte-Carlo-integrated over the random-parameter distribution) via BHHH, and
  reports each respondent's *individual conditional posterior mean* as a point
  estimate with no explicit prior -- this individual estimate is exactly the
  quantity that cannot exist for a new respondent. **This experiment fits the
  same class of model with a genuine, proper Bayesian hierarchical prior**
  (Normal population mean + Inverse-Wishart population covariance, explicit,
  estimated via Gibbs sampling / MCMC, not simulated ML) and -- critically --
  **the held-out prediction mechanism never touches any respondent's individual
  posterior.** For a new respondent, a fresh random-coefficient vector is drawn
  from the *population*-level posterior (integrating over both parameter
  uncertainty and between-respondent heterogeneity) at every retained MCMC draw,
  and predictions are averaged over that population-marginal predictive
  distribution. This is the direct Bayesian analogue of `brms`'s
  `allow_new_levels = TRUE, sample_new_levels = "gaussian"`, or Stan's standard
  "draw a new group-level effect from the fitted hyper-parameters" recipe for
  out-of-sample groups.
- **Honest expectation:** if the earlier failure was really about *new
  respondents structurally having no random effect to condition on* (as the
  project's own logs conclude), switching estimators cannot fix that -- a
  population-marginal prediction for a mixed logit is mathematically close to
  (though not identical to) the corresponding fixed-effects model's prediction,
  just with the mean/variance integrated properly instead of point-estimated.
  If the earlier failure was instead partly an artifact of frequentist SML's
  known small-sample instability (biased simulated-likelihood surfaces, no
  shrinkage/regularization at all on the individual estimates), proper
  hierarchical shrinkage could plausibly do a bit better, especially through
  more disciplined partial pooling of the population covariance. A null result
  that simply reproduces the earlier failure to transfer is treated as a
  real, informative, and likely possible outcome of this pre-registered test,
  not as a failure of execution.

## 2. Toolchain finding (checked before assuming a workaround was needed)

`rstan`, `brms`, `cmdstanr`, `rstanarm`, and `StanHeaders` are **not installed**
in this R 4.6.0 environment. Checked whether they *could* be installed and used:
- No `gcc`/`g++`/`make` on `PATH`; no `Rtools` installation found.
- `pkgbuild::has_build_tools(debug = TRUE)` returns `FALSE` and explicitly fails
  a trial C compile (`'make' not found`).
- This blocks not just source installs but, more importantly, **actually fitting
  any new Stan model**: `rstan`/`brms`/`cmdstanr` all require compiling
  model-specific C++ at the time a model is fit (`stan()`/`brm()`/
  `cmdstanr::cmdstan_model()`), which is a hard requirement independent of
  whether the R package itself installs as a CRAN binary.
- Per the task's own explicit instruction ("If Stan/brms prove impractical to
  install or run in a reasonable time in this environment, report that
  honestly rather than forcing a workaround that compromises the
  population-level-prediction requirement"), Stan/brms/cmdstanr are ruled
  out here, honestly, rather than attempting a multi-hour Rtools install
  detour with an uncertain payoff.

**Substitute:** `bayesm::rhierMnlRwMixture` -- a mature, peer-reviewed (Rossi,
Allenby & McCulloch, *Bayesian Statistics and Marketing*, ch. 5), CRAN-binary-
installable (confirmed: installs from a precompiled Windows binary, no
compilation needed) hierarchical Bayes mixed-logit sampler. It is a genuine
MCMC estimator (Gibbs sampling for the population Normal-InverseWishart
hyperparameters, conditional on a random-walk Metropolis step per respondent for
the individual coefficients) with explicit, user-set priors -- a different
estimation philosophy from `mlogit`'s simulated MLE in exactly the sense this
track is meant to test, and it is the standard tool this exact class of problem
(conjoint/choice-based hierarchical Bayes with tens of respondents' worth of
partworths) was built for. A feasibility timing check (60 respondents, R=500
draws, `nvar=23`) completed in 1.19s (0.0024s/draw) -- confirms the full-scale
run (~908 respondents/fold) is comfortably tractable in minutes, not hours.

## 3. Model specification (fixed before any CV result is seen)

**Design matrix** (`nvar = 23` columns), one row per respondent-task-alternative:
- 19 standardized attribute levels: `z(CC), z(GN), ..., z(HU)` (raw integer level
  codes, standardized using that fold's TRAINING rows' mean/sd, computed across
  all 4 alternatives' rows, including alt 4's structural zeros).
- 1 standardized Price: `z(Price)`, same standardization convention.
- 3 alternative-specific constants: `ASC2 = 1{alt=2}`, `ASC3 = 1{alt=3}`,
  `ASC4 = 1{alt=4}` (alt 1 is the reference; continuous attribute coding, so
  -- unlike the factor-coded `m8trpg` family -- there is no ASC identification
  conflict here, exactly as already documented for `mod1` in `cleaning_log.md`).
  **`ASC4` is this experiment's operationalization of the task's requested
  "inside-good" random term** (a per-respondent random opt-out utility is the
  mirror image of a per-respondent random inside-good propensity).

**Random coefficients:** `beta_i ~ N(mu, Sigma)`, a single population component
(`ncomp = 1`, i.e. literally the "population Normal prior ... with an estimated
population standard deviation" language from the brief, not a latent-class
mixture) for **all 23** coefficients, estimated hierarchically (partial pooling)
via `bayesm::rhierMnlRwMixture`. `Z` (respondent covariates predicting the mean)
is omitted, i.e. left at bayesm's default intercept-only column -- deliberately:
the observed-heterogeneity covariate/segment interactions are already the
confirmed, adopted mechanism in `m8trpg`; this experiment isolates the *other*
half of mixed-logit theory (unobserved heterogeneity via a population
covariance), which is what mod4/mod5 were about and what a Bayesian re-estimate
is meant to test.

**Deviation from the letter of the brief, disclosed up front:** the brief
suggests random slopes specifically "on price (and optionally other key terms
... such as the inside-good indicator)," i.e. a mix of fixed population-level
terms and a couple of random ones. `bayesm::rhierMnlRwMixture` has no supported
mechanism to declare *some* `X` columns fixed (shared exactly across
respondents) and others hierarchically random within one joint sampler --
every column of `X` gets a person-level coefficient by construction. Rather than
hand-roll a custom, unverified, high-dimensional Metropolis-within-Gibbs sampler
under this track's time budget (a materially higher implementation-risk path for
a hypothesis already flagged as the weakest-motivated of the four), this
experiment makes **all 23** terms random -- a strict superset of "Price and
inside-good random." This is the direct Bayesian analogue of `mod4` (full random
attributes) rather than `mod5`/`mod10` (Price-only random), which is arguably a
*harder*, not easier, bar to clear, since `mod4` was the worse of the two
frequentist specifications. If the data do not support heterogeneity beyond
Price/inside, Bayesian partial pooling should show it directly: the posterior
for the other 20 dimensions' population variance should shrink toward small
values (this is checked and reported, not assumed). Price's and `ASC4`'s
population mean/SD are reported as the headline diagnostics matching the
brief's specific interest, in addition to the overall predictive comparison.

**Priors** (bayesm argument names):
- `ncomp = 1`
- `mubar = rep(0, 23)`, `Amu = 0.5` (population-mean prior precision;
  deliberately set higher than bayesm's own default of 0.01, which the
  package's documentation itself flags as "too small for many applications" --
  `Amu = 0.5` implies a prior SD of `sqrt(1/0.5) ≈ 1.41` on each
  standardized-scale population-mean coefficient, weakly informative on the
  standardized scale actually used here)
- `nu = nvar + 3 = 26`, `V = nu * diag(23)` (bayesm's own default, minimally
  informative proper Inverse-Wishart prior on the population covariance
  `Sigma`)
- `Ad = 0.01 * diag(23)`, `deltabar = rep(0, 23)`, `a = 5` (bayesm defaults;
  `Z` is intercept-only so `Ad`/`deltabar` only touch the single population-mean
  row of `Delta`, redundant with `mubar`/`Amu` above)
- `w = 0.1` (fractional-likelihood weight for the individual RW-Metropolis
  proposal, bayesm default), `s = 2.93/sqrt(23)` (bayesm default RW-MH scale)

**MCMC:** 2 independent chains per fold (seeds `4821` and `9001` plus a
per-fold offset), `R = 8000` raw draws/chain, `keep = 8` (1000 retained
draws/chain). Burn-in: discard the first 250 retained draws/chain (=2000 raw
draws). Pool the remaining 750 retained draws x 2 chains = **1500 post-burn-in
draws** per fold for all posterior summaries and posterior-predictive
integration. Convergence check: compare each chain's post-burn-in posterior
mean of `mu_Price`, `mu_ASC4`, and `diag(Sigma)_Price`, `diag(Sigma)_ASC4`
between the 2 chains per fold (a between-chain vs. within-chain-SD comparison,
not a formal multi-chain R-hat, which bayesm's output does not directly
support) -- reported as a limitation, not hidden.

## 4. CV design

Canonical 5-fold respondent-grouped CV: `fold_of_case` from
`data_processed/oof_ensemble_v10.rds` (seed 4821, verified 227/227/227/227/227
split of the 1,135 training respondents -- reproduced and checked in Section 5).
For each fold `k`: refit **both chains from scratch** using only that fold's
~908 training respondents' 19-task panels (held-out respondents' rows never
enter `lgtdata` for that fold in any way).

**Held-out (population-level posterior-predictive) prediction --the specific
place this class of model is most likely to leak or misrepresent genuine
out-of-sample prediction:** for every held-out respondent in fold `k` (who by
construction was never included in that fold's `lgtdata`, hence has no
`betadraw` entry at all), and for every one of the 1500 pooled post-burn-in
draws `r`: draw **one fresh** `beta_new ~ N(mu_r, Sigma_r)` via
`bayesm::rmixture(1, 1, list(list(mu_r, rooti_r)))$x` (the package's own
mixture-of-normals sampler, applied with `ncomp=1`), independently for every
(held-out respondent, draw) pair -- never reusing a draw across respondents,
never reading `out$betadraw` (the *training*-respondent individual draws) for
any held-out prediction. Compute that respondent's 19-task choice
probabilities under `beta_new` via the standard conditional-logit softmax, and
average over the 1500 draws. This average is the final OOF prediction for
that respondent. Pool all 5 folds into one 21,565 x 4 OOF matrix (matching the
project's own canonical CV reporting convention).

**Smoke test, run before the real 5-fold loop (this is an implementation
verification step, not a place to peek at CV results):** on a small subset
(~60 respondents split into a ~45/15 mini-fold), verify (a) the pipeline runs
end-to-end and produces valid row-stochastic probability matrices; (b) that
the held-out prediction code path only ever reads `nmix$compdraw`
(population-level draws) and asserts (via `stopifnot`) that it never
references `out$betadraw`; (c) empirically, that re-running the held-out
prediction for the *same* respondent with a different RNG seed changes the
predicted probabilities by a Monte-Carlo-noise amount (genuine fresh-draw
variability), whereas substituting -- as an explicit, deliberately-WRONG
comparison condition -- the nearest *training* respondent's fitted posterior
mean `beta_i` in place of a fresh population draw produces predictions that
are (i) invariant to reruns/seeds and (ii) visibly different from the
population-marginal predictions, concretely demonstrating that the two
mechanisms differ and confirming which one the real pipeline uses.

## 5. Baseline reconstruction (raw-artifact reproducibility check)

Copied (read-only) from the primary worktree, verified against the numbers
already logged in `AGENTS.md`/`submissions_log.csv` before any new modeling:
- `data_processed/oof_ensemble_v10.rds`: `oof_mlogit` (m8trpg OOF) reproduces
  **1.147021** exactly; `oof_xgb` (original_xgb OOF) reproduces **1.178668**;
  `0.8*oof_mlogit + 0.2*oof_xgb` reproduces **1.145094** exactly (`ensemble_v11`);
  `fold_of_case` is the canonical seed-4821 assignment.
- `data_processed/codex_behavioral_round/mlp_oof.rds`: the shallow-MLP
  (`h08_d0.100`) canonical-fold OOF and its own logged fold-cross-fitted blend
  weights against `ensemble_v11`; recombining exactly reproduces
  `crossfit_blend_logloss = 1.143789` (`ensemble_v11 + MLP`, the current best).

`current_best_OOF` for this experiment's comparison is reconstructed directly
from these two files by re-running the identical fold-cross-fitted weight-search
logic already used to produce 1.143789 (grid search `mlp_weight` in
`seq(0, 0.30, 0.01)` per fold using only the other 4 folds, exactly mirroring
`R/codex_mlp_ensemble.R`'s own `stage == "cv"` block) -- not re-derived from a
different or approximate recipe.

## 6. Candidate construction and promotion rule (fixed before results are seen)

`my_bayes_OOF` = the pooled 5-fold OOF matrix from Section 4.

`candidate_OOF`: for each fold `k`, grid-search a single blend weight
`w in seq(0, 0.30, 0.01)` minimizing pooled log loss of
`(1-w)*current_best_OOF + w*my_bayes_OOF` **using only the other 4 folds'
rows**, then apply that fold's selected `w` to fold `k`'s held-out rows
(fold-cross-fitted, no leakage -- matching every other candidate-blend
evaluation already used throughout this project).

**Respondent-level gain:** for each of the 1135 training respondents,
`gain_i = mean over that respondent's 19 rows of (row_loss(current_best_OOF) -
row_loss(candidate_OOF))`.

**Respondent-clustered paired bootstrap:** resample the 1135 respondents'
`gain_i` with replacement, **100,000 replicates**, ordinary 95% percentile CI
of the mean gain (seed `4821`).

**Promotion bar (fixed, non-negotiable per the task brief):** the model is only
promoted/recommended if the ordinary 95% bootstrap CI's lower bound is `> 0`.
A positive point estimate alone is not sufficient. Given this is explicitly the
weakest-motivated of four parallel hypotheses, and directly re-tests a
structural failure mode (new respondents have no individual random effect to
condition on) that switching estimators cannot mechanically repair, a result
that fails this bar and simply reproduces the earlier mixed-logit's null/negative
transfer is treated as a legitimate, informative, pre-registered negative
result -- not as a sign the experiment was executed incorrectly.

## 7. What will be reported regardless of outcome

- The reconstructed `current_best_OOF` and its exact match (or lack thereof) to
  1.143789.
- `my_bayes_OOF`'s standalone canonical CV log loss.
- The fold-cross-fitted candidate blend's CV log loss and gain vs. current best.
- The full bootstrap summary (point estimate, mean, SD, 95% CI, win rate).
- Population-level posterior summaries for `mu_Price`, `mu_ASC4`, and
  `diag(Sigma)` for every one of the 23 terms (to honestly show whether
  heterogeneity concentrates on Price/inside-good as hypothesized, or is
  diffuse/negligible everywhere, or something else).
- The between-chain convergence comparison.
- The smoke-test verification output demonstrating the population-level vs.
  individual-posterior prediction distinction concretely.
