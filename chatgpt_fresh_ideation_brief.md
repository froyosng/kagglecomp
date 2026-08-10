# Brief for a fresh literature/ideation pass -- paste this whole file into ChatGPT

You are being asked for a **second opinion** on a Kaggle discrete-choice
competition that has already had a very deep, methodical modeling search.
Your job is NOT to re-suggest anything on the "already tried" list below --
that list is exhaustive and independently verified. Your job is to find
either (a) a genuinely different mechanism this list does not cover, grounded
in real, citable literature, or (b) a specific paper/method applied to
repeated/partial-profile conjoint choice data with an outside option that
this project's own search vocabulary would not have surfaced. If you cannot
find something genuinely new, say so plainly -- do not pad the answer with
already-closed ideas restated in new words.

## The competition and data

- Predict which of 4 car safety-feature bundles a respondent chooses in a
  repeated choice-based conjoint (CBC) survey. Alternative 4 is always a
  fixed "opt-out"/no-purchase option (all attributes and price = 0).
  Scored by multiclass log loss.
- `train.csv`: 21,565 rows, 1,135 respondents (`Case`), each completing
  exactly 19 choice tasks (`Task`). 19 categorical attribute codes (varying
  level counts, 2-7 levels each) + `Price` (integer levels 1-12 for the 3
  inside alternatives, 0 for the opt-out), each suffixed 1-4 for the 4
  alternatives. Respondent covariates: segment (6 levels), year, miles,
  night-driving %, parking situation, gender, age, education, region,
  urbanicity, income -- each has a raw/integer-coded version and some have a
  finer-grained numeric version.
- `test.csv`: 4,997 rows, 263 respondents, **entirely disjoint from train's
  1,135 respondents** -- this is the single most important structural fact.
  No respondent-specific effect (random coefficients, mixed logit, memorized
  individual taste) can ever transfer to test; only effects driven by
  *observed* covariates/design content generalize.
- This is a genuine **partial-profile** design: every alternative has
  exactly 9 of its 19 attributes at a non-reference ("active") level in
  every row, a fixed structural constant of the experiment (not something
  that varies you could exploit directly). The conjoint is heavily blocked:
  each of the 19 task positions draws from ~296 distinct designs, ~3.8
  respondents/design on average; ~299 recurring "questionnaire versions"
  (probably Sawtooth CBC's default 300-version pool) exist across the full
  1,398 train+test respondents, ~4-5 respondents share all 19 tasks per
  version.
- Real, confirmed covariate shift: test respondents skew ~33% higher median
  income than train, with top income brackets 3x+ over-represented
  (adversarial-validation AUC 0.634 distinguishing train/test by covariates
  alone).
- Validation throughout is **strictly respondent-grouped** (never row- or
  task-based): canonical single 80/20 split (seed 7402) for cheap screening,
  canonical 5-fold CV (seed 4821) for confirmation, a further 5 repeated-CV
  seeds (1907, 2719, 6151, 8293, 104729) for near-misses, always pooled
  out-of-fold log loss, never averaged per fold. Promotion requires an
  ordinary 95% respondent-clustered bootstrap CI on the paired gain vs. the
  current best to exclude zero, plus >=5/6 positive repeats when escalated --
  a positive point estimate alone is never enough.

## Current best model (the one to beat)

`set_context_utility_network_v14`: **canonical CV log loss 1.143533,
public leaderboard 1.200** (benchmark/uniform-guess = 1.38629). It is a
blend: 88.9% x (an ensemble of a hand-built conditional logit + xgboost +
a shallow MLP) + 11.1% x a permutation-invariant feed-forward network whose
per-alternative features include mean/max-pooled summaries of the other
alternatives in the same choice task.

The dominant single component is a conditional logit with: all 19
attributes as factors (not linear -- confirmed non-linear), Price as a
saturated 12-level factor (not linear or quadratic -- confirmed strongly
non-linear and convex), Price/opt-out-propensity interacted with income,
age, miles, night-driving %, gender, urbanicity, education, market segment
(6 levels), survey-position/task-fatigue (price sensitivity rises across
the 19-task survey), region, and parking situation, plus two choice-set
*context* effects found late and worth a lot: `is_cheapest`/`is_dearest`
(rank flags) and `price_gap_min`/`price_gap_max` (magnitude of distance to
the choice set's cheapest/dearest alternative). This single logit alone
scores ~1.147 CV.

## Exhaustive list of what has already been tried and independently verified (DO NOT re-suggest these)

**Choice-model families:**
- Fixed/conditional logit (the backbone above) -- extensively tuned, this is
  the strong baseline.
- Mixed/random-coefficient logit: full 20-parameter random coefficients
  (badly overfits: better training log-likelihood, worse validation),
  Price-only random coefficient (negligible gain once observed heterogeneity
  is already in the model), a Bayesian hierarchical version via
  `bayesm::rhierMnlRwMixture` (deliberately re-tested under a totally
  different estimation philosophy specifically to rule out an estimation
  artifact -- still rejected, small confident harm).
- Latent-class / finite-mixture logit: task-fatigue-based 2-class EM
  (unstable, coefficients flip sign across folds, slightly worse than
  pooled), and a price-sensitivity-scale 2-class version (genuinely stable
  and reproducible classes, but statistically tied with the existing
  continuous covariate interactions -- redundant, not wrong).
- Nested logit (3 bundles nested against the opt-out): no improvement over
  a fixed zero-utility opt-out reference, tested independently by two team
  members.
- Random Regret Minimization: promising single-split screen, null-to-
  negative on honest CV.
- A genuinely dedicated two-part/hurdle decomposition: a freshly-trained
  binary opt-out propensity model (new task-difficulty features: price
  spread/CV across the choice set, count of attributes actually varying in
  the task, plus a frozen-model offset) and a freshly-trained 3-way
  conditional model among the inside bundles only (same core feature set,
  refit on the inside-only subsample, via ridge-penalized `glmnet`
  stratified-Cox since plain MLE hit quasi-complete separation on the
  smaller subsample) -- recombined as `P(opt-out)=q`, `P(bundle_j) =
  (1-q) r_j`. Rejected decisively: zero blend weight in every one of 30
  fold-fits (5 folds x 6 CV seeds) against the current best.
- A "two-head" experiment that separately re-weights (via a penalized
  convex combination, NOT fresh training) four already-fitted components'
  predictions for the opt-out margin vs. the conditional-bundle margin:
  closest-ever near-miss (89.6% bootstrap win rate) but CI still crosses
  zero after repeated CV; an OOF residual audit of this same result found
  the opt-out-margin reweighting has no signal at all and the bundle-margin
  reweighting has a small, real, but non-localizable signal.

**Model-form / functional-form / loss-function changes:**
- Exact-4-way-softmax shared-alternative-utility objective (a custom
  gradient-verified xgboost objective, and separately a weight-shared torch
  MLP applying the same encoder to every alternative under exact
  cross-entropy): cold-start fails outright; as a residual on top of the
  best logit, tiny effect, CI crosses zero. Two different function classes
  agree the *exact likelihood*/*exchangeability* was never the missing
  lever -- the hand-built features are doing the real work.
- Global (single-scalar) post-hoc temperature/shrinkage recalibration
  against both the ordinary AND a covariate-shift-target-weighted loss:
  identity is optimal, every deviation makes it worse, monotonically.
- Global utility-scale heterogeneity as a function of respondent/task
  covariates (task position, income, market segment): decisively harmful at
  every ridge strength tried.
- **Three additional small-parameter alternative probability *link* families**
  (holding the fitted utilities completely frozen, only changing the
  softmax-equivalent map from utility to probability): (i) single global
  scale exponent (replicates the above temperature-scaling null exactly);
  (ii) scale + a single opt-out-specific additive log-odds shift; (iii)
  scale + a shape exponent on the surprisal `-log(p)` (a semiparametric
  generalization in the spirit of the Marginal Distribution Model /
  distributionally-robust choice-probability literature, e.g. Natarajan et
  al. 2009 and Mishra/Natarajan/Padmanabhan/Teo/Li 2014, *Management
  Science*). **All three reject decisively, CIs entirely below zero, none
  even a near-miss.**
- A task-content-*conditioned* local temperature (a linear function of
  choice-set difficulty -- gap between the top-2 predicted bundle
  probabilities, price coefficient of variation -- ridge-shrunk toward
  plain softmax): also rejects decisively, CI entirely below zero; the
  fitted intercept alone reproduces the already-rejected global-scale
  result almost exactly, and the task-difficulty terms contribute nothing.
- A formal nested-cross-fitted check for whether per-task/per-respondent
  loss is *predictable* from test-time covariates and design content: it
  genuinely is (out-of-fold R^2 ~0.03-0.05, confirmed non-tautological by
  removing the model's own predicted probabilities from the feature set and
  finding the signal survives) -- but four independent attempts to *exploit*
  this predictability via rescaling/gating (the two items directly above,
  plus a difficulty-gated ensemble-weight blend) all failed. Read as
  irreducible aleatoric heterogeneity the model already calibrates for, not
  fixable miscalibration.
- Component-wise stagewise residual boosting on the fitted utility (a
  custom exact-softmax objective letting regularized selection pick terms):
  smallest-ever near-miss in the project, an order of magnitude below
  anything that has actually cleared the bar.
- A low-rank demographic-x-partworth factorization: single-split near-miss
  that *reverses sign* under repeated CV.
- Continuous three-way interactions (`Price x z(income) x z(mileage)`,
  deliberately avoiding an earlier binned-covariate quasi-separation
  failure): real, stable coefficient, but predictive gain unstable across
  folds, CI crosses zero; the one submission built on it scored *worse*
  publicly than predicted, traced to a handful of extreme-income test
  respondents the training-respondent bootstrap simulation underweighted.
- Choice-set geometry/crowding features (each alternative's pairwise
  attribute/price similarity to its two choice-set competitors, decomposed
  into set-mean/own-centered/nearest-excess terms, as ADDITIVE terms in the
  logit's raw attribute space): near-miss, crosses zero.
- A representation-gap fix to the set-context neural network specifically
  targeting this same "how similar are the 3 bundles" information, but in
  *learned embedding space* instead: added an explicit variance-across-
  alternatives pooling statistic (broadcast the same way the existing
  mean/max pooling already is, no other architecture change). Rejected at a
  cheap single-split screen stage (worse both standalone and blended) before
  any expensive retraining was committed.
- Covariate-shift correction: a genuine importance-weighted (Shimodaira-
  style density-ratio) refit of the best logit, evaluated against an honest
  target-weighted held-out loss: decisively harmful at every weighting
  strength (effective sample size collapses under weighting). Separately,
  quantile-mapped moment matching and confident self-training (transductive
  test-covariate adaptation): both null.
- Questionnaire-version effects: the ~299-version structure is real and
  confirmed (not sequential/cyclical assignment), but every attempted
  correction (a one-step Newton per-version opt-out correction, a
  neighbor-smoothed version-pool version) failed -- versions average only
  ~3-5 respondents, too sparse for even one well-shrunk correction
  parameter.
- Sequence/history effects: design-exposure history (observable-only, since
  a respondent's own past *choices* are unusable -- a whole future
  respondent's sequence is unlabeled simultaneously at test time) is real
  but too small after multiple-testing correction; a prior-smoothed version
  is the closest-ever near-miss on this angle but still crosses zero under
  repeated CV and a follow-up isolating just the price-anchoring term alone;
  a full respondent-sequence transformer, and a 6-configuration rescue sweep
  varying its regularization/initialization, closed this direction
  definitively -- relaxing the regularizers only reveals harm, never hidden
  signal.

**Non-choice-model-family attempts (all as ensemble diversity/comparison):**
- Wide xgboost (`multi:softprob`): consistently worse alone than the logit,
  modest ensemble diversity value.
- A real ranking-loss xgboost (`rank:ndcg`, task-grouped): meaningfully
  better than the naive xgboost alone, but only a small, statistically
  unconfirmed contribution in a larger blend.
- LightGBM and CatBoost: both clean, unambiguous negatives (zero ensemble
  weight in every fold for both).
- A calibrated RBF-kernel SVM: individually the weakest diversity candidate
  tried, and unlike every tree/neural diversity source, blending it actively
  *hurts* rather than helping.
- Shallow MLP (adopted, real ensemble diversity), a 2-layer torch MLP with
  dropout (real signal vs. the plain logit+xgboost blend, but does not beat
  the already-adopted shallow MLP), and the permutation-invariant
  set-context network (adopted, the current 11.1%-weight component).
- A regularized stratified-Cox reformulation of the conditional logit
  (`glmnet`, exploiting the Cox-partial-likelihood = conditional-logit
  equivalence) for automated interaction selection: independently
  rediscovers the hand-built interaction structure from a 195-term
  candidate pool, doesn't beat the hand-built version, but is a useful,
  independently-validated ~10% ensemble component.
- Stacking/meta-learning in place of a fixed arithmetic blend weight: a
  log-linear/geometric-pooling conditional-logit meta-model (2 components),
  a full nested ridge log-pool meta-model (6-8 components), and an xgboost
  meta-model: all null or actively harmful. Simple fixed-weight arithmetic
  averaging has never been beaten by anything learned.
- Bagging in both directions: seed-bagging the minority xgboost/MLP
  components is real alone but the gain nearly vanishes once blended at
  their small ensemble weight; bootstrap-resampling the *dominant*
  conditional-logit component is decisively harmful (a smooth low-variance
  MLE doesn't benefit from bagging).
- The best-ever arithmetic blend across 8 diverse components: ordinary 95%
  CI barely excludes zero vs. the (then-)current best, but 99%/Bonferroni-
  adjusted CIs cross zero -- correctly not promoted, and a repeated-CV
  refit pulled even the ordinary CI back across zero.

**Diagnostics that closed off other directions (not model attempts, but rule out hypotheses):**
- Calibration is excellent and flat across every slice checked (segment,
  task-position, region, true class) -- no exploitable miscalibrated
  subgroup by simple slicing.
- The confirmed income-based covariate shift explains only ~19% of the
  CV-to-public gap (bootstrap CI includes zero on the point estimate); the
  rest is unexplained but not attributable to any single tested mechanism.
- A design/block-structure empirical-frequency shrinkage (same exact
  design recurs for ~98.5% of test tasks, ~3.8 respondents/design on
  average): negligible gain, too few respondents per design cell for a
  reliable empirical estimate.

## What "shares this exact or a very similar dataset" already means here

The Marginal Distribution Model literature (Natarajan et al. 2009; Mishra,
Natarajan, Padmanabhan, Teo, Li 2014, *Management Science*, "On Theoretical
and Empirical Aspects of Marginal Distribution Choice Models") is understood
to have been applied to a General Motors safety-feature conjoint dataset
resembling this one. That family's core idea (semiparametric/
distributionally-robust choice probabilities replacing the softmax/Gumbel-
independence assumption) has now been tested here in small-parameter form
and rejected (see the "alternative probability link" item above). If you
know this literature better, useful follow-ups would be: (a) does the full
semidefinite-program version of MDM (not just a small-parameter
approximation) have a tractable way to be estimated on ~17,000 training
choice tasks with 4 alternatives each in R, without a GPU cluster; (b) are
there other empirical papers using this specific GM dataset (or a
near-identical one) whose reported best log-likelihood/log-loss would tell
us whether ~1.14 CV / 1.20 public is already close to that dataset's
practical ceiling, independent of this project's own internal diagnostics.

## What a useful answer looks like

For each candidate you propose, answer all of:
1. Exactly which modeling assumption does it change (be specific: is it the
   utility function, the link/probability function, the estimation
   objective, the ensemble/combination rule, or something else)?
2. Why does the "already tried" list above not already cover it -- what is
   structurally different?
3. What information does it use at test time, and can you confirm every
   input is computable for the 263 entirely-new test respondents without
   using any test choice/outcome?
4. Roughly how many free parameters, and what is the realistic
   identification risk given only 1,135 training respondents (908 in any
   training fold)?
5. Expected computational cost in R (packages available: `mlogit`, `dfidx`,
   `xgboost`, `glmnet`, `survival`, `torch`, `lightgbm`, base `stats`/`nnet`;
   no GPU, moderate CPU-only compute budget).
6. A minimum viable implementation sketch -- do not propose starting with
   the largest/most flexible version.
7. What leakage risk exists and how you'd rule it out.
8. What result, if you saw it, would tell you this idea is wrong (a
   falsification criterion), not just "if the number goes up, it's right."

Do not produce a generic literature survey. Rank at most 3 candidates by:
structural novelty relative to the list above, plausibility of a real
mechanism (not just "add more flexibility"), identifiability with this
sample size, test-time availability, and computational cost. If nothing
clears a reasonably high bar, say that plainly instead of forcing a ranking.
