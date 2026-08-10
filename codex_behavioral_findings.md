# Behavioral-model round: design history, regret, and neural diversity

## Verdict

No Kaggle submission was made. `ensemble_v11` remains the adopted model
(canonical CV `1.145094212987`, public `1.202`).

All comparisons below reuse the saved canonical respondent folds (seed 4821),
hard-assert the known m8trpg and ensemble_v11 OOF losses before comparing a new
model, and keep complete respondent blocks together. The generated artifacts
are in the gitignored directory
`data_processed/codex_behavioral_round/`. None of `AGENTS.md`,
`cleaning_log.md`, or `submissions_log.csv` was edited.

## 1. Observable design-exposure history

`R/codex_design_history.R` constructs history solely from alternatives shown in
earlier tasks for the same respondent. Task 1 is set to zero, task `t` uses only
designs from tasks `1,...,t-1`, alternative 4 remains zero, and no `Ch*` outcome
enters feature construction.

The price-reference hypotheses failed the seed-7402 screen. Relative to the
running mean of previously shown prices worsened log loss by `0.001182`;
relative to the preceding task worsened it by `0.000649`; their combination
worsened it by `0.001205`.

Attribute exposure was more interesting. The best three-term specification
uses:

- the share of the current alternative's active attribute levels never shown
  previously;
- their average prior exposure frequency; and
- the maximum level-match similarity to any previously shown alternative.

Its coefficients had the same signs in all five folds. The canonical pooled
OOF results were:

| Comparison | Baseline | Candidate | Gain |
|---|---:|---:|---:|
| m8trpg versus m8trpg + attribute history | 1.147021211 | 1.146805776 | +0.000215435 |
| fixed 80/20 ensemble_v11 replacement | 1.145094213 | 1.144864007 | +0.000230206 |
| fold-cross-fitted shrinkage toward the history model | 1.145094213 | 1.144867640 | +0.000226573 |

For the fold-cross-fitted result, four outer folds selected full weight on the
history-enhanced mlogit and one selected 0.95. A 100,000-replicate
respondent-clustered bootstrap gave an ordinary 95% CI of
`[+0.0000078, +0.0004471]` and a 97.875% win rate.

This does **not** clear the submission bar after accounting for selection among
eight related CV-confirmed history variants. The 99% interval was
`[-0.0000625, +0.0005155]`, and the Bonferroni family-wise 95% interval was
`[-0.0000796, +0.0005334]`. The signal is technically positive under an
unadjusted 95% interval but too close to zero to call clearly established after
candidate search. It was not adopted.

`R/codex_history_precision.R` reproduces the high-precision bootstrap,
fold-cross-fitted shrinkage, 99% interval, and multiplicity-adjusted interval
without rerunning the mlogit fits.

## 2. Random Regret Minimization

Apollo was not installed, so `R/codex_rrm.R` implements a scoped classical RRM
likelihood directly:

`R_i = sum_(j != i) sum_k log(1 + exp(beta_k (x_jk - x_ik)))`,
with choice probabilities proportional to `exp(-R_i)`.

Attributes enter as level dummies, while price was tested as both continuous
and factor-coded. The fuller specifications also retain the already validated
observed-heterogeneity, task, and price-context utility terms. The analytic
gradient was checked against centered finite differences on twelve randomly
selected parameters; the maximum absolute discrepancy was below `5e-10`.
Two deterministic screen starts converged to essentially identical objectives
for every specification.

The single split was promising:

| RRM specification | Component loss | Best RRM weight | Blend loss | Screen gain |
|---|---:|---:|---:|---:|
| full, continuous price | 1.162579 | 0.36 | 1.159670 | +0.000899 |
| full, factor price | 1.160339 | 0.54 | 1.159584 | +0.000985 |

Both were therefore run through canonical five-fold CV. The apparent benefit
did not transfer:

| RRM specification | Component CV | Globally optimized blend | Honest cross-fitted blend | Honest gain | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| full, continuous price | 1.149974 | 1.144952 | 1.145216 | -0.000122 | [-0.000514, +0.000242] |
| full, factor price | 1.148103 | 1.145071 | 1.145321 | -0.000226 | [-0.000499, +0.000031] |

The globally optimized weights are reported only as an optimistic diagnostic;
the honest result selects each fold's weight using the other four folds.
RRM is a respectable standalone competitor to m8trpg, especially with
factor-coded price, but it did not add transferable ensemble diversity. It was
not adopted.

The formulation follows:

- Chorus, C. G. (2010), “A New Model of Random Regret Minimization,” *European
  Journal of Transport and Infrastructure Research*:
  <https://journals.open.tudelft.nl/ejtir/article/view/2881>
- Hess, S. and Palma, D., *Apollo manual*, RRM model component:
  <https://www.apollochoicemodelling.com/files/manual/Apollo.pdf>

## 3. Non-tree neural ensemble member

The preferred embedding backends (`keras`, `tensorflow`, and `torch`) were not
available. `R/codex_mlp_ensemble.R` therefore uses `nnet` for a small
single-hidden-layer softmax network with 306-307 one-hot and continuous inputs,
including alternative-specific attribute levels, factor-coded price,
respondent covariates, task position, and price context.
The exact count varies only because a rare income-level dummy is absent from
some training folds; train/test columns are explicitly aligned for every fit.

The seed-7402 screen averaged three independent seeds for each of nine
size/decay configurations. The best was eight hidden units with decay 0.1:
component loss `1.237758`, but an 8% blend weight improved the screen
ensemble from `1.160568` to `1.159999` (gain `0.000569`). All screen fits hit
the fixed 200-iteration cap. A diagnostic 600-iteration rerun improved the
training objectives but worsened validation (`1.253686` component,
`0.000331` best blend gain), confirming that extra optimization was fitting
noise rather than resolving a useful underfit.

The 200-iteration cap was therefore fixed explicitly as early stopping before
canonical five-fold evaluation; it was not re-selected within the five folds.

The result survived canonical five-fold CV:

| Comparison | Baseline | Candidate | Gain |
|---|---:|---:|---:|
| standalone MLP | -- | 1.190543350 | -- |
| globally optimized v11 + MLP (optimistic diagnostic) | 1.145094213 | 1.143686618 | +0.001407595 |
| fold-cross-fitted v11 + MLP | 1.145094213 | **1.143789442** | **+0.001304771** |

The fold-cross-fitted MLP weights were 0.14, 0.17, 0.15, 0.15, and 0.13.
Four folds improved and one worsened. A 100,000-replicate respondent-clustered
bootstrap reproduced the point gain and gave:

- ordinary 95% CI `[+0.000092, +0.002513]`;
- 98.24% bootstrap win rate;
- 99% CI `[-0.000276, +0.002897]`; and
- nine-configuration Bonferroni 95% interval
  `[-0.000396, +0.003007]`.

This **clears the project's pre-stated ordinary respondent-bootstrap 95% bar**,
unlike every other lead in this round. It is still important not to oversell
the evidence: the lower endpoint is small, and stronger intervals that account
for the nine-configuration screen cross zero. The honest description is a
CV-confirmed candidate under the project's established protocol, not a
guaranteed leaderboard improvement.

`R/codex_mlp_precision.R` reconstructs the fold-specific blend from saved OOF
predictions, asserts exact baseline/component/candidate losses, checks positive
normalized probabilities and alignment, and reproduces all high-precision
intervals.

### Extended ensemble diagnostic

`R/codex_mlp_extended_ensemble.R` also allowed the MLP to compete alongside the
earlier rank-xgboost, retuned xgboost, and glmnet-Cox components. The best
five-family fold-cross-fitted loss was `1.143129`, a gain of `0.001966` over
v11 (95% CI `[+0.000404, +0.003490]`). However, its incremental gain over the
simple fixed-v11 + MLP candidate was only `0.000661`, with 95% CI
`[-0.000279, +0.001609]`. The extra three-family machinery is therefore not
justified by a confirmed incremental benefit. The simpler candidate retains
v11's already demonstrated public-facing variance reduction and is the
preferred option for review.

### Candidate file prepared, not submitted

`R/codex_mlp_candidate_submission.R` fits the frozen MLP specification on all
training respondents using five seeds, then gives it 15% weight against the
exact existing `submission_ensemble_v11_pricegap.csv` probabilities. Using the
saved v11 CSV avoids silently changing the already public-scored
mlogit/xgboost random state.

It generated `submission_codex_mlp_v12_candidate.csv` locally with 4,997 rows,
matching `No` values, no missing probabilities, minimum probability
`0.009452`, and maximum row-sum error below `2e-15`. It was **not submitted to
Kaggle**. Claude should review the scripts and raw artifacts before deciding
whether to use a future submission slot.
