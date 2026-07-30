# Analytics Edge Data Competition 2026 -- Project Memory / State Summary

*This file doubles as (a) project memory for AI coding assistants working in this
repo, and (b) a self-contained brief you can paste into another LLM (GPT, Claude,
etc.) to get a second opinion on what might be missing. Everything needed to
understand the project is inline below -- no need to read other files first.*

## Competition setup
- Kaggle competition: predict which of 4 car safety-feature bundles a respondent
  chooses (discrete choice / conjoint task). Alternative 4 is always a constant
  "opt-out"/no-purchase option (all attributes and price = 0).
- Scored by multi-class log loss. Benchmark (predict 1/4 for every alternative)
  scores 1.38629.
- Dates: competition ends 1 Aug 2026 (12:00 SGT); report due 10 Aug 2026.
- Max 2 Kaggle submissions/day, shared across the whole team (one Kaggle account).
  R only for submissions, but any package (and AI assist) allowed, as long as
  sources are cited in the report for non-class methods.
- Report requirements: max 8 pages, NO executive summary, NO appendix. Must cover
  (i) best model on public LB + brief discussion of alternatives tried,
  (ii) public vs. private leaderboard fit discussion,
  (iii) insights + limitations.
- Grading: 7 pts private LB, 8 pts public LB (>= benchmark gets partial credit
  on both), 15 pts report.
- Public leaderboard is ~70% of the test set (~3,500 of 4,997 rows) per the
  competition's own README description; private (final grading) leaderboard is
  the other ~30%, revealed only after the competition closes.

## Data
- `train.csv`: 21,565 obs, 113 vars. 1,135 respondents (`Case`), each completed
  exactly 19 choice tasks (`Task`). Columns: 19 attribute codes (CC, GN, NS, BU, FA,
  LD, BZ, FC, FP, RP, PP, KA, SC, TS, NV, MA, LB, AF, HU) + Price, each suffixed
  1-4 for the 4 alternatives (integer level codes, different max per attribute,
  e.g. CC 0-3, NS 0-5, BU 0-6, Price 1-12 for alts 1-3 / 0 for the opt-out).
  Outcome: Ch1-Ch4 (one is 1, rest 0 per row). Respondent covariates: segment,
  year, miles, night, ppark, gender, age, educ, region, Urb, income -- each has a
  raw text version, an integer `*ind` version (clean 1:1 recode), and some also a
  `*a` version (finer-grained numeric estimate *within* the `*ind` bin -- NOT a
  redundant duplicate, e.g. `milesa` holds actual values like 60/80/100 within the
  "51-100 miles" bin; genuinely richer information than `*ind`).
- `test.csv`: 4,997 obs, same schema, Ch1-Ch4 are NA (the targets to predict).
  **Test respondents (Case 1136-1398) are entirely disjoint from train (Case
  1-1135) -- zero overlap.** This is the single most important structural fact
  in the whole project: it means respondent-specific personalization (random
  effects, mixed logit, memorized individual coefficients) CANNOT transfer to
  test, no matter how well it fits training data. Only OBSERVED heterogeneity
  (covariates/segment/task/region interacted with attributes) generalizes. This
  is why every mixed-logit attempt underperformed the fixed-effects models with
  the same observed-heterogeneity terms, and why respondent-level (not row-level,
  not task-level) train/val splitting is mandatory for any honest local score.
- `sample_submission.csv`: format `No, Ch1, Ch2, Ch3, Ch4`, `No` = test row id
  (21566-26562).
- No missing values or duplicate rows in train. Opt-out chosen 30.2% of the time
  overall; opt-out share rises from ~24% (Task 1) to ~34% (Tasks 15-19) --
  survey-fatigue effect, `Task` position is a usable transferable predictor.

## Key technical/identification findings
1. **ASC identification conflict.** Dummy-coding attribute levels while keeping
   alternative-specific intercepts (ASCs) causes a singular Hessian in `mlogit`,
   because alt 4's all-reference-level profile never co-occurs with alts 1-3, so
   ASC4 is unidentifiable once attributes are factor-coded. Fix: drop ASCs
   entirely (alt 4's utility fixed at 0 by construction) or add dummies only for
   alts 2-3.
2. **Price-as-factor collinearity (found 2026-07-26).** Turning Price from a
   continuous slope into a full 0-12 level factor makes the Hessian exactly
   singular again, for a different, non-obvious reason: every inside alternative
   has **exactly 9 of its 19 attributes at a non-reference level** (a constant of
   this partial-profile conjoint design -- confirmed via frequency tables), so
   `sum(19 attribute-active indicators)/9` reconstructs the "inside" indicator
   exactly, the same way `sum(all 12 price dummies)` does. Two different column
   sets both exactly reproducing "inside" is a rank-1 collinearity. Diagnosed via
   `model.matrix()` + `qr()` rank / regressing the dropped column on the rest
   (fast, avoids repeatedly re-fitting mlogit to guess) rather than trial and
   error. Fix: drop one price level's dummy (levels 2-12, reference = level 1;
   opt-out's Price=0 shares that reference cell, which is fine since "inside vs
   opt-out" is already fully captured by the attribute block).
3. For `mlogit` panel/mixed models, use nested dfidx syntax:
   `dfidx(data, idx = list(c("chid","Case"), "alt"), choice = "chosen")`.
4. When aligning `predict()` output back to original rows, always match by the
   `chid` rownames `predict()` returns -- never assume row order is preserved.
   This silently mis-aligns and inflates log loss if skipped.

## Reproducibility pipeline
- `R/log_loss.R`: matches the competition's exact metric; verified against
  benchmark (1.386294 for uniform 1/4 guess).
- Canonical single-split validation: long-format reshape + **respondent-level**
  80/20 split (seed 7402), saved to `data_processed/train_val_split.rds`
  (gitignored, regenerate from `train.csv` locally -- see `competition_report.qmd`
  for the reshape code, or any of the `R/*.R` scripts for a from-scratch version).
- Canonical 5-fold CV: respondent-grouped, seed 4821, pooled out-of-fold log loss
  (not averaged per-fold, pooled across all folds' held-out predictions). Used to
  confirm every real finding below -- single-split screening first (fast), CV
  confirmation before trusting a result.
- `submissions_log.csv`: every model tried, submitted or not, with val/CV log
  loss, public LB score, gap, and source citation. Single source of truth for
  what's been tried -- check before re-trying something.
- `cleaning_log.md`: full narrative log of every data/modeling finding, in
  chronological order, with the reasoning behind each result (positive or
  negative). More detail than this file; read it for the "why", not just "what".
- `R/` scripts of note: `cv_price_factor_context.R` and `cv_ensemble_v10.R` are
  the current canonical 5-fold CV harnesses (mlogit, and mlogit+xgboost blend
  respectively); `submit_ensemble_v11.R` generates the ensemble_v11 base
  (still used as the frozen baseline the MLP candidate blends against, see
  `R/codex_mlp_candidate_submission.R`); `R/codex_mlp_candidate_submission.R`
  generates the current best submission; `error_analysis.R` / `calibration_check.R`
  are the diagnostic scripts behind
  the "is there more signal left" analysis below.
- Local environment note: this machine's security policy blocks some compiled
  tidyverse DLLs (tibble/utf8 printing) -- use plain data frames (`as.data.frame()`)
  instead of tibbles if this happens. Also: xgboost 3.x's R API no longer exposes
  `$evaluation_log` on the returned booster object (bare external pointer only)
  -- pick nrounds from the printed per-iteration curve or via `xgb.cv()` instead
  of relying on `$best_iteration`/`$evaluation_log`.

## Model progression (best single logit and best ensemble at each stage)
CV = 5-fold respondent-grouped, seed 4821, pooled. Val = single 80/20 split, seed 7402.

| Model | Val | CV | Public LB | Notes |
|---|---|---|---|---|
| benchmark (uniform 0.25) | 1.386 | -- | 1.38629 | sanity check |
| mod1: continuous conditional logit + ASCs | 1.236 | -- | 1.270 | first submission; gap 0.034 |
| mod2b: factor-coded attributes + alt2/3 dummies, no ASC | 1.219 | -- | -- | factor-coding attribute levels beats linear (non-linearity confirmed) |
| mod6: mod2b + Price x covariate (income/age/miles/night) | 1.205 | -- | -- | biggest single gain from observed heterogeneity |
| mod7: mod6 + inside-good x covariate | 1.202 | -- | 1.230 | gap 0.028 |
| mod8: mod7 + Price/inside x segment (6 levels) | 1.190 | 1.166 | -- | segment is strong, not diminishing returns |
| m8t: mod8 + task-fatigue (Price x centered task position) | 1.187 | 1.162 | -- | price sensitivity rises over the 19-task survey |
| m8tr: m8t + region x / ppark x (previously unused covariates) | 1.170 | 1.157 | -- | best single mlogit before this session |
| xgboost (wide-format, multi:softprob, nrounds=73) | 1.204 | 1.179 | -- | worse alone; different error pattern, useful in ensemble |
| ensemble_v9: 0.70 m8tr + 0.30 xgboost | -- | 1.152 | 1.204 | gap jumped to 0.052 (vs 0.028-0.034 on simpler models) |
| **mlogit_m8trp: m8tr + Price-as-12-level-factor + is_cheapest/is_dearest** | 1.166 | **1.152** | -- | single logit now matches the whole v9 ensemble; see finding #2 above |
| ensemble_v10: 0.75 m8trp + 0.25 xgboost | -- | 1.148 | -- | not submitted (superseded before a slot was used) |
| **mlogit_m8trpg: m8trp + price_gap_min/max (distance to cheapest/dearest, not just rank)** | 1.160 | **1.147** | -- | biggest single incremental gain of the session, from just 2 params |
| ensemble_v11: 0.80 m8trpg + 0.20 xgboost | -- | 1.145 | 1.202 | best model 2026-07-26/27; gap 0.057. Superseded 2026-07-28 |
| ensemble_v11 + 0.15 MLP (nnet, 8 hidden units, 5 seeds) | -- | 1.1438 | 1.201 | best model 2026-07-28/29/30; gap 0.0572. First non-tree ensemble member. Superseded 2026-07-30 |
| **0.889 x (ensemble_v11+MLP) + 0.111 x set-context network** | -- | **1.1435** | **1.200** | **CURRENT BEST** (`set_context_utility_network_v14`), both CV and public. First learned choice-set-context component; gap 0.0565, in line with the established pattern |

Negative/null results (all real attempts, logged for the report's "alternatives
tried" section, not dead ends to re-try):
- Nested logit (bundles vs. opt-out): no improvement over the fixed zero-utility
  opt-out reference.
- Mixed logit (any variant -- full random, Price-only random, on top of mod8):
  either overfits badly (full random) or gives negligible gain (Price-only, once
  observed heterogeneity is already in the model) -- expected given test
  respondents are entirely new.
- Attribute x segment (mod9): slightly worse than mod8; feature valuation doesn't
  vary by segment beyond what Price/inside x segment already capture.
- Relative price (Price - task mean, applied to inside goods): near-zero effect,
  degenerate by construction (a constant shift across alternatives cancels in
  the logit). The *rank-based* version (is_cheapest/is_dearest) is what carries
  signal -- relative price only matters through rank/magnitude vs. the choice
  set's extremes, not a mean-centered shift.
- Relative feature load (attribute sum vs. task mean): noise, no effect.
- Price x {gender, urbanicity, education}: individually looked significant
  (p<1e-8) but single-split validation got WORSE (1.1699 vs 1.1657) -- a
  significance-vs-validation disconnect, left out.
- Quadratic task-fatigue term (P_task^2): same story, looked significant,
  validation got worse.
- Choice-structured xgboost (binary chosen/not-chosen on long-format rows,
  renormalized within task, instead of wide-format multi:softprob): null result,
  1.2053 vs 1.2042 for the original -- naive renormalization doesn't actually
  teach the model the within-task comparison structure. A real ranking-loss
  objective (`rank:pairwise`/`rank:ndcg` with `qid` grouping) might, but wasn't
  tried -- this is genuinely still open if someone wants to push xgboost further.
- Binned (categorical) Price x {age, miles, night} interactions, extending the
  Price-as-factor logic to other covariates: mostly negative. All three combined
  caused a large regression (1.1934) from extreme sparsity in nightind's top two
  bins (~6 respondents each -> one coefficient blew up to -0.815, quasi-
  separation). Individually: age alone (balanced, 5 levels) gave a tiny real
  gain (1.16495 vs 1.16573); miles alone was worse. Lesson: the continuous-to-
  categorical trick that worked for Price does NOT generalize automatically --
  it depends on per-cell sample size, must be checked category-by-category.
- Proper stacking (conditional-logit meta-model on log(p_mlogit), log(p_xgb),
  i.e. log-linear/geometric pooling instead of a fixed arithmetic blend weight):
  null result, 1.1461 vs the simple blend's 1.1451 (nested 5-fold CV). With only
  two base models and xgboost getting a small minority weight anyway, there's
  essentially one real degree of freedom in the ensemble and the simple weighted
  average already finds it.
- K-means "persona" clustering (5 clusters on scaled income/age/miles/night,
  fit on training respondents only) as a Price/inside interaction axis: negative,
  1.1657 vs 1.1597 without it -- redundant with the individual covariate
  interactions already in the model, re-slices existing information.

## Diagnostic: is there more legitimate signal left to find?
Ran a multi-angle check on the ensemble's out-of-fold predictions before
concluding this (see `cleaning_log.md`, 2026-07-26 entries, for full numbers):
1. **Calibration is excellent.** Binning every predicted probability against
   whether that alternative was actually chosen: predicted and actual match
   within 1-2 percentage points across the entire 0-0.8 probability range.
2. **"Confident misses" are the expected flip side of calibration, not a flaw.**
   49.5% of tasks are argmax-misses; 36.3% of those have a large gap between the
   top pick and the true alternative's probability. This is mathematically
   necessary for a well-calibrated model on a genuinely stochastic process (a
   60%-confident model IS wrong 40% of the time), confirmed by the calibration
   check rather than assumed.
3. **No identifiable subgroup drives the misses** -- confident-miss rate is flat
   across segment/task-position/region/true-class (0.15-0.21 everywhere). A
   fixable, generalizable pattern would show up as an outlier slice; none does.
4. **Log loss isn't concentrated in a few catastrophic failures** -- worst 10% of
   tasks account for 21% of total loss, not 80%+.
5. **xgboost (flexible, non-parametric, same raw covariates) doesn't out-predict
   the hand-built logit** (1.179 vs 1.147) and only earns 20% ensemble weight --
   if there were a large pocket of exploitable interactions/nonlinearities left
   in the existing covariates, a flexible learner should be finding more of it.
6. **Theoretical ceiling check:** the most flexible model ever tried (mod4, full
   random-coefficient mixed logit, 20 parameters per respondent) reached a
   *training* log-likelihood of -16449 over 17,252 tasks = training log loss
   ~0.953 -- and this is achieved via respondent-specific memorization that
   provably does NOT transfer (mod4's own validation score, 1.247, was worse
   than simpler models). Since test respondents are entirely new, the
   achievable floor via only-generalizable modeling sits above, not below, that
   ~0.95 ceiling.

**Conclusion:** this is a reasonably strong, multi-angle case that ensemble_v11
(CV 1.145, public 1.202) is close to the practical floor for this dataset via
legitimate, generalizable modeling -- not a claim that literally nothing more
exists, but real evidence rather than an assumption. Every actual attempt after
the price-gap finding (stacking, K-means, binned covariates beyond age) came
back negative, consistent with this.

## Resolved: external-review round (2026-07-26) -- every concrete idea tested
**The public leaderboard's current top score is reportedly in the low-1.1x
range** -- meaningfully below our 1.202. Two independent LLM reviews (fed this
file as a brief) both flagged the same set of checks; ran every concrete one
rather than taking them on faith (full detail in `cleaning_log.md`,
2026-07-26 "External review round" entries). **Conclusion: two genuinely new
structural facts confirmed and worth citing in the report; every idea aimed at
actually improving the score came back null once properly validated.**

- **Adversarial validation (`R/adversarial_validation.R`): real covariate
  shift confirmed.** AUC 0.634 distinguishing train/test respondents by
  covariates alone; driven mainly by income (test skews ~33% higher at the
  median, with top income brackets 3x+ over-represented). Partially explains
  the growing CV-to-public gap. A simpler univariate income-tercile check gave
  an ambiguous, not clearly-actionable signal (cleaning_log.md point 4). An
  actual weighted-likelihood refit was later tried (2026-07-27, Shimodaira-style
  density-ratio reweighting of the mlogit fit, not just a reweighted
  evaluation) and came back decisively negative -- every nonzero weighting
  strength made the model worse on an honest target-weighted held-out loss,
  bootstrap CI excluding zero on the harmful side. See "Resolved: covariate-
  shift refit and attribute-rank experiments" below. The shift itself remains
  real and report-worthy; it's just not fixable via this particular correction.
- **Design/block structure (`R/design_fingerprint_check.R`): a real, novel
  structural fact, but not exploitable.** The conjoint design is heavily
  blocked (~296 distinct designs per task position, ~3.8 respondents/design),
  and 98.5% of test tasks' exact designs recur in train. Tried exploiting this
  via empirical-frequency shrinkage per design cell -- negligible gain
  (0.00035, inside the noise floor), because too few respondents (~3-4) share
  each design for a reliable empirical estimate.
- **Bootstrap CV uncertainty (`R/bootstrap_cv_uncertainty.R`): now
  quantified.** Absolute CV noise SD ~=0.01; paired-comparison noise SD
  ~=0.001. This project's big wins (price-factor, price-gap) are solidly real;
  the smallest ones (xgboost's ensemble contribution) are real but closer to
  the edge than they looked; the stacking null result is confirmed correct
  (not a coin flip).
- **Latent-class logit (`R/cv_latent_class.R`): tried properly, null result,
  and unstable.** Both reviews independently flagged this as the single
  structurally different approach worth trying (unlike continuous mixed
  logit, class membership from *observed* covariates transfers to new
  respondents by construction). `gmnl` (the standard package) has no
  `predict(newdata=...)` method and no native "restricted" class structure,
  so hand-rolled a scoped EM version: fixed the confirmed m8trpg utility as a
  shared baseline, fit a small 2-class extension on just the task-fatigue
  terms, class membership from segment/income/age. Caught and fixed a real
  M-step bug along the way (a duplicated-row weighting scheme that was
  mathematically independent of the class posterior -- explained the
  suspiciously identical convergence across random restarts). With the fix,
  full 5-fold CV: **1.147826, slightly WORSE than the existing shared-
  coefficient model (1.147021)**, and the class-specific coefficients flip
  sign across folds (beta_task2 ranges -0.248 to +0.165) -- the multimodal-
  likelihood instability both reviews warned about, demonstrated concretely
  rather than avoided. Not adopted.
- **Latent-class logit, take 2 (`R/cv_latent_class_price_scale.R`, 2026-07-27):
  tested the CENTRAL idea properly, not just the safe substitute.** The
  original pitch from both reviews was price-sensitivity/opt-out segmentation,
  not task-fatigue -- task-fatigue was chosen for the first attempt specifically
  because it's additively safe from the price-factor collinearity. Went back
  and tested price-sensitivity directly via a class-specific MULTIPLICATIVE
  SCALE on the price-related linear predictor (avoids the collinearity a
  direct additive shift on Price/inside would recreate). Result this time:
  **stable, not multimodal** -- lambda converges to the same pair (~0.50 and
  ~1.85-2.0) across all 4 single-split restarts AND all 5 CV folds, a
  genuinely reproducible discrete split (roughly half vs. nearly double
  normal price sensitivity). But pooled CV log loss (1.147629) is essentially
  tied with the shared single-population model (1.147021, diff 0.0006, inside
  the noise floor). Interpretation: the heterogeneity is real, but the
  continuous covariate interactions already in the model (P_income, P_seg,
  P_age) already capture it in smooth rather than discrete form -- redundant,
  not wrong. This is a cleaner, more informative null than the task-fatigue
  attempt (real+stable+redundant vs. unstable+worse) and closes the
  latent-class question on both the scoped and central versions of the idea.

**Conclusion:** the two most theoretically credible levers (design-cell
empirical information, latent/discrete heterogeneity, tested in both its safe
and central forms) both failed on honest
validation. Combined with the earlier calibration diagnostic (well-calibrated,
no exploitable pattern in the misses, xgboost can't out-predict the logit),
the case that ensemble_v11 is near this dataset's practical ceiling for
legitimate, generalizable modeling is supported from multiple independent
angles, not just one. The confirmed income shift and blocked-design overlap
are real, report-worthy insights regardless -- cite them in the report even
though neither improved the score.

**2026-07-27 update -- the "1.187 mystery" is resolved, not just softened.** A
teammate's report that a competing team scored 1.187 public (only 0.015 below
our 1.202) prompted two follow-ups: (1) submitting `mlogit_m8trpg` standalone
(no xgboost) to test whether the blend was an "overfitting tax" -- refuted,
see the gap table below; (2) a properly respondent-clustered simulation of
public-LB-sized samples (184 respondents, matching the ~70% public share of
263 test respondents), since a naive row-level SE ignores that rows cluster
within respondents. Result: the clustered SD (0.0225) is 2.14x the naive
row-level estimate (0.0105), and **64.6% of simulated same-model draw-pairs
show a gap of >=0.015 purely from sampling luck** -- the observed gap is the
MAJORITY outcome between two equally-good models, not a rare or notable one.
Separately, quantified how much of the gap the confirmed income shift
explains: point estimate ~0.010 (~18% of the 0.057 total gap) via two
independent weighting schemes that agree closely, but the bootstrap 95% CI on
that estimate is [-0.0057, 0.029] -- includes zero, so even this can't be
stated as a confirmed nonzero effect. A follow-up model test (log-transformed
income, motivated by income's skew) came back null (1.1601 vs 1.159681,
negligible), consistent with per-bracket bootstrap CIs showing neither
flagged income bracket is statistically distinguishable from the average
once its own small sample size is accounted for (bracket 14, n=28: 95% CI
[1.10, 1.41] contains the overall mean 1.147; bracket 28, n=13: 95% CI
[0.94, 1.18] also contains it). **This is now a closed investigation, not an
open question**: no further lever has cleared the bar, and the 1.187 score
specifically should not be treated as evidence of a missing modeling
breakthrough.

## Known weakness: CV-to-public gap, and what actually drives it (updated 2026-07-30)
| Model | CV/Val | Public | Gap |
|---|---|---|---|
| mod1 | 1.236 | 1.270 | 0.034 |
| mod7 | 1.202 | 1.230 | 0.028 |
| ensemble_v9 | 1.152 | 1.204 | 0.052 |
| ensemble_v11 (mlogit+xgboost blend) | 1.145 | 1.202 | 0.057 |
| mlogit_m8trpg standalone (no xgboost) | 1.147 | 1.213 | 0.066 |
| ensemble_v11 + MLP | 1.1438 | 1.201 | 0.0572 |
| triple_mlp_v13 (triple-interaction+xgb+MLP) | 1.1433 | 1.210 | 0.0667 (largest) |
| set_context_utility_network_v14 (current best) | 1.1435 | 1.200 | 0.0565 |

The MLP candidate's gap (0.0572) sits right in line with ensemble_v9/v11's
0.052-0.057 range -- another ensemble-class model, another similar gap, no
new anomaly. It's also the first case this session where a CV-predicted gain
(0.0013, bootstrap CI barely excluding zero) showed up on the public LB in
the same direction (observed gain 0.001) rather than vanishing or reversing
-- a small, encouraging data point that the project's CV protocol is tracking
real signal, not just noise, even for gains this close to the noise floor.

Tested and refuted the obvious hypothesis: that the blend's small CV gain
(0.0019 from adding xgboost, right at the noise floor) was an "ensemble
complexity tax" and a leaner standalone logit would transfer better. It
didn't -- the standalone logit's gap (0.066) is the LARGEST in the project,
worse than the full ensemble's (0.057), despite being the simpler model. The
real pattern: blending in xgboost (which never beats the logit alone, CV
1.1787 vs 1.147) genuinely reduces the public-facing gap rather than adding
to it -- a real, confirmed instance of ensemble variance reduction, not
assumed. **Practical takeaway: keep the xgboost blend; gap size alone is not
a reliable signal of which model generalizes better, and the earlier
"complexity always grows the gap" framing was too simple.** Prompted by a
teammate reporting a competing team's public score of 1.187 (only 0.015 below
ours, not the "low-1.1x" gap originally assumed) -- this test doesn't explain
that score; it remains open.

**2026-07-29 update -- `triple_mlp_v13` submitted, confirms the pattern a
second way.** Best unsubmitted CV number in the project (1.143328, only
+0.000462 vs. the current best, CI crossing zero) was submitted as the
honest best-available bet once repeated CV closed off every other lead.
Result: public **1.210**, worse than the current best (1.201) and outside
the paired-simulation's predicted 95% range ([1.1979, 1.2033]) by 0.0067 --
not just noise playing out inside the simulation's own stated uncertainty.
Its gap (0.0667) is now the largest in the project, close to the
standalone-mlogit gap (0.066) rather than the ensemble-class 0.052-0.057
range. Mechanism: the model's flagged extrapolation risk (an unbounded
Price x z(income) x z(mileage) term, most sensitive to a handful of
extreme-income respondents) was underestimated by a simulation built from
training-respondent resampling, given that test is known to contain
proportionally more such extreme respondents than training. **Second real
submitted data point (after the m8trpg-alone test) showing a model with
extra flexible structure generalizing worse publicly than its CV number
alone predicted, while the plainer ensemble continues to hold the best
public score** -- concrete evidence for the report's "why the ensemble
should be retained" argument, not just a repeated assertion of it.

**2026-07-30 update -- `set_context_utility_network_v14` submitted,
CV-predicted direction confirmed a second time.** A set-context feed-forward
network (permutation-invariant summaries of the other alternatives in each
task, learned end-to-end) cleared repeated CV's ordinary 95% bootstrap bar
(pooled gain +0.0011636, CI [+0.0000106,+0.0023180], 6/6 repeats positive)
after a wide, unconfident canonical-split near-miss -- the reverse of this
session's usual near-miss-shrinks-under-repeated-CV pattern. Submitted after
a full-data build enforced by a two-independent-run reproducibility gate
(0.0 difference between runs). Result: public **1.200**, beating the prior
best (1.201) -- the SECOND time this project's CV-predicted improvement
direction has been confirmed on the public leaderboard (the first was the
original MLP candidate). Gap (0.056467) sits squarely inside the established
ensemble-class 0.052-0.057 pattern, in sharp contrast to the two flexible
submissions that broke it (m8trpg-alone 0.066, triple_mlp_v13 0.0667) -- a
third independent data point for the same conclusion: well-behaved,
ensemble-class refinements transfer more reliably than additions with
unbounded/high-variance structure. `set_context_utility_network_v14`
(1.143533 CV / 1.200 public) is now the current best model.

## Resolved: Codex modeling push (2026-07-27) -- 4-way ensemble candidate, not adopted
Asked Codex (a separate coding agent, working in isolation on branch
`codex-modeling`, commit `b37ecc8`, now merged into `zhenhao`) to push toward a
genuine sub-1.200 public score using the "genuinely still open" ranking-objective
xgboost idea flagged in the negative-results list above, plus a broader xgboost
retune and a from-scratch reconstruction of the 2026-07-25 glmnet stratified-Cox
regularized logit (the original script was never committed). Independently
reviewed every new script line by line (not just the write-up in
`codex_findings.md`) before deciding whether to submit -- full review process,
including one stale-artifact inconsistency found and resolved (didn't affect the
headline numbers below), in `cleaning_log.md`, 2026-07-27.

- **`rank:ndcg` xgboost (real ranking loss, `qid`-grouped by choice task) is a
  genuinely better xgboost**: 1.163610 alone vs. the original's 1.178668,
  confirming this specific idea (naive binary renormalization had failed; a real
  ranking objective doesn't have the same problem). Retuning xgboost's own
  hyperparameters more broadly helps similarly little (1.176029).
- **The reconstructed glmnet-Cox model** (1.164331, nested 5-fold CV) is a new,
  more rigorous result than the original single-split-only version, and takes a
  stable 9-13% weight in every ensemble it enters.
- **Best 4-way blend (mlogit 68% / rank:ndcg 10% / retuned xgboost 12% /
  glmnet-Cox 11%): 1.144363 under honest fold-cross-fitted weight selection**,
  vs. ensemble_v11's official 1.145094 -- a ~0.0007 gain. A respondent-clustered
  bootstrap (1000 resamples, same method as the bootstrap-uncertainty section
  above) puts this at mean gain 0.000742, win rate 93.3%, **95% CI
  [-0.000214, 0.001775] -- crosses zero.**

**Not adopted; no Kaggle submission made.** Two independent reasons: the
bootstrap CI includes zero (not distinguishable from noise by this project's own
established standard), and separately, this candidate roughly doubles the model
count behind the current best (2 components -> 4) at a moment when the project
has already confirmed that added complexity here tends to WIDEN the public-LB
gap even when CV improves (see the gap table above) -- exactly the wrong
direction to bet a submission slot on for an unconfirmed CV gain. All 5
`R/codex_*.R` scripts and `codex_findings.md` are kept in the repo for
reproducibility, same as every other tested-but-not-adopted model here.

## Resolved: covariate-shift refit and attribute-rank experiments (2026-07-27) -- both negative
Follow-up ask to Codex (branch `codex-shift-ranks`, commit `51ad3d8`, merged
into `zhenhao`): two specific, previously-untested leads rather than another
generic optimization pass -- an actual weighted-likelihood refit for the
confirmed income/covariate shift (only its *evaluation* had been reweighted
before, never a retrain), and extending the price_gap/is_cheapest/is_dearest
rank mechanism (the biggest single win of the session) to the other 19
attributes. Independently reviewed the code and cross-checked every reported
number against the raw generated CSVs before accepting the write-up -- both
held up exactly, no corrections needed this time.

- **Importance-weighted refit: decisively negative.** A genuine density-ratio
  weighted-MLE refit of mlogit_m8trpg (Shimodaira 2000), evaluated against a
  fixed, honest target-weighted held-out loss. `alpha=0` exactly reproduces the
  known baseline (confirms the machinery is correct); every nonzero weighting
  strength makes the model worse, monotonically, in all 5 folds. At the
  gentlest setting tested, bootstrap 95% CI **[-0.002614, -0.000422] --
  excludes zero on the harmful side.** Effective sample size collapses under
  weighting (908 -> ~521 of 908 respondents at full strength), and the
  variance cost outweighs any targeting benefit. The shift itself is still
  real; this particular correction just doesn't work.
- **Attribute min/max rank flags: null, didn't replicate.** Single-split screen
  looked promising (-0.000803) but reversed under 5-fold CV (+0.000254, worse
  in 4/5 folds, bootstrap CI [-0.001138, 0.000572] crossing zero). Likely
  cause: unlike price, most attribute codes are categorical labels without a
  stable cardinal ordering, so min/max comparisons on them aren't as
  meaningful.

**Not adopted; no submission made.** Closes out both leads identified as
genuinely open after the ensemble-candidate review above -- `ensemble_v11`
remains the best and current submission. Full detail in `cleaning_log.md`,
2026-07-27, and `codex_shift_rank_findings.md`.

## Resolved: continuous higher-order interactions (2026-07-27) -- one genuine but unconfirmed clue
Third and (for now) final follow-up to Codex (branch `codex-triple-products`,
commit `d987efa`, merged into `zhenhao`), explicitly designed to avoid the
earlier binned-covariate failure by using **continuous** three-way products
(`Price x z(covariate1) x z(covariate2)`, no binning -- a different risk
profile from the sparse-cell quasi-separation that broke the binned version).
Independently reviewed and cross-checked every number against the raw
generated CSVs -- exact match, same clean result as the prior two rounds.

**Best candidate: `Price x z(income) x z(mileage)`.** Coefficient is negative
and stable across all 5 CV folds (-0.0541 to -0.0270) -- a real signal in the
parameter. But the predictive gain isn't as stable (3 folds improve, 2
worsen): blended into the *fixed* (not re-optimized) ensemble_v11 weight,
1.145094 -> 1.144599 (+0.000495). Respondent-bootstrap 95% CI
**[-0.000689, 0.001660] -- crosses zero**, so it doesn't clear the submission
bar. Segment-specific mileage-x-price slopes, tested alongside it, were
decisively harmful (CI excludes zero on the harmful side).

**Not adopted; no submission made.** Flagged as a clue worth revisiting only
if independent evidence appears, not a validated improvement.

**Where this leaves the modeling search:** three consecutive, independently-
verified rounds (4-way ranking/regularized-logit ensemble; covariate-shift
refit + attribute ranks; continuous higher-order interactions) have each come
back null or too small to distinguish from noise. Combined with the earlier
diagnostic (excellent calibration, no exploitable subgroup, xgboost can't
out-predict the logit), this is a strong, repeatedly-tested case that
`ensemble_v11` is at or very near the practical ceiling for this dataset.
With ~5 days left before the competition closes (2026-08-01), further effort
is better spent on the report than another modeling round unless a genuinely
new structural idea surfaces.

## Resolved: noise-floor check, post-hoc calibration, and seed-bagging (2026-07-27)
Deliberately re-audited whether "near the practical ceiling" was premature
convergence. First: the leaderboard gap to the reported leaders (1.187/1.190
vs. our 1.202, 0.012-0.015) is *smaller* than this project's own confirmed
public-LB noise floor (64.6% of same-model draw-pairs differ by >=0.015 from
sampling luck alone) -- the gap itself isn't strong evidence of a missing
lever. Then two more concrete techniques, run directly rather than via Codex:

- **Post-hoc temperature/shrinkage calibration against the target-weighted
  (shift-aware) loss: clean no.** Identity (no adjustment) is optimal on both
  the ordinary AND target-weighted loss; every deviation makes both worse.
  Closes off recalibration as a lever -- the model's confidence, not just its
  coefficients, is already close to optimal even under the shift-aware
  objective.
- **Seed-bagging xgboost (`R/xgb_seed_bagging.R`, 20 seeds x 5 canonical
  folds): a real effect that doesn't reach the ensemble.** Bagged xgboost
  alone is genuinely better (1.178668 -> 1.176836, bootstrap CI
  [0.000336, 0.003297], excludes zero -- confirmed real). But blended at the
  fixed 0.80/0.20 weight, the gain nearly vanishes (1.145094 -> 1.145087,
  CI centered on zero) because xgboost's small blend weight means its own
  noise was already mostly absorbed by the ensemble. The more promising
  version -- bagging the *dominant* mlogit component via bootstrap-resampled
  respondents -- wasn't attempted (real implementation risk around
  respondent-ID collisions under resampling) and is queued as a Codex
  follow-up.

**Net effect: unchanged, `ensemble_v11` remains best.** But this converts "we
stopped finding things" from possible premature convergence into "we checked
the two most obvious remaining technique classes (recalibration, bagging) and
both are genuinely exhausted," which is a stronger, more honestly-earned
claim. Full detail in `cleaning_log.md`, 2026-07-27.

## Resolved: final round -- mlogit bagging, partial pooling, SHAP interactions (2026-07-27)
Third and final Codex round of the day (branch `codex-bagging-pooling-shap`,
commit `ac6e844`, merged into `zhenhao`). Independently reviewed and
cross-checked every number against the raw generated CSVs -- exact match.

- **Bootstrap-bagging the dominant m8trpg component: genuinely hurts.**
  Unlike xgboost, bagging the 80%-weight mlogit component makes it *worse*
  (15-bag blend 1.145658 vs. ensemble_v11's 1.145094) -- every point on the
  1-15 bag learning curve is on the harmful side. A conditional-logit MLE is
  already a smooth, low-variance estimator; bootstrap resampling injects more
  noise than it removes. Bagging is not a universal remedy.
- **Partial pooling of segment slopes: independently confirms full pooling.**
  A completely different estimation method (penalized Cox-equivalent
  likelihood, ridge selected by nested CV) shrinks every new segment
  deviation to ~1e-40 -- genuinely zero. Agrees with the earlier fully-
  unpooled segment experiment via an independent method: no exploitable
  segment heterogeneity exists here at any magnitude.
- **SHAP-guided interaction discovery: finds real structure, still too
  weak.** The project's first data-driven (not hypothesis-driven) interaction
  search. Best candidate (mileage x urbanicity) blended: 1.144976 vs.
  1.145094 (+0.000118), bootstrap CI [-0.001228, 0.001431] -- the smallest,
  least confident positive estimate of the day.

**Not adopted; no submission made.** This closes the deepest single-day
modeling push of the project: six independently-verified experiments in this
final stretch, on top of three earlier full Codex rounds. Every genuinely new
mechanism -- model diversity, shift correction, rank features, higher-order
interactions (both hypothesis- and data-driven), recalibration, and bagging
in both directions -- has been tested to the same standard, and none clears
the bar. As exhaustive a search as the remaining time reasonably allows.

## Resolved: MLP ensemble diversity -- new best model, confirmed on public LB (2026-07-27/28)
One more round after the above (branch `codex-history-rrm-mlp`, commit
`f7bc47c`, merged into `zhenhao`) tested three genuinely different angles:
design-exposure history, Random Regret Minimization, and a non-tree neural
ensemble member. Independently reviewed and cross-checked every number
against the raw generated CSVs -- exact match throughout, including a
gradient-check verification of RRM's custom likelihood (<5e-10 discrepancy
vs. finite differences).

- **Design-exposure history**: real but too small after multiple-testing
  correction (99% CI and Bonferroni-adjusted CI both cross zero). Notably,
  the original idea (actual choice history) had a fatal flaw caught before
  implementation -- a test respondent's full 19-task sequence is unlabeled
  simultaneously, so a feature built from observed past choices is
  uncomputable at test time even though it would look fine in CV. Corrected
  to use only observable design/exposure sequence instead.
- **RRM**: promising screen, null-to-negative honest CV. A respectable
  standalone competitor to m8trpg but no transferable ensemble diversity.
- **MLP (nnet, single hidden layer, since keras/tensorflow/torch weren't
  available): the first result all day to clear the ordinary bootstrap-CI
  bar.** Fold-cross-fitted blend with the fixed ensemble_v11: **1.143789 vs.
  1.145094 (gain 0.001305)**. High-precision 100,000-replicate bootstrap:
  **95% CI [+0.000092, +0.002513] -- excludes zero**, 98.24% win rate. Does
  NOT survive stricter correction (99% CI and a 9-configuration
  Bonferroni-adjusted CI both cross zero, since 9 architectures were
  screened). An extended 5-family ensemble reached 1.143129 but wasn't
  significantly better than the simple 2-way blend -- not worth the extra
  complexity.

**Structurally the safest candidate produced all session**: correlates 0.9965
with ensemble_v11's predictions (vs. 0.987 for the triple-interaction
candidate), max deviation 0.07 (vs. 0.61), zero test rows with any >0.15
swing -- a bounded softmax output doesn't have the unbounded-product
outlier-sensitivity risk that affects the triple-interaction candidate.
`submission_codex_mlp_v12_candidate.csv` (full-data fit, 5 seeds, 15% weight
-- matching the mean of the honestly cross-fitted fold weights -- against
the exact already-public-scored ensemble_v11 CSV) was submitted 2026-07-28:
**public 1.201, beating ensemble_v11's 1.202 -- the first public-LB
improvement of the project since ensemble_v11 became the standing best.
This is now the officially adopted model** (CV 1.143789, public 1.201, gap
0.0572 -- in line with the established ensemble-class gap pattern, no new
anomaly). The observed public gain (0.001) was smaller than the CV point
estimate (0.0013) but in the same direction, not a reversal -- consistent
with a genuine, if modest, effect. `submission_triple_income_miles.csv` and
`submission_ensemble_v12_4way.csv` remain queued for future submission
slots, both still testing open questions independent of this result.

**2026-07-28 follow-up (branch `codex-mlp-seed-bagging`, commit `066ac0b`):**
tested whether averaging the MLP over more seeds (5 -> 20) improves it, since
it plays the same minority-weight role that benefited from seed-bagging
earlier (xgboost). Result: the component improves a lot (1.190543 -> 1.168530)
and the point estimate improves too (gain 0.001305 -> 0.001956), but the
bootstrap interval WIDENS instead of tightening (95% CI width 0.002421 ->
0.003339), and the direct 20-vs-5-seed comparison CI crosses zero
([-0.000122, +0.001416]). Verified mechanism: more weight naturally flows to
the improving MLP under fold-cross-fitted selection (0.13-0.17 -> 0.20-0.26),
which amplifies both its benefit and its respondent-level variance
contribution -- a real trade-off, not a bug. **No new candidate generated;
the original 5-seed `submission_codex_mlp_v12_candidate.csv` remains the
recommended submission** -- more seeds do not make it clearly better.

## Resolved: triple interaction + MLP combined -- submitted, worse public score (2026-07-28/29)
Explicit push toward clearing public 1.2 (other teams reportedly at 1.186).
The triple-interaction mlogit and the MLP were each validated independently
but never blended together -- tested directly (own analysis, using already-
cached OOF predictions, no refitting needed for the CV comparison).

Honest fold-cross-fitted 3-way blend: **1.143328** -- ties the session's
best-ever CV number (the 5-family ensemble's 1.143129). Vs. the current best
(v11+MLP, 1.143789): gain +0.000462, CI **[-0.000754, +0.001642] -- crosses
zero**, not confirmed better than what's deployed. Vs. plain v11: gain
+0.001767, CI **[+0.000054, +0.003420] -- excludes zero**, barely. A paired
public-LB-sized simulation, anchored to the current best's real public score
(1.201), implied a 95% range of **[1.1979, 1.2033]** with a **63.6% chance**
of beating the current best on the same draw -- a real, if not overwhelming,
lean toward clearing 1.2. Same known caveat as the standalone triple
candidate: sensitive to the same extreme-income test respondent (48 of 4997
rows show a >0.15 swing, max deviation 0.52) -- test has proportionally more
such extreme respondents than training, so real-world variance could exceed
the simulation's estimate.

`submission_triple_mlp_v13.csv` (reuses the already-fit MLP full-data test
predictions, fits mlogit+triple and xgboost fresh, blend weights
0.732/0.112/0.156) was submitted 2026-07-29 as the best-available honest bet
once repeated CV (see below) closed off every other candidate. **Result:
public 1.210 -- worse than the current best (1.201), and outside the paired
simulation's own predicted 95% range by 0.0067.** Gap vs. its CV (0.066672)
is now the largest in the project. Consistent with the flagged extrapolation
risk: the unbounded income x mileage term is most sensitive to a handful of
extreme-income respondents, and test has proportionally more of them than
the training-respondent-based simulation could represent. **Not promoted;
the current best is unchanged.** See "Known weakness: CV-to-public gap"
above and the 2026-07-29 cleaning-log entry for the full comparison against
the earlier m8trpg-alone submission, which showed the same pattern.

## Resolved: deep learning, full stacking, LightGBM -- best-ever CV, still can't reach 1.186 (2026-07-28)
Explicit push for public 1.186, needed for a good module grade (branch
`codex-deep-stack-boost`, commit `8510b01`, merged into `zhenhao`).
Independently reviewed all four scripts and cross-checked every number
against the raw generated CSVs -- exact match, including verified-leakage-free
nested log-pool selection, a correct analytic softmax-gradient derivation for
the arithmetic blend, and confirmed architecture-freezing-before-CV
discipline for the deep MLP.

- **R `torch` installs successfully** (previously assumed unavailable -- an
  environment-setup gap, not a permanent limit). A real 2-layer (128/64
  unit) dropout MLP contains real signal (replacing the shallow MLP in v11
  clears zero vs. plain v11), but does **not** clear the bar that matters --
  improving over the already-submitted shallow-MLP candidate (CI crosses
  zero for both the incremental and joint comparisons).
- **Full nested stacking across 6-8 diverse components, retested with much
  more diversity than the first attempt: learned combiners still lose to
  simple arithmetic averaging.** A nested ridge log-pool (properly
  leakage-free, boundary-checked) reaches at best +0.000634 (CI crossing
  zero); a shallow xgboost meta-model is decisively harmful (~1.152). Same
  conclusion as the very first stacking attempt (2 components), now
  confirmed with far more diversity: this ensemble has reached what a simple
  weighted average can extract.
- **Best CV of the entire project: 1.142112** (8-component arithmetic
  blend), improving all 5 folds. Bootstrap vs. current best: ordinary 95%
  CI **[0.0000466, 0.0032790] -- excludes zero, but only just**; 99% and
  Bonferroni-adjusted CIs both cross zero. Correctly not submitted.
- **LightGBM**: clean negative, all 3 configs got zero screen weight, did
  not proceed to CV.

**The number that matters most for the 1.186 target**: even taking the best
(unconfirmed) result at full face value, the implied public-score movement
is from 1.201 to roughly **1.199-1.200 -- not 1.186**. Across this entire
project's search (ensembling, shift correction, rank features, higher-order
interactions, bagging, RRM, deep learning, learned stacking, LightGBM), no
single gain has exceeded ~0.002 in CV terms. Closing a 0.015 public gap
would need roughly 10x any single improvement found anywhere in this
search. Strong evidence that 1.186 is not reachable via further iteration
on the techniques already tried, though it doesn't rule out a fundamentally
different approach.

**2026-07-28 follow-up (branch `codex-catboost`, commit `12676eb`):** tested
whether CatBoost's native ordered-boosting categorical mechanism recovers
anything LightGBM's split-based categorical treatment missed. Installed via
CatBoost's official prebuilt Windows binary (no Rtools on this machine).
Clean, unambiguous negative: fold-cross-fitted blend weight was exactly
zero in all 5 CV folds (component alone 1.203659799, worse than every other
tree-based component tried); the resulting blend is byte-identical to the
current best. Two independent tree frameworks with genuinely different
categorical-handling mechanisms now agree on the same 1.142-1.144 ceiling --
changing how the tree learner treats categoricals is not the missing lever.

## Resolved: shared-utility exact-softmax models, scale heterogeneity, yearind (2026-07-28)
Five directions from an external technical review, fact-checked before
delegating (branch `codex-shared-utility`, commit `3e1ca29`, merged into
`zhenhao`).

- **A genuine shared-alternative-utility model on the exact 4-way softmax
  loss** (custom xgboost objective, gradient-verified to 1.06e-10): closes a
  real gap -- neither `multi:softprob` (no exchangeability across
  alternatives) nor `rank:ndcg` (shared function, but a ranking loss, not
  cross-entropy) actually optimizes the choice likelihood directly.
  Cold-start, it failed even the screen (1.218569 vs. v11's 1.160568) --
  confirms the loss function wasn't the bottleneck, m8trpg's hand-built
  features were always doing the real work.
- **The same objective as a residual on m8trpg**: small real effect
  (+0.0000759 propagated), CI [-0.0000839, +0.0002417] crosses zero --
  dominated by `price_gap` features, a tiny refinement of an already-known
  mechanism.
- **Global (whole-utility) scale heterogeneity**: decisively harmful, CI
  entirely below zero at every tested ridge strength.
- **`yearind` interactions**: harmful, not adopted -- closes the one
  genuinely untested covariate with a clean negative answer.
- **Test-like-respondent re-ranking**: the 8-component blend and triple+MLP
  keep the same ranking at every population cutoff; on the top-30%-most-
  test-like slice the 8-component blend's edge actually excludes zero
  ([+0.000262, +0.007557]) -- corroborating, but doesn't repair its
  already-failed stricter interval.

## Confirmed structural fact: ~299 recurring questionnaire versions (2026-07-28)
The external review also proposed a specific, checkable hypothesis: a fixed
pool of ~300 questionnaire "versions" (Sawtooth CBC's documented default),
assigned by sequential Case-number cycling. Tested directly (own analysis,
`R/check_questionnaire_version_structure.R`, before sending anything to
Codex) by fingerprinting every respondent's entire ordered 19-task sequence
(design only, no choices) across all 1398 train+test respondents.

**The sequential-cycling mechanism is refuted** (tested every candidate
period 50-500; purity is exactly 0 at all of them). **But the underlying
structure is real and confirmed**: exactly **299 unique full-sequence
fingerprints** among 1398 respondents, matching Sawtooth's 300-version
default almost exactly. 286 of 299 recur (~4.7 respondents/version on
average), train and test mixed into the same version groups, with no
relationship between Case-number gaps and version sharing (consistent with
random, not sequential, assignment).

This is materially stronger than the existing `design_cell_empirical_shrinkage`
null (which grouped by individual task-position designs, ~3.8 respondents/cell,
19 separate small-sample problems). Grouping by the full sequence instead
means the same ~4-5 respondents share **all 19** tasks, pooling ~80-95 data
points per version instead of ~3-4 -- a materially different, better-powered
version of the same idea. This is now the single most promising untested
lever, since it's the only hypothesis from the review round that was
independently confirmed to exist in the data before any modeling was built
on top of it. Cached as `data_processed/questionnaire_fingerprints.rds`.

**2026-07-28 follow-up -- version correction tested and rejected (branch
`codex-overnight-queue`, commit `798662c`):** implemented as a one-step
Newton correction to the opt-out utility per version (utility-space, not a
naive probability residual), with a verified leakage-free nested
inner-OOF-before-outer-refit structure. Rejected: negative gain on two
independent evaluations (-0.000172 official OOF, -0.000169 fresh refit).
Diagnostics directly confirm why -- versions average only ~3-5 training
respondents, many folds have respondents with literally zero same-version
peers, and single-peer versions are 100% dominated by that one respondent.
**The confirmed 299-version structure is real, but too sparse per version to
support even one well-shrunk correction parameter.** Closes this lever.

## Resolved: overnight queue -- one strong unconfirmed lead, a genuinely pre-registered search (2026-07-28)
Two more results from the same overnight round, branch `codex-overnight-queue`,
commit `798662c`, merged into `zhenhao`.

- **Prior-smoothed design-history features (most interesting unconfirmed
  lead of the round):** a fold-fitted population-prior initialization (verified
  leakage-free) instead of zeroing Task 1. Best candidate: 1.143686618 ->
  1.142951450 (+0.000735), improving 4/5 folds. Ordinary 95% CI
  [-0.000090, +0.001564] -- close, doesn't exclude zero. A cumulative
  13-candidate Bonferroni check (this round's 5 plus the original round's 8)
  widens to [-0.000483, +0.001960]. The price-history coefficient is
  negative and stable across all 5 folds (behaviorally coherent -- reference-
  price anchoring); the attribute-familiarity coefficient's sign flips.
  **Not adopted, but the closest near-miss of any lead this session.**
- **A genuinely pre-registered 24-config deep-MLP search:** the full
  registry and Bonferroni-24 decision rule were committed to git *before*
  the screening run started -- independently verified via commit and file
  timestamps, not just claimed. This is the first search this session where
  the multiplicity correction was fixed in advance rather than applied
  after seeing results. The frozen winner (256-128-64 layout) reached
  1.143321960 (+0.000365, 4/5 folds), but its pre-declared Bonferroni bound
  (-0.000857) automatically rejects it -- no judgment call needed. Seed-
  bagging was correctly gated (and correctly skipped, since the ordinary
  lower bound also failed), avoiding a repeat of the shallow MLP's
  point-estimate-improves-but-interval-widens pattern.

**Net effect: no submission from this round.** Version correction is now
closed definitively; the prior-smoothed history lead remains open (worth
revisiting if further evidence accumulates); the pre-registration discipline
worked exactly as designed.

## Resolved: repeated CV puts both near-misses to a harder test -- neither survives (2026-07-29)
Pre-registered (branch `codex-repeat-cv`, commit `64d9c0e`, merged into
`zhenhao`; pre-registration commit `b452966` independently verified via git
timestamp to predate all results) re-test of the two closest overnight
near-misses against 5 additional genuine respondent-grouped five-fold refits
(seeds 1907/2719/6151/8293/104729, alongside canonical 4821) -- every
component refit from scratch per fold, verified directly in
`R/codex_repeated_cv.R` (held-out respondents genuinely excluded from
`source_wide`/`source_long` before refitting; fold-cross-fitted blend
weights). Promotion required gain>0, family-adjusted lower bound>0, and
>=5/6 positive repeats -- fixed before results were seen.

- **Prior-smoothed history features**: mean gain across repeats **+0.000574**,
  ordinary 95% CI **[-0.000239, +0.001381]** -- still crosses zero, slightly
  worse than the single-split estimate.
- **Eight-component arithmetic blend**: mean gain **+0.001169**, ordinary 95%
  CI **[-0.000199, +0.002513]**. This is the more important result -- the
  canonical single-split CI had barely *excluded* zero
  ([0.0000466, 0.0032790]); repeated CV pulled the pooled estimate back
  across zero. Exactly what repeated CV is for: distinguishing a real gain
  from a favorable fold-assignment draw.
- Both candidates improved in **6/6 repeats** (24/30 and 22/30 individual
  folds respectively) -- consistent direction, but pooled respondent-level
  uncertainty still dominates.
- **New reason to avoid the eight-component blend regardless**: independently
  re-audited `submission_codex_8component_candidate.csv` (valid file, never
  submitted) and found severe extrapolation on the same known extreme-income
  respondent (`No=22637`) flagged for the standalone triple-interaction
  candidate -- max probability change 0.362630, driven by the
  triple-interaction mlogit's 47.9% blend weight (differs by up to 0.683
  alone on that row).
- The one pre-registered joint history+triple-interaction mlogit follow-up
  failed its own single-split screen (1.157992 vs. plain triple's 1.157576)
  and correctly did not proceed to CV.
- Respondent-bootstrap bagging of the whole m8trpg model was correctly not
  re-run -- already a completed, logged negative (2026-07-27: 1.145094 ->
  1.145658, every point harmful).

Independently re-verified before merging: re-ran `R/codex_repeat_cv_audit.R`
myself (not just read the write-up) -- it recomputes log loss and
per-respondent gains directly from raw saved probability matrices, rebuilds
the bootstrap summary from the raw 100,000-replicate draws, and
reconstructs the submission CSV byte-for-byte from its 8 weighted
components; every number matched exactly. Also read the fold-construction
code directly to confirm genuine per-fold respondent exclusion.

**No Kaggle submission was made.** This closes the eight-component-blend
question with a second, more rigorous negative -- it looked like the
project's single best lead on one split, and isn't once measured more
carefully. `submission_triple_mlp_v13.csv` (CV 1.143328, its own gain vs.
current best also crosses zero: CI [-0.000754, +0.001642]) was **not**
re-tested under repeated CV and remains the best-CV unsubmitted candidate
with a specific stacking rationale, still queued for a submission slot.
Prior-smoothed history is the only lead not yet definitively rejected,
though repeated CV has made its case measurably weaker too. At this point,
no untested candidate in the project has a respondent-bootstrap CI that
cleanly excludes zero against the current best -- the search has reached
the point where further iteration on already-tried techniques is unlikely
to move the needle further.

## Resolved: price-only history isolation -- closes the last open lead (2026-07-29)
The one remaining unrejected lead (prior-smoothed design history, `both_k3`)
bundled a stable, negative-across-folds price-anchoring term with an
attribute-familiarity term whose sign flipped across folds. Isolating the
price term alone -- pre-registered (`codex_price_history_preregister.md`,
committed before any fit) at the same k in {3,9,27} grid, straight to full
five-fold CV plus the same five repeated-CV seeds already used for `both_k3`,
no single-split screen gate -- is a legitimate, materially different
follow-up, not a re-run.

Before running anything: directly checked the raw CSVs and confirmed this
dataset is genuinely **partial-profile** (exactly 9 of 19 attributes active
per alternative in all 21,565 rows, alt 4 always all-zero) -- matching this
file's own finding #2, not an external assumption of full-profile design.
Doesn't reopen a lever (active-attribute count never varies; which 9 are
active is already absorbed by the existing `factor(attribute)` terms and the
design-cell/299-version fingerprints).

Canonical CV: all three candidates (1.142918755 / 1.142911201 / 1.142956136,
gains +0.000768/+0.000775/+0.000730, 4/5 folds) match or slightly beat
`both_k3`'s own canonical gain. Repeated CV (6 fold assignments,
100,000-replicate bootstrap): point gains +0.000602/+0.000615/+0.000592, all
95% CIs crossing zero (e.g. `price_only_k9` [-0.000252,+0.001476]), 6/6
repeats positive, and the price coefficient negative in **all 30 of 30**
fold fits for every candidate (full sign stability). Per the pre-registered
rule, all three pass 3 of 4 criteria and fail only the family-adjusted lower
bound -- **none promoted.**

Dropping the noisy attribute term neither unlocked hidden signal it had been
masking nor cost anything -- point estimates and stability are close to
identical to the bundled version's. This is a cleaner null than before: it
rules out the specific hypothesis that the attribute term was suppressing
the price term's confirmability. The effect is real and directionally
coherent but its size (~0.0006) sits inside the same respondent-level noise
floor (bootstrap SD ~0.0004-0.0005) that has closed every other near-miss
this session. **Not adopted; no submission made.** Full detail in
`codex_price_history_findings.md`. The one remaining materially-different
angle -- a version borrowing strength from *other*, similar versions rather
than the single global population prior already used -- was not attempted.

## Resolved: shared-alternative-utility MLP -- screens well, fails CV by 8x (2026-07-29)
Filled the one untested cell of a 2x2 the brief asked about: shared-
alternative-utility constraint (already tried, tree function class, cold-
start failure) x neural function class (already adopted, but never with the
shared-weight/exchangeable-alternative constraint -- the existing shallow/
deep MLP concatenate all 4 alternatives into one flat row per task). Built a
weight-shared torch MLP applied identically to each alternative's own
feature row (opt-out included, its all-zero profile is just another valid
input), trained on the exact 4-way cross-entropy via task-grouped batching.
Feature treatment mirrors m8trpg's own established choices (one-hot
attributes/price/segment/region/ppark, standardized continuous covariates,
existing price-gap/rank context) for a fair comparison.

Two real bugs caught via smoke-testing before the full run: an off-by-4
assertion (task rows vs. row count) and checkpointing a raw torch
`state_dict` across process restarts (invalid external pointer on reload) --
fixed to checkpoint only the plain prediction matrix, matching
`R/codex_torch_deep_mlp.R`'s existing safe pattern.

Screen (single split): all 3 architectures give a positive incremental gain,
best (`shared_64_32`) +0.001724 -- comparable to previously-promoted
screens, despite the component alone being far weaker than m8trpg. Pre-
registered the frozen winner and a stopping rule (only escalate to repeated
CV if canonical CI excludes zero) before running CV. **Canonical five-fold
CV: gain drops 8x to +0.000143**, respondent-bootstrap 95% CI
[-0.000567, +0.000850], win rate 65.5% -- crosses zero comfortably, unlike
the price-history near-miss. Per the pre-registered rule, repeated CV was
correctly not run. **Not adopted; no submission made.** Closes the shared-
utility-objective direction: two different function classes (tree, neural)
now agree the exchangeability constraint and exact likelihood were never
the missing lever -- m8trpg's hand-built interaction structure is what does
the real work. Full detail in `codex_shared_utility_mlp_findings.md`.

## Resolved: four parallel experiments, all null -- search remains exhausted (2026-07-29)
After the report was rendered, four genuinely new hypotheses were dispatched
as parallel background agents, each in its own isolated git worktree, each
pre-registering before running and explicitly told which already-rejected
result not to repeat. All four finished and were independently re-verified
from raw cached artifacts (not just their write-ups) before logging.

- **Smooth splines** (age/mileage/income Price/inside interactions,
  materially different from the rejected binned version): only `miles_df3`
  passed screening; canonical CV looked promising (+0.000709, 84.6% win rate)
  but crossed zero, triggering repeated CV (the same escalation rule that
  caught the eight-component blend); pooled repeated-CV gain +0.000352, CI
  still crosses zero. Coefficients stayed dense/stable across all 30 fits --
  genuinely avoids the rejected version's quasi-separation failure, just
  doesn't clear the bar.
- **Transductive test-covariate adaptation** (quantile-mapped moment matching
  + confident self-training, both mechanistically different from the two
  already-rejected reweighting-based shift corrections): both null.
  Quantile-matching's isolated mlogit effect was real (+0.000172) but diluted
  below the noise floor by ensemble blend weights; self-training only had
  0.4-0.6% of test tasks confident enough to matter.
- **Version-pool (neighbor-smoothed) opt-out correction** (lets a
  questionnaire version borrow Newton gradient/curvature mass from
  design-similar versions, rather than the rejected per-version-isolated
  estimate): null (gain -0.0000231, CI crossing zero), but with a genuine
  mechanistic diagnosis -- pooling makes the exact 0/1-peer respondents it
  targets *worse* (-0.0013, -0.0009), direct evidence design-marginal
  similarity between CBC versions doesn't carry transferable signal.
- **Bayesian hierarchical mixed logit** (`bayesm::rhierMnlRwMixture` MCMC with
  explicit priors, since `rstan`/`brms` needed an unavailable C++ toolchain --
  disclosed before running): re-tests the already-rejected frequentist mixed
  logit under a completely different estimation philosophy. Population-level
  posterior-predictive scoring for held-out respondents was smoke-tested
  against a deliberately-wrong comparison first (confirmed a large,
  unambiguous difference) to rule out silent leakage. Result: not just null
  but a small, fairly confident **harm** (gain -0.0006138, CI [-0.0012461,
  +0.0000324], 3.16% win rate) -- confirms the mixed-logit rejection isn't a
  frequentist-estimation artifact.

**Two process notes.** (1) The parallel-worktree dispatch did not reliably
branch every worker from `zhenhao`'s tip -- two of four (version-pool,
Bayesian mixed logit) were rooted in a stale `main` snapshot from
2026-07-26, missing nearly this whole session's `AGENTS.md`. Caught via
`git merge-base`, not assumed. Did not appear to compromise either result
(each brief already contained the specific relevant rejected finding
verbatim, and every number was independently reproduced from raw data
regardless), but worth fixing before relying on this pattern again --
worktrees should be explicitly checked out from `zhenhao`, or have the
current three canonical files copied in, before a worker starts. (2) The
Bayesian mixed-logit worker hit the platform's monthly API spend limit after
finishing its analysis but before committing/pushing; nothing was lost (full
computation and write-up were on disk), and the coordinating session
independently verified and committed/pushed on its behalf.

**No submission made; no change to the standing best.** This is now the
third independent search effort this session (this session's own history,
a separate adversarially-instructed modelling-lead session, and this
four-way parallel dispatch) to conclude, via genuinely new hypotheses each
time, that nothing clears the promotion bar. `mlp_ensemble_v12_candidate`
(1.143789 CV / 1.201 public) remains the final recommendation.

## Resolved: three more genuinely new experiments, two near-misses + one clean reject (2026-07-29)
A fourth wave of new hypotheses arrived directly as R scripts (not a git
branch); three had already fully run before review, a fourth (SVM
ensemble-diversity) was run live by the user in RStudio and is logged
separately. All three reviewed here were independently re-verified from raw
cached artifacts (log loss and per-respondent gain recomputed directly from
saved prediction matrices, bootstrap rerun with fresh seeds) before logging.

- **Choice-set geometry/crowding features** (each inside alternative's
  pairwise attribute/price similarity to its two choice-set competitors,
  decomposed into set-mean/own-centered/nearest-excess -- different from the
  existing price-only rank/gap terms): canonical CV +0.000110667 (near miss),
  repeated CV pooled +0.000102185, still crosses zero, 6/6 repeats positive.
- **Component-wise exact-softmax residual boosting** (stagewise additive
  boosting over m8trpg's fitted-utility offset, letting regularized
  selection pick terms rather than testing a hand-picked set): canonical CV
  +0.0000649 (barely a near miss), repeated CV pooled +0.0000382, still
  crosses zero -- the smallest confirmed-consistent-direction near-miss of
  the session, an order of magnitude below anything that has ever cleared
  the bar.
- **Price-curve curvature shrinkage** (tests whether the saturated 12-level
  price factor is overfit and would benefit from smoothing): **clean,
  decisive reject** (CV -0.001901387, CI entirely below zero) -- confirms the
  existing saturated price treatment (Section 2.1.1 of the report) is not
  overfit, reinforcing rather than undermining that part of the model.

Same qualitative pattern as most of this session's near-misses: a small,
consistently-signed positive effect that doesn't survive repeated-CV's added
fold-assignment variance. **No submission made; no change to the standing
best.**

## Resolved: calibrated SVM as ensemble diversity -- new function class, still not competitive (2026-07-30)
The fourth new script from the same wave (run live by the user in RStudio,
~4 hours), tests a calibrated RBF-kernel SVM (e1071/libsvm) as ensemble
diversity -- a genuinely new function class never tried in this project.
Independently re-verified from raw cached prediction matrices before
logging; fold-construction code read directly and confirmed leakage-free
(outer-fold exclusion and the nested inner hyperparameter search both scoped
correctly).

**Rejected, and not competitive even as a weak diversity source.** Canonical
CV gain -0.000168231, 95% CI [-0.000443,+0.000105] -- correctly not
escalated (neither a pass nor a near miss). Component alone: 1.166470 CV,
weaker than every other diversity member tried (xgboost 1.178668, shallow
MLP 1.190543) -- but unlike those two, blending it in makes the ensemble
*worse*. Closes the "different function class" diversity question: being
different isn't sufficient by itself -- xgboost and the MLP each still add
real, confirmed diversity despite being individually weak; the SVM does not.

**Unrelated process note.** An iCloud Drive sync conflict briefly renamed
`AGENTS.md`/`submissions_log.csv` to `AGENTS 2.md`/`submissions_log 2.csv`
in the local working directory during this round -- git itself was
unaffected (both confirmed byte-identical to the last commit before being
restored). A few empty, oddly-named stray directories were also found
nearby and left in place (harmless, not touched without being asked).

**No submission made; no change to the standing best.**
`mlp_ensemble_v12_candidate` (1.143789 CV / 1.201 public) remains the
recommendation.

## Resolved: set-context network becomes the new best model -- CV-predicted improvement confirmed a second time (2026-07-30)
A feed-forward network where each alternative's features include
permutation-invariant summaries of the *other* alternatives in its own
choice task (the same choice-set-context idea behind the price-rank/gap
terms and the choice-set-geometry experiment, but learned end-to-end
instead of hand-built). Canonical single-split CV was a wide, unconfident
near miss (gain +0.000153, win rate only 63.8%), auto-escalating per its
pre-registered rule to 6-seed repeated CV -- where, unlike every other
near-miss this session, the pooled signal came in **stronger**, not weaker:
pooled gain +0.0011636, ordinary 95% bootstrap CI **[+0.0000106,
+0.0023180] -- excludes zero**, win rate 97.6%, all 6 individual repeat
seeds positive (0.00015-0.0017 each). The margin is thin -- it does not
survive a 99% CI -- but that is the identical standard the original MLP
candidate was promoted under, not a new exception. Component alone is weak
(1.261810 CV), the same individually-weak-but-genuinely-diverse pattern as
xgboost and the shallow MLP. Frozen blend weight: 11.1% (mean of the six
repeats' cross-fitted fold weights).

Before recommending submission, the full-data build (`R/codex_set_context_
candidate_submission.R`) enforced real, checkable safeguards rather than
asserted ones: MD5-locks the runner code and the actual publicly-scored
baseline submission (refuses to run against anything else); hard-checks the
saved repeated-CV verdict genuinely says `promote=TRUE`/`lower_95>0` before
proceeding; and requires the full 3-seed network to be trained twice
independently, refusing to write a candidate CSV unless the two runs agree
to within 1e-6 -- they agreed exactly (difference 0.0 on both the component
and the final blend). Independently re-verified rather than trusted:
recomputed log loss and per-respondent gain directly from the raw
canonical/repeated-CV result files; read the fold-construction code and
confirmed the same leakage-safety pattern used throughout this project
(hard assertion of zero respondent overlap between fitting and validation,
scalers fit strictly on training-fold data); independently recomputed every
full-data-build audit statistic (max change, argmax-flip rate, correlation,
mean absolute change) directly from the actual submission and baseline
files, and confirmed the submission's MD5 on disk -- everything matched the
build script's own report exactly.

**Public result: 1.200**, beating the prior best (1.201). This is the
**second** time this project's CV-predicted improvement direction has been
confirmed on the real public leaderboard (the first was the original MLP
candidate) -- real evidence the respondent-grouped CV methodology tracks
something genuine about the public split, not just internal consistency.
Gap to CV (0.056467) sits squarely inside the established 0.052-0.057
ensemble-class pattern, unlike the two flexible-model submissions that broke
it (m8trpg-alone 0.065979, triple_mlp_v13 0.066700) -- a third independent
data point for the same conclusion.

**`set_context_utility_network_v14` (1.143533 CV / 1.200 public) is now the
current best model**, replacing `mlp_ensemble_v12_candidate` (1.143789 CV /
1.201 public), which is retained as the prior-best fallback reference.

## Resolved: sequence-transformer rescue closes definitively; four more experiments rejected (2026-07-31)
The set-pooling correction network's canonical near-miss showed every
seed/fold converging to a near-zero correction (~1e-6 vs a 0.35 bound) --
strong evidence two stacked regularizers (`weight_decay=0.002`,
`correction_penalty=0.01`) were pinning it at zero-init. A rescue suite
swept 6 configs across init scale and regularization strength to test this.
Its smoke test first caught a real bug (fixed): the checkpoint-reload path
re-normalizes via `validate_probability()` while the fresh-fit path
doesn't, so a harmless floating-point artifact was failing an `identical()`
check that should have used the same 1e-6 tolerance as every other
reproducibility gate here. Fixed, verified (cleared the stale checkpoint,
reran, passed cleanly with a real ~1000x larger correction on the
regularizer-free capacity check).

The full run then completed in **under 15 minutes**, not the ~4.5-5 hour
worst case, with a cleaner reject than before: the fold-1 screen found a
stark dichotomy -- the 3 configs with regularizers relaxed enough to move
the correction substantially (RMS ~0.33) all showed it **hurting**
predictions (-0.035 to -0.057 vs the frozen offset); the 3 that stayed
regularized were harmless no-ops. No config was both non-trivial and
helpful, so all six were correctly rejected on fold 1 alone without
touching folds 2-5 -- discipline saving most of a night's compute once the
evidence was unambiguous. Closes this direction for good: relaxing the
regularizers reveals overfitting, not hidden signal.

Four more experiments independently verified and logged: a low-rank
demographic-partworth factorization (near-miss that reverses sign under
repeated CV); a two-head opt-out/bundle ensemble (closest near-miss of the
wave, 89.6% win rate, independently reconstructed and confirmed genuine,
still crosses zero); a "safe" gated diversity recombination (crosses zero,
75% weight stuck on v14 itself); and an OOF residual audit (diagnostic, not
a candidate) finding neither the opt-out nor bundle head of the two-head
model satisfies its own promotion rule -- concluding "STOP MODEL SEARCH...
redirect effort to the final report" independently.

**No submission made; no change to the standing best.**
`set_context_utility_network_v14` (1.143533 CV / 1.200 public) remains
final, now with an even more exhaustively closed search behind it.

## Team / git state
- Working branch: `zhenhao` (this repo's primary author, GitHub `froyosng`).
  Team: Imelda Lee, Woon Zee Ning ("Zeening"), Clarence Elvareta (she/her),
  Sng Zhenhao.
- `main` was reconciled 2026-07-25/26 via PR: brings in `zhenhao`'s full
  progression + Imelda's models (her best: `mlogit_v3_combined`-family at
  1.202-1.220, all behind the team's own mlogit line -- see submissions_log.csv
  for the individual experiment results, both hers and the negative results her
  own testing found). Branch protection now enabled on `main` (PR + 1 approval
  required, no direct pushes).
- Clarence's `eda_clarence`/`xgboost.R` deliberately NOT yet merged into `main`:
  her validation split is Task-based (<=12 vs >12), not respondent-grouped,
  which leaks a respondent's tasks across train/test -- her reported number
  isn't trustworthy until fixed. Also has a hardcoded machine-specific `setwd()`.
  Submitted 2026-07-27: public **1.221** -- since no internal log-loss number
  was ever captured, this is the first trustworthy signal on this model at
  all. Worse than ensemble_v11 (1.202) and the standalone mlogit (1.213), but
  meaningfully better than Zeening's random forest (1.259). Doesn't change the
  current-best recommendation; the model itself is reasonably competent even
  though its internal validation methodology still isn't fixed.
- Zeening's random forest (`rf_gridsearch`, ranger, grid-searched mtry/
  min.node.size) submitted: public 1.259. Her internal CV (1.162) is unreliable
  for the same reason as Clarence's -- her split is fully row-level random
  (`sample(1:nrow(...))`), not respondent-grouped, so nearly every respondent's
  tasks are scattered across both her train and validation sets. Gap (0.097) is
  the largest of any model in the project, empirically confirming the concern.
  Her K-means "persona" clustering idea was re-tested as a logit heterogeneity
  axis (see negative results above) -- didn't transfer, but was a legitimate
  idea worth checking.
- Imelda's `mnl+xgb` (branch `imelda`, `notebooks/experiments/ensemble_mnl_xgb.Rmd`)
  submitted 2026-07-28: public **1.255**. Unlike Clarence's/Zeening's, her
  internal validation IS respondent-grouped (20 splits sampling unique `Case`)
  and honestly gave 1.18616 -- yet the gap (0.06884) is the LARGEST of any
  properly-validated model in the project, bigger than this project's own
  standalone-mlogit gap (0.066). Plausible (not confirmed) structural reasons:
  her formula has zero respondent-covariate interactions at all (this
  project's single biggest source of legitimate gains), only 6 of 19
  attributes are factor-coded (rest linear/continuous), and her xgboost uses
  the same binary-renormalization architecture this project found null. A
  genuinely useful independent data point for the report's generalization-gap
  discussion: sound validation methodology alone doesn't guarantee a small
  gap on this dataset.
- `competition_report.qmd` was rewritten by Codex on branch `report-rewrite`
  (commit `cb42fe7`) and merged into `zhenhao` (commit `95ce340`); it now covers
  ensemble_v11 as best model, the identification findings, and the
  public-vs-private gap discussion. ~20 specific numbers fact-checked against
  the actual logs -- all matched. Still needs a PDF render (no Quarto/TeX in
  this environment; use RStudio/Positron's bundled Quarto, or install
  Quarto+TinyTeX here) and a wording/layout pass before submission.
- Codex's modeling push (`codex-modeling`, commit `b37ecc8`) is also merged into
  `zhenhao` -- see the section above. Not adopted as the new best model, but the
  scripts and write-up are kept for reproducibility.

## Next steps
1. **`competition_report.qmd` needs another headline update**: best model is
   now `set_context_utility_network_v14` (1.143533 CV / 1.200 public),
   superseding the `ensemble_v11 + MLP` writeup from 2026-07-29. Needs: the
   new formula/weight (0.889 x prior best + 0.111 x set-context network), a
   description of the set-context architecture, the repeated-CV-strengthens-
   not-weakens result (unusual, worth highlighting), the full-data build's
   two-run reproducibility gate, and the updated CV-to-public gap table/
   discussion (this is now the SECOND CV-predicted-direction confirmation on
   the public LB, strengthening that argument). Re-render to PDF (Quarto +
   TinyTeX now installed in this environment from the prior render -- see
   `quarto render competition_report.qmd --to pdf` with the RStudio-bundled
   quarto on PATH) and recheck the 8-page limit after the addition.
2. `submission_triple_mlp_v13.csv` (2026-07-29, public 1.210, worse) and
   `submission_set_context_v14_candidate.csv` (2026-07-30, public **1.200**,
   new best) are both logged. No other submission slot currently has a
   queued candidate; only 2 Kaggle submissions/day (shared team-wide) -- keep
   requiring a CI that clearly excludes zero before spending one. Clarence's
   model still needs a fixed validation split before it's worth trusting.
3. **An external review (2026-07-30) proposed "test a genuinely new
   structural model: panel mixed logit or latent-class logit" as a next
   step -- this is NOT new.** Both have been tried multiple times and
   rejected: frequentist mixed logit (full and price-only), a Bayesian
   hierarchical mixed logit (deliberately re-tested under a totally
   different estimation philosophy specifically to rule out an estimation
   artifact -- same rejection), latent-class logit (twice), and a
   price-scale latent class (real, stable classes, but redundant with
   existing continuous heterogeneity). Do not re-run any of these without a
   materially different angle; if someone proposes this again, point them
   at this file's "Model progression" negative-results list and the
   Bayesian-mixed-logit "Resolved" section first.
4. The same review's other suggestion -- an OOF residual-correlation check
   to test whether different models fail on the same rows (representation
   limitation) vs. different rows (exploitable complementary signal) -- was
   run directly (own analysis, not delegated): m8trpg's worst-decile rows
   are also dramatically worse than average for every other cached model
   (e.g. xgboost 1.18 overall vs. 2.17 on those rows; MLP 1.19 vs. 2.44).
   Real evidence for a shared, not per-model, error ceiling -- consistent
   with the project's broader "near the practical floor" conclusion.
5. As of 2026-07-30, no untested candidate has a respondent-bootstrap CI
   that cleanly excludes zero against the NEW current best
   (`set_context_utility_network_v14`) -- everything logged so far was
   measured against the prior best. The four-parallel-experiment wave, the
   three-more-experiments wave, and the SVM-diversity result were all
   measured before this promotion and should be considered closed regardless
   (their rejections don't depend on which model is "current best" by this
   small a margin). If a genuinely new idea surfaces, exhaust it via
   respondent-grouped CV before assuming it's real, and keep the same bar:
   ordinary 95% CI must exclude zero, not just a positive point estimate.
