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
  respectively); `submit_ensemble_v11.R` generates the current best submission;
  `error_analysis.R` / `calibration_check.R` are the diagnostic scripts behind
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
| **ensemble_v11: 0.80 m8trpg + 0.20 xgboost** | -- | **1.145** | **1.202** | **CURRENT BEST**, both CV and public. Gap 0.057, largest yet |

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

## Known weakness: CV-to-public gap, and what actually drives it (updated 2026-07-27)
| Model | CV/Val | Public | Gap |
|---|---|---|---|
| mod1 | 1.236 | 1.270 | 0.034 |
| mod7 | 1.202 | 1.230 | 0.028 |
| ensemble_v9 | 1.152 | 1.204 | 0.052 |
| ensemble_v11 (mlogit+xgboost blend) | 1.145 | 1.202 | 0.057 |
| mlogit_m8trpg standalone (no xgboost) | 1.147 | 1.213 | **0.066 (largest)** |

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

## CV-confirmed candidate awaiting submission: MLP ensemble diversity (2026-07-27)
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
`submission_codex_mlp_v12_candidate.csv` is generated (full-data fit, 5
seeds, 15% weight -- matching the mean of the honestly cross-fitted fold
weights -- against the exact already-public-scored ensemble_v11 CSV) but
**not yet submitted**. Recommended as the top-priority candidate for the
next available submission slot, ahead of the triple-interaction and 4-way
ensemble candidates prepared earlier, given it is the only one to clear the
ordinary bootstrap bar and the most robust to outlier respondents.
`ensemble_v11` remains the officially adopted model until a submission
confirms or refutes this.

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
1. Render `competition_report.qmd` to PDF and do a final wording/layout pass
   (no Quarto/TeX available in this environment yet).
2. Only 2 Kaggle submissions/day (shared team-wide) -- use CV to decide what's
   worth a slot. Clarence's model still needs a fixed validation split before
   it's worth trusting or submitting.
3. If a genuinely different structural idea surfaces (see "open question"
   above), it's worth testing -- but exhaust it via CV before assuming it's a
   real gain, given how many individually-significant terms have turned out to
   hurt validation this session. The bar for spending a submission slot: the
   respondent-bootstrap CI must clearly exclude zero, not just have a positive
   point estimate (see the Codex ensemble candidate above for why this matters).
