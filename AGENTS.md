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
  the growing CV-to-public gap; not yet turned into a validated fix (a
  distribution-shift correction can't be honestly scored via in-training CV,
  and a simpler univariate income-tercile check gave an ambiguous, not
  clearly-actionable signal -- see cleaning_log.md point 4).
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

**Conclusion:** the low-1.1x leaderboard score, if genuine, is not explained
by anything in this round's investigation -- the two most theoretically
credible levers (design-cell empirical information, latent/discrete
heterogeneity) both failed on honest validation. Combined with the earlier
calibration diagnostic (well-calibrated, no exploitable pattern in the
misses, xgboost can't out-predict the logit), the case that ensemble_v11 is
near this dataset's practical ceiling for legitimate, generalizable modeling
is now supported from multiple independent angles, not just one. The
confirmed income shift and blocked-design overlap are real, report-worthy
insights regardless -- cite them in the report even though neither improved
the score.

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
- Zeening's random forest (`rf_gridsearch`, ranger, grid-searched mtry/
  min.node.size) submitted: public 1.259. Her internal CV (1.162) is unreliable
  for the same reason as Clarence's -- her split is fully row-level random
  (`sample(1:nrow(...))`), not respondent-grouped, so nearly every respondent's
  tasks are scattered across both her train and validation sets. Gap (0.097) is
  the largest of any model in the project, empirically confirming the concern.
  Her K-means "persona" clustering idea was re-tested as a logit heterogeneity
  axis (see negative results above) -- didn't transfer, but was a legitimate
  idea worth checking.

## Next steps
1. Update `competition_report.qmd` -- currently stale, still features mod8 as
   best. Needs: ensemble_v11 as best model, the price-factor/price-gap findings
   with the identification story (good technical narrative for the report), the
   test-respondent-disjointness insight, the calibration/error diagnostic as
   evidence for the limitations section, and the growing CV-to-public gap for
   the public-vs-private section.
2. Only 2 Kaggle submissions/day (shared team-wide) -- use CV to decide what's
   worth a slot. Clarence's model still needs a fixed validation split before
   it's worth trusting or submitting.
3. If a genuinely different structural idea surfaces (see "open question"
   above), it's worth testing -- but exhaust it via CV before assuming it's a
   real gain, given how many individually-significant terms have turned out to
   hurt validation this session.
