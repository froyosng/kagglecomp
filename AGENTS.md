# Analytics Edge Data Competition 2026 -- Project Memory

## Competition overview
- Kaggle competition: predict which of 4 car safety-feature bundles a respondent
  chooses (discrete choice / conjoint task). Alternative 4 is always a constant
  "opt-out"/no-purchase option (all attributes and price = 0).
- Scored by multi-class log loss. Benchmark (predict 1/4 for every alternative)
  scores 1.38629.
- Dates: competition ends 1 Aug 2026 (12:00 SGT); report due 10 Aug 2026.
- Max 2 Kaggle submissions/day. R only, but any package (and AI assist) allowed,
  as long as sources are cited in the report for non-class methods.
- Report requirements: max 8 pages, NO executive summary, NO appendix. Must cover
  (i) best model on public LB + brief discussion of alternatives tried,
  (ii) public vs. private leaderboard fit discussion,
  (iii) insights + limitations.
- Grading: 7 pts private LB, 8 pts public LB (>= benchmark gets partial credit
  on both), 15 pts report.

## Data files (in `csv files/`, gitignored -- not in repo)
- `train.csv`: 21,565 obs, 113 vars. 1,135 respondents (`Case`), each completed
  exactly 19 choice tasks (`Task`). Columns: attribute codes (CC, GN, NS, BU, FA,
  LD, BZ, FC, FP, RP, PP, KA, SC, TS, NV, MA, LB, AF, HU) + Price, each suffixed
  1-4 for the 4 alternatives. Outcome: Ch1-Ch4 (one is 1, rest 0 per row).
  Respondent covariates: segment, year, miles, night, ppark, gender, age, educ,
  region, Urb, income -- each has a raw text version, an integer `*ind` version
  (clean 1:1 recode of the raw version, verified), and some also have a `*a`
  version (finer-grained numeric estimate *within* the `*ind` bin, NOT a
  redundant duplicate -- e.g. milesa holds actual values like 60/80/100 within
  the "51 To 100 Miles" bin).
- `test.csv`: 4,997 obs, same schema but Ch1-Ch4 are NA (the targets to predict).
- `sample_submission.csv`: format is `No, Ch1, Ch2, Ch3, Ch4` with `No` = test
  row id (21566-26562).

## Data quality findings (see `cleaning_log.md` for full detail)
- **Test respondents are DISJOINT from train**: train is `Case` 1–1135, test is
  1136–1398 (263 new respondents × 19 tasks = 4,997 rows), zero overlap. Consequence:
  we can never personalize to a specific test respondent, so respondent-specific random
  effects (mixed logit) cannot transfer -- only OBSERVED heterogeneity (covariate/segment
  interactions) generalizes. This is why the mixed-logit models never beat the fixed
  observed-heterogeneity models, and why the respondent-level train/val split & CV are
  the right validation design (they mimic "predict for unseen respondents").
- Survey-fatigue effect: opt-out share rises from ~24% (Task 1) to ~34% (Tasks 15–19);
  `Task` position is a usable, transferable predictor (test respondents also did 19 tasks).
- No missing values or duplicate rows in train.
- Alternative 4 (opt-out): all attributes/price = 0 always; chosen 30.2% of the
  time overall. The all-reference-level attribute profile NEVER occurs among
  alternatives 1-3 (verified) -- this matters for modeling (see below).
- Attribute levels are integer codes with different max per attribute (e.g. CC
  0-3, NS 0-5, BU 0-6, Price 0/1-12). Higher = "more advanced" per the codebook
  (see automobiles/oscars-style course notebooks for analogous datasets/vars).

## Key technical finding: ASC identification conflict
Dummy-coding (factor) the attribute levels while keeping alternative-specific
intercepts (ASCs) causes a **singular Hessian error** in `mlogit`. Cause:
alternative 4's all-reference-level profile never co-occurs with alts 1-3, so
ASC4 is unidentifiable once attributes are factor-coded. Fixes used: (a) drop
ASCs entirely (alt 4's utility is then fixed at 0 by construction), or (b) add
explicit 0/1 dummies for alts 2 and 3 only (not 4).

For `mlogit`'s panel/mixed-logit models, use nested dfidx syntax to expose the
respondent id: `dfidx(data, idx = list(c("chid", "Case"), "alt"), choice = "chosen")`
then `id <- idx(mf, 1, 2)` internally gives panel grouping by respondent.

## Reproducibility pipeline (already built, re-run as needed)
- `R/log_loss.R`: `log_loss(actual, pred)` -- matches the competition's exact
  metric; verified against benchmark (1.386294 for uniform 1/4 guess).
- Long-format reshape (`train_long`, `tidyr::pivot_longer`) and a **respondent-
  level** 80/20 train/validation split (seed 7402, `val_cases`), saved to
  `data_processed/train_val_split.rds` (gitignored). Splitting by respondent
  (not row) avoids leaking a person's 19 repeated tasks across the split.
- `submissions_log.csv`: running tracker of every model tried -- local
  validation log loss, public LB score (once submitted), gap, and a
  `source_citation` column (report must cite non-class methods).
- `cleaning_log.md`: full narrative log of data audit + modeling findings, for
  pulling into the report.
- Local environment note: this machine's security policy blocks some compiled
  tidyverse DLLs (tibble/utf8 printing) -- use plain data frames (`as.data.frame()`)
  instead of tibbles when this happens.

## Models tried so far (validation log loss, best to worst)
Note: rows added 2026-07-25 (bottom-up review onward) report BOTH single-split val
and 5-fold respondent-grouped CV where available; CV (seed 4821 folds) is the more
reliable number. Under the same CV folds, plain mod8 = 1.1662 (baseline for comparing
the new terms).

| Model | Val. log loss | Public LB | Notes |
|---|---|---|---|
| **ensemble_v9**: 0.70×(mod8+task+region+ppark) + 0.30×xgboost | **1.1517 (CV)** | pending | **CURRENT BEST**; `submission_ensemble_v9_mlogit_xgb.csv` (ready to submit). Blend weight chosen by 5-fold CV on out-of-fold preds; flat optimum 0.65–0.75 |
| m8tr: mod8 + task-fatigue + region× + ppark× interactions | 1.1696 / **1.1567 (CV)** | -- | best single mlogit; region/ppark gain CV-confirmed (not just single-split) |
| m8t: mod8 + task-fatigue (In_task, P_task) | 1.1866 / 1.1622 (CV) | -- | P_task highly significant: price sensitivity rises over the 19 tasks (survey fatigue) |
| xgboost: gradient-boosted trees, multiclass (nrounds=73) | 1.2042 / 1.1787 (CV) | -- | worse alone, but valuable in ensemble (makes different errors than the logit) |
| mod8: mod7 + Price×segment + inside×segment interactions | 1.1896 / 1.1662 (CV) | pending | superseded by m8tr/ensemble; `submission_mlogit_v8_segment_interactions.csv` |
| glmnet cox LASSO (stratified-Cox = conditional logit, L1 interaction selection) | 1.1937 | -- | rediscovers mod8's structure from a 195-term pool; confirms but doesn't beat it |
| mod7: mod2b + Price×covariate + inside-good×covariate interactions | 1.2024 | **1.230** | gap vs val 0.028, comparable to mod1's; beats mod1 public (1.270) |
| mod6: mod2b + Price×covariate interactions (income, age, miles, night) | 1.2049 | -- | big gain from observed price-sensitivity heterogeneity; superseded by mod7 |
| mod2b: factor-coded conditional logit + alt2/alt3 dummies | 1.219 | pending | submitted as `submission_mlogit_v2_factors_dummies.csv` |
| mod2a: factor-coded conditional logit, no ASC | 1.220 | -- | superseded by mod2b |
| lasso_multinomial_v1: glmnet LASSO multinomial (wide format, per-class independent coefs, grouped CV by respondent) | 1.226 | -- | different model family (no shared slope across alts); 58/151 predictors retained, pulled in respondent covariates not yet in mlogit models |
| mod5: mixed logit, Price random only | 1.235 | -- | avoids overfitting seen in mod4 |
| mod1: continuous conditional logit + ASCs | 1.236 | **1.270** | only model submitted to Kaggle so far; gap ~0.034, reassuring (no overfitting) |
| mod4: mixed logit, all 20 attrs random | 1.247 | -- | **overfits**: training LL much better (-16449 vs -20432) but validation worse than mod1 -- cautionary tale used in report |
| cart_v1: default rpart CART | 1.293 | -- | worse than all logit models, as expected for this task type |
| benchmark (1/4 each) | 1.386 | 1.38629 | sanity check |

Regsubsets screening (linear-probability heuristic, not a real model) confirmed
all 20 attributes are worth keeping, with Price entering first and dominating
(R^2 = 0.066 of 0.077 total across all 20 vars).

## Git workflow
- Working branch: `zhenhao`. `.gitignore` excludes `csv files/` and
  `data_processed/` (raw + derived data, regenerate locally).
- Merged into `main` once already; resolved a `.gitignore` conflict by taking
  the union of both versions' rules.

## Next steps / what to do in a new session
1. **Submit the ensemble** (`submission_ensemble_v9_mlogit_xgb.csv`, CV 1.1517) --
   current best. Also worth submitting mod8 (`submission_mlogit_v8_segment_interactions.csv`)
   and m8tr for public-LB calibration. mod7 already submitted: public 1.230 (val 1.2024,
   gap 0.028). If the ~0.03 val-to-public gap holds, ensemble expected public ~1.18.
   Log public scores in `submissions_log.csv` when available.
2. DONE (bottom-up review 2026-07-25): added task-fatigue (In_task, P_task -- price
   sensitivity rises over the 19 tasks) and region×/ppark× interactions to mod8; both
   CV-confirmed. Built a 0.70/0.30 mod8(+task+region+ppark)/xgboost ensemble, weight
   chosen by 5-fold CV on OOF preds. The final feature builder is `build_all(df,ctr,scl)`
   in the session (segment + task + region + ppark interactions; scaler from TRAINING).
3. **The report (`competition_report.qmd`) is now STALE** -- it features mod8 (1.1896)
   as best. Update it to: (a) feature the ensemble as best (CV 1.1517), (b) add the
   task-fatigue and region/ppark findings, (c) add the KEY insight that test respondents
   are entirely new people (Case 1136–1398, zero overlap with train 1–1135) -- this
   explains why mixed-logit random effects don't transfer and justifies the
   observed-heterogeneity strategy; our respondent-level CV mirrors this correctly.
4. Remember: only 2 Kaggle submissions/day -- use local CV log loss to choose what's
   worth submitting rather than testing everything on Kaggle.
5. If prompting a fresh session: ask me (the assistant) to read this file plus
   `cleaning_log.md` and `submissions_log.csv` first, then say what you want to try
   next (e.g. "submit the ensemble", "update the report", "test attribute interactions").
