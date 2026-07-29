# Findings: transductive covariate-shift correction (codex_transductive track)

Verdict up front: **neither candidate clears the pre-registered promotion bar.**
Both respondent-bootstrap 95% CIs cross zero. `ensemble_v11+MLP` (CV 1.143789,
public 1.201) remains the best model; nothing here is promoted or submitted.
This is an honest null result on both of the two mechanistically-distinct
corrections this track was assigned to test, reported in full per the task's
explicit instruction to report a null result if that is what is found.

Method, validation design, and promotion rule were pre-registered in
`codex_transductive_preregister.md` (committed `3a33415`, 2026-07-29 12:16:08
+0800) before any of the real 5-fold refits or bootstraps below were run;
`git log --format="%h %ci %s"` on this branch confirms that commit predates every
results-bearing commit that follows it.

## 1. What was tested, in one paragraph each

**Candidate A (quantile-mapped moment matching).** m8trpg's four continuous
Price/inside interaction covariates (`incomea`, `agea`, `milesa`, `nighta`) were
recoded, before standardizing, via a quantile mapping that preserves each training
respondent's percentile rank within their own fold's training distribution but
reassigns the corresponding value from `test.csv`'s known (label-free) marginal
distribution for that covariate — a genuinely nonlinear recoding, not an affine
rescale (verified algebraically before implementing that a pure affine
recenter/rescale of these particular interaction terms is either an exact no-op
or a constrained, redundant way to add terms the model doesn't otherwise have,
given m8trpg has no free "plain Price" or "plain inside" main effect — see
Section 3 of the pre-registration). Every training respondent kept likelihood
weight 1; nothing was reweighted.

**Candidate B (confident self-training).** For each of the 5 canonical CV folds, a
vanilla m8trpg was fit on that fold's ~908 real training respondents, used to
score `test.csv`, and its highest-confidence predictions
(`max predicted prob > 0.85`, fixed threshold, not swept) were added back as
pseudo-labeled training tasks (likelihood weight 1, same as genuine rows) before
refitting m8trpg and evaluating on the SAME fold's held-out 227 genuine training
respondents — never on the pseudo-labeled rows themselves.

Both candidates were plugged into the exact same downstream recipe as the current
best (`0.80 * mlogit + 0.20 * xgboost`, then the existing fold-cross-fitted MLP
blend weights), with the xgboost OOF and MLP OOF/weights reused completely
unchanged from the officially-saved artifacts, so the only thing that differs
between "current best" and each candidate is the one treatment under test.

## 2. Results

### Isolated mlogit-component effect (before blending into the full ensemble)

| Component | Pooled OOF log loss | vs. official m8trpg (1.1470212111) |
|---|---|---|
| m8trpg (reimplemented, `recode_fn=NULL`) | 1.1470212111 | exact match (0 abs. difference; see below) |
| m8trpg + quantile-match (Candidate A) | 1.1468490637 | +0.000172 (mlogit-only) |
| m8trpg + confident self-training (Candidate B) | 1.1470908924 | -0.000070 (mlogit-only) |

**Reimplementation correctness check** (`R/codex_transductive_verify_m8trpg_cv.R`):
before trusting either candidate, this track's own from-scratch reimplementation of
the m8trpg feature/formula pipeline (`R/codex_transductive_common.R`) was run
through the unmodified canonical 5-fold CV and compared to the officially-saved
`data_processed/oof_ensemble_v10.rds$oof_mlogit`. Result: **identical to floating-
point precision** (max absolute per-cell difference = 0, pooled log loss matches
to all 10 printed digits: 1.1470212111 vs 1.1470212111). Everything built on top of
this pipeline inherits that same base-case correctness guarantee.

### Full-ensemble verdict (the number that actually matters): respondent-bootstrap vs. current best

| Candidate | Point gain (current_best − candidate) | Ordinary 95% CI | Win rate | Clears bar? |
|---|---|---|---|---|
| A: quantile-match | +0.0000318 | **[-0.000432, +0.000499]** | 55.2% | **No** |
| B: confident self-train | -0.0000065 | **[-0.0000357, +0.0000224]** | 33.3% | **No** |

(Positive point gain = candidate beats current best; both intervals are 100,000-
replicate respondent-clustered bootstraps, identical methodology to
`R/codex_mlp_precision.R`'s `bootstrap_case_means`, reused verbatim.) Current-best
crossfit log loss was hard-reproduced to 1e-10 tolerance
(`1.14378944178` against the officially-logged `1.14378944178118`) in both
`R/codex_transductive_qmatch_bootstrap.R` and
`R/codex_transductive_selftrain_bootstrap.R` before either comparison was computed.

Neither candidate's ordinary 95% CI excludes zero, so — per the pre-registered
promotion rule — **neither is promoted.** (A Bonferroni-adjusted 95% CI for a
family of 2 was also computed for transparency but is moot here since neither
cleared even the unadjusted bar.)

## 3. Why both plausibly came back null (diagnosis, not just the number)

**Candidate A.** The quantile mapping does make a substantively real change to the
covariates it touches — e.g. a training respondent at their fold's median income
(~60,000, matching the project's known figure) is recoded to ~82,000-83,000
(`income_shift_example` in `data_processed/codex_transductive/qmatch_fold_
diagnostics.csv`: +22,000 to +23,000 across the 5 folds), squarely consistent with
the previously-confirmed median-income shift (60,000 train vs 80,000 test). And the
isolated mlogit-component effect is directionally positive and consistent in sign
across all 5 folds (+0.000172 pooled). But m8trpg's income/age/mileage/night
interaction terms are already a small slice of a model with ~140 parameters
dominated by segment, task-position, region/parking, and price-gap/rank effects
(see `cleaning_log.md`'s repeated finding that price-gap and segment terms dwarf
the individual covariate interactions). A small, real gain concentrated in 8 of
~140 coefficients, further diluted first by the fixed 0.80 mlogit blend weight and
then again by the ~0.85 non-MLP ensemble weight, arrives at the full-ensemble level
an order of magnitude smaller than the noise floor this project's own bootstrap
work has repeatedly measured (~0.0002-0.001 SD range for genuine single-mechanism
changes). This is the same dilution-through-blending pattern the project has
already documented for other small, real-but-unconfirmed mlogit-level effects
(e.g. the income x mileage triple interaction, cleaning_log.md 2026-07-27).

**Candidate B.** The self-training mechanism only had a small amount of fuel to
work with by design: an 0.85 confidence threshold on a model whose own CV log loss
is ~1.147 is a strict bar, and only 19-28 of `test.csv`'s 4,997 choice tasks per
fold (0.4-0.6%) cleared it — well under 1% augmentation relative to the ~17,252
real training rows already in each fold. An augmentation this small, even if its
pseudo-labels are of reasonably high quality (they are literally the model's own
most-confident predictions), cannot move a conditional-logit MLE fit on 908
respondents by more than a rounding error, and the point estimate (-0.0000065, an
utterly negligible net harm) and extremely tight CI (width 0.000058) confirm
exactly that: this is not "a real effect too small to detect," it is "too small an
intervention to matter either way." A materially larger effect would require either
a looser confidence threshold (more pseudo-labeled volume, at the cost of label
quality — a real bias/volume trade-off this run did not explore, since the
threshold was pre-registered and fixed, not swept, specifically to avoid a
multiplicity fishing expedition) or a fundamentally different self-training design.

## 4. Anti-circularity safeguard: stated and stress-tested

The task brief is explicit that a self-training candidate must never be validated
by checking consistency with its own pseudo-labels. This was enforced structurally
(the augmented refit's honest evaluation set — the fold's 227 genuine held-out
train respondents — never overlaps with the 19-28 confident test tasks used to
generate pseudo-labels, and the pseudo-labels themselves were generated by a model
that never saw those 227 respondents' true labels either), and then stress-tested
directly rather than merely asserted: for fold 1, I deliberately computed and
report the WRONG, circular number —
`log_loss_matrix(pseudo_truth, predict(augmented_model, newdata = pseudo_rows))`
— alongside the honest number:

| Fold 1 | Log loss |
|---|---|
| Circular (evaluated on the model's own manufactured pseudo-labels) | **0.1255** |
| Honest (evaluated on genuine held-out train respondents, never touched) | **1.2084** |

The ~10x gap between 0.1255 and 1.2084 is exactly the artifact the task warned
about: a model fit on data that includes a label will trivially "predict" that
label well, proving nothing about generalization. This number
(`data_processed/codex_transductive/selftrain_circularity_stress_test.csv`) is
reported here explicitly as a labeled negative control and was NOT used anywhere
in Candidate B's verdict in Section 2 — the verdict there uses only the honest
held-out figure (1.2084 for fold 1, matching the `fold_logloss_honest` column in
`selftrain_fold_diagnostics.csv`).

## 5. Scope decisions and what was deliberately NOT done

- Both candidates modify ONLY the m8trpg mlogit component; xgboost and the MLP are
  reused unchanged from the officially-saved OOF artifacts. This was a deliberate
  tractability choice (stated in the pre-registration) to isolate each treatment's
  effect cleanly; it also matches how every other single-mechanism candidate in
  this project's history has been tested (frozen ensemble, one swapped ingredient).
- The 0.80/0.20 mlogit/xgb weight and the per-fold MLP blend weights were reused
  exactly as saved, never re-optimized for either candidate — re-optimizing on the
  same data used to evaluate the candidate would be a form of leakage this project
  has flagged before ("optimistic since the weight was chosen on the same evaluated
  data" — cleaning_log.md, 2026-07-27 triple-interaction section).
- Candidates A and B were kept orthogonal (B's scaler uses only real respondents,
  never the quantile-matched recoding) so each is separately attributable. A
  combined A+B candidate was NOT tested: both individual components came back at or
  below noise, a combination is very unlikely to clear the bar, and testing it now
  — after seeing both null results — would be exactly the kind of post-hoc,
  unregistered fishing this project has correctly avoided elsewhere. If a future
  session wants to test the combination, it should be pre-registered as its own
  round before running, not appended here.
- The self-training confidence threshold (0.85) was fixed by the task brief and not
  swept; a threshold sweep would reopen the same multiplicity problem this project
  has repeatedly had to correct for elsewhere (Bonferroni-adjusted intervals on the
  MLP architecture screen, the history-feature family, etc.).

## 6. Reproducibility

All numbers above are regenerated from the copied raw data
(`csv files/train.csv`, `csv files/test.csv`) plus the two copied canonical
artifacts (`data_processed/oof_ensemble_v10.rds`,
`data_processed/codex_behavioral_round/mlp_oof.rds` +
`data_processed/codex_behavioral_round/mlp_cv_weights.csv`), by, in order:

1. `R/codex_transductive_verify_baseline.R` — reconstructs and hard-asserts the
   current best's officially-logged OOF numbers from the copied artifacts alone.
2. `R/codex_transductive_datacheck.R` — data-availability check (no NAs), run
   before pre-registration since it touches no modeling/evaluation logic.
3. `R/codex_transductive_common.R` — shared feature/formula/quantile-map/bootstrap
   helpers.
4. `R/codex_transductive_smoke.R` — unit-level smoke tests on synthetic/subset data
   (quantile-map monotonicity and hand-checked values, `to_long` structural
   invariants, a 150-respondent subset mlogit fit, and a "recode_fn is actually
   wired in" sanity check), run before the full pipeline was trusted.
5. `R/codex_transductive_verify_m8trpg_cv.R` — full 5-fold reimplementation-vs-
   official parity check (exact match).
6. `R/codex_transductive_qmatch.R` -> `R/codex_transductive_qmatch_bootstrap.R` —
   Candidate A.
7. `R/codex_transductive_selftrain.R` -> `R/codex_transductive_selftrain_bootstrap.R`
   — Candidate B, including the anti-circularity stress test.

Generated artifacts live under `data_processed/codex_transductive/` on disk in this
worktree. Per this project's established convention, `data_processed/` (all of it,
project-wide, not just this track's subfolder) is gitignored and not committed --
every prior codex round's raw output CSVs/RDS files followed the same rule
("gitignored but present on disk from the actual run", per `cleaning_log.md`) -- so
none of these files travel with the git branch. A reviewer who wants to recheck the
headline numbers without trusting this write-up should rerun the 7 scripts in the
order listed above against their own copy of the 3 required inputs
(`csv files/train.csv`, `csv files/test.csv`, `data_processed/oof_ensemble_v10.rds`,
`data_processed/codex_behavioral_round/mlp_oof.rds` +
`.../mlp_cv_weights.csv`); every script hard-asserts its own intermediate
reproduction of the officially-logged baseline numbers before computing anything
new, so a silent divergence cannot pass unnoticed. Files produced on disk (not
committed): `qmatch_precision.csv`, `selftrain_precision.csv`,
`qmatch_fold_diagnostics.csv`, `selftrain_fold_diagnostics.csv`,
`selftrain_circularity_stress_test.csv`, `m8trpg_reimplementation_check.rds`,
`current_best_reconstruction.rds`, plus the full OOF `.rds` files for both
candidates.

## 7. Bottom line for the coordinating session

Both mechanistically-distinct corrections assigned to this track — moment-matching
the continuous covariates used in m8trpg's interactions, and confident self-
training on the current best's own test predictions — were implemented, verified
against a byte-exact reimplementation of the existing pipeline, and evaluated
honestly via the canonical respondent-grouped 5-fold CV and a 100,000-replicate
respondent-clustered paired bootstrap against `ensemble_v11+MLP`. **Neither clears
the pre-registered ordinary-95%-CI promotion bar.** Do not merge or submit either
candidate. `ensemble_v11+MLP` remains the best model this project has.
