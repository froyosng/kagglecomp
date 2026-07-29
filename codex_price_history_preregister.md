# Pre-registration: price-only design-history anchoring

Fixed on branch `zhenhao`, working tree at commit `c411db8`, before running any
new fit. This follows directly from the brief's correction: the pre-specified
`both_k3` candidate in `R/codex_history_prior_smoothing.R` (submissions_log.csv
row `history_prior_smoothed_features`) bundled two mechanisms -- a price-history
anchoring term and an attribute-familiarity term -- and reached the closest
near-miss of the whole session (CV gain +0.000735, CI [-0.000090,+0.001564]).
Its price coefficient was negative and stable in all 5 canonical folds and, per
the repeated-CV audit, negative in essentially every fold across all 6 repeats
(see audit below); the attribute-familiarity coefficient flipped sign. The
project's own discipline correctly refused to cherry-pick a price-only refit
out of that pre-specified set at the time. That refit is a legitimate, new
follow-up now, not a re-run of a rejected experiment -- it isolates a mechanism
with a stable, behaviorally coherent sign from one whose sign already looks
like noise.

## Audited for leakage before pre-registering (facts, not claims)

Read directly from `R/codex_history_prior_smoothing.R`:

- `history_design_prior()` is computed **only** from the fitting fold's source
  long data (`fitting_long`); at the CV stage this is
  `full_long[!(full_long$Case %in% validation_cases), ]` -- the held-out
  fold's respondents never contribute to the prior used to score them.
- `add_prior_smoothed_history()` accumulates `prior_prices`/`prior_attributes`
  strictly from a respondent's own tasks processed in Task order, and appends
  the current task's own values to that running history **after** computing
  the current task's feature (`prior_prices <- c(prior_prices, current_prices)`
  is the last line of the per-task loop body). Task 1 therefore has
  `length(prior_prices) == 0`, so its price reference collapses exactly to the
  fold-local population mean and its familiarity collapses exactly to the
  fold-local population frequency -- confirmed algebraically
  (`strength * prior + 0) / (strength + 0)`), not merely asserted.
- No other respondent's tasks enter a given respondent's history (the loop is
  `for (respondent in sort(unique(df$Case)))`, entirely respondent-scoped).
- `stopifnot(all(... df$alt==4L history columns ... == 0))` confirms the
  opt-out alternative never receives a history feature, consistent with the
  rest of the m8trpg specification.

Conclusion: the feature is fold-safe and leakage-free by construction, exactly
as already documented; this pre-registration changes only which of the two
bundled terms enters the formula, not the leakage-relevant machinery.

## Partial-profile claim, checked directly -- brief's premise was wrong, not the reviewer's

The brief asked to confirm this dataset is full-profile before assuming any
partial-profile mechanism, on the basis that "every alternative shows all 19
attributes in every task." Checked directly against `csv files/train.csv`
(21,565 rows, all four alternatives): for alternatives 1-3,
`rowSums(attrs_a != 0)` is **exactly 9 for every single row, no exceptions**;
for alternative 4 (opt-out) every attribute column is exactly 0, as expected.
This is the opposite of full-profile -- it is a genuine, constant partial-
profile design (9 of 19 attributes shown per alternative, the other 10 at
their reference/unshown level), exactly matching AGENTS.md's own finding #2
("every inside alternative has exactly 9 of its 19 attributes at a
non-reference level -- a constant of this partial-profile conjoint design").
The brief's stated premise for this check does not hold; AGENTS.md's existing
characterization does. Flagged back to the user rather than silently
corrected, since acting on the brief's version would have been wrong.

This does not reopen a new lever, though: which 9 of 19 attributes are active
never varies in count (always exactly 9, so there is no "number of features
shown" axis to exploit), and *which* 9 are active is already fully absorbed by
the existing `factor(attribute)` terms (level 0 already serves as the
"not featured" reference category for every attribute in every fitted model
this session) and by the design-cell/questionnaire-version fingerprints
(already tested and found too sparse -- see AGENTS.md's design-cell-shrinkage
and 299-version sections). No new partial-profile-specific mechanism is
pursued in this round.

## Frozen candidate family: 3 price-only prior strengths

Reuses the *exact* strength grid already fixed in `history_prior_specs`
(3, 9, 27) -- not a new, hand-picked grid chosen after seeing anything new.
Each candidate keeps every existing m8trpg term and adds exactly one new term,
`hist_prior_price_gap_k{K}`, with the attribute-familiarity term dropped
entirely:

| Candidate | Prior strength (K) | New term |
|---|---:|---|
| `price_only_k3` | 3 | `hist_prior_price_gap_k3` |
| `price_only_k9` | 9 | `hist_prior_price_gap_k9` |
| `price_only_k27` | 27 | `hist_prior_price_gap_k27` |

(`price_only_k9`'s single-split screen number already exists from the original
round, as `price_k9`, and was mildly negative, -0.0000502 -- but per the
brief's instruction this family proceeds straight to full CV and repeated CV
without a single-split gate, since a single-split screen is already known to be
too noisy to trust for effects this small; that is the entire reason repeated
CV exists in this project.)

## CV design

- Canonical respondent-grouped five-fold CV, seed 4821 (the project's
  canonical fold assignment, `data_processed/oof_ensemble_v10.rds
  $fold_of_case`).
- Repeated CV: the same five additional seeds already used in the project's
  repeated-CV round -- `1907, 2719, 6151, 8293, 104729` -- six fold
  assignments in total, matching `codex_repeated_cv_preregister.md` exactly so
  this result is directly comparable to the existing `history_both_k3` and
  `augmented8_arithmetic` repeated-CV numbers.
- Every mlogit refit is genuinely refit per fold from that fold's training
  respondents only (fresh `history_design_prior` + `add_prior_smoothed_history`
  + `mlogit()` call). The `original_xgb` and `shallow_mlp` ensemble components
  are **not** refit for the five additional seeds -- they are reused byte-for-
  byte from the already-completed, already-cached
  `data_processed/codex_repeat_cv/checkpoints/seed_<seed>_fold_<fold>.rds`
  files produced by the prior repeated-CV round (identical fold assignments,
  identical model specs, so this is exact reuse, not an approximation). This
  matches the project's established "fixed blend architecture" pattern: only
  the mlogit component being tested changes; xgboost and the shallow MLP are
  held fixed, exactly as they were for `history_both_k3`.
- Propagation architecture (frozen, matches `history_both_k3`'s comparison
  exactly): `candidate = 0.85 * (0.80 * history_mlogit + 0.20 * original_xgb)
  + 0.15 * shallow_mlp`, compared against the submitted current best,
  `baseline = 0.85 * (0.80 * mlogit + 0.20 * original_xgb) + 0.15 *
  shallow_mlp`.

## Estimand and bootstrap

Per respondent and repeat, the mean row-loss gain of candidate over baseline;
average the six repeat-level gains within respondent first (1,135 respondent-
level averages); bootstrap those 1,135 values with 100,000 respondent
resamples, matching `repeat_bootstrap_average_gain()` exactly.

Report per candidate:
- fold-level gain (30 rows: 6 repeats x 5 folds) and the sign of the price
  coefficient in every one of the 30 fits;
- the six repeat-level mean gains and how many are positive;
- ordinary 95%/99% respondent-bootstrap CI;
- a family-adjusted CI using family size 3 (this fresh family alone);
- for context only (not the primary bar), a cumulative-family CI folding in
  the 5 candidates from the original `history_prior_smoothing` round plus the
  8 from the original design-history round (family 16), so the result is not
  reported more favorably than the project's own running multiplicity
  accounting would imply.

## Promotion rule (fixed now, before any result is seen)

A candidate is promoted only if **all** of the following hold:
1. Pooled repeated-CV point gain > 0 vs. the submitted current best.
2. The family-3-adjusted 95% lower bound > 0.
3. At least 5 of 6 repeat-level mean gains are positive.
4. The price-history coefficient is negative in at least 27 of the 30
   fold-level fits (sign stability, mirroring the qualitative check that
   already distinguished this term from the discarded attribute-familiarity
   term).

If more than one candidate passes, the one with the largest family-adjusted
lower bound is recommended for a submission slot; if none passes, this closes
the price-only refinement of the lead without further k-value searching.

## Explicitly excluded from this round

No re-litigation of `both_k3`, `both_k9`, `both_k27`, `attribute_k9`, the
version-level opt-out correction, or any other already-logged candidate. No
new k-values beyond {3, 9, 27}. No re-optimization of the 0.80/0.20 or
0.85/0.15 blend weights -- they stay fixed at the values already used for
every other candidate compared against the submitted current best.
