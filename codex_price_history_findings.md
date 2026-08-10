# Price-only design-history anchoring: pre-registered, repeated-CV from the start

Pre-registration: `codex_price_history_preregister.md`, committed at `011b254`
before any fit. Implementation: `R/codex_price_history_only.R`. Raw output:
`data_processed/codex_price_history/` (`price_history_folds.csv`,
`price_history_repeat_summary.csv`, `price_history_bootstrap.csv`,
`price_history_decision.csv`, `price_history_results.rds`).

## Two corrections to the brief, made before running anything

1. **Partial-profile check.** The brief assumed this dataset is full-profile.
   Checked directly against `csv files/train.csv`: for every one of 21,565
   rows, alternatives 1-3 have `rowSums(attrs != 0) == 9` exactly (never 8,
   never 10), and alternative 4 (opt-out) has every attribute at exactly 0.
   This is a genuine, constant partial-profile design (9 of 19 attributes
   active per alternative), matching AGENTS.md's own finding #2, not the
   brief's stated premise. It does not reopen a lever: the active-attribute
   count never varies (nothing to exploit there), and *which* 9 are active is
   already fully absorbed by the existing `factor(attribute)` terms (level 0
   already serves as "not featured") and by the already-tested design-cell and
   299-version fingerprints.
2. Confirmed the version-Newton correction (`codex-overnight-queue`,
   `798662c`, logged as `version_newton_optout_correction`) is closed and not
   re-run as specified; not revisited here.

## Leakage audit (done before pre-registering, re-confirmed against the code)

- `history_design_prior()` is computed only from the fitting fold's source
  respondents (verified: `full_long[!(full_long$Case %in% validation_cases),]`
  at the CV stage).
- Task 1 for every respondent uses the fold-local population prior alone --
  `prior_prices`/`prior_attributes` are empty when the first task is scored,
  so the weighted formula collapses exactly to the population prior
  (algebraically, not just by inspection).
- A respondent's history feature never uses another respondent's tasks.
- Two independent plumbing checks were run and hard-asserted **before**
  trusting any new number: reconstructing `both_k3` (the already-logged
  bundled candidate) from scratch, for (a) canonical fold 1 propagated through
  the full blend architecture (max abs diff vs. the cached
  `history_prior_cv.rds` value: **0.0**) and (b) repeat-seed-1907 fold 1's raw
  mlogit prediction (max abs diff vs. the cached repeated-CV checkpoint:
  **2.22e-16**, machine epsilon). Both confirm the fold/prior/feature
  reconstruction used for the new price-only candidates is exactly correct,
  not approximately similar.

## Family and design

Three candidates, reusing the exact prior-strength grid already fixed in
`history_prior_specs` (not a new grid chosen after seeing anything): the
`both_k3`/`both_k9`/`both_k27` price term alone, with the attribute-
familiarity term dropped entirely.

| Candidate | Term |
|---|---|
| `price_only_k3` | `hist_prior_price_gap_k3` |
| `price_only_k9` | `hist_prior_price_gap_k9` |
| `price_only_k27` | `hist_prior_price_gap_k27` |

Per the brief's explicit instruction, no single-split-only screen gate was
used -- all three went straight to canonical respondent-grouped five-fold CV
(seed 4821) plus five additional repeated-CV seeds
(`1907, 2719, 6151, 8293, 104729`), six fold assignments in total. The
`original_xgb` and `shallow_mlp` ensemble components for the five additional
seeds were reused byte-for-byte from the already-completed repeated-CV round's
cached checkpoints (`data_processed/codex_repeat_cv/checkpoints/`) -- only the
mlogit component differs, exactly matching the established "fixed blend
architecture" pattern.

## Results

**Canonical CV (seed 4821), matching this project's usual reporting format:**

| Candidate | Baseline | Candidate | Gain | Folds improved |
|---|---:|---:|---:|---:|
| `price_only_k3` | 1.143686618 | 1.142918755 | +0.000767863 | 4/5 |
| `price_only_k9` | 1.143686618 | 1.142911201 | +0.000775417 | 4/5 |
| `price_only_k27` | 1.143686618 | 1.142956136 | +0.000730482 | 4/5 |

All three are at least as good as `both_k3`'s own canonical-CV gain
(+0.000735168) -- dropping the noisy attribute term did not cost anything on
this split, and `price_only_k9` is marginally the best of the three.

**Repeated CV (6 fold assignments, respondent-clustered, 100,000-replicate
bootstrap):**

| Candidate | Point gain | 95% CI | Family-3 CI | Win rate | Positive repeats | Negative-coefficient folds |
|---|---:|---|---|---:|---:|---:|
| `price_only_k3` | +0.000602 | [-0.000212, +0.001409] | [-0.000393, +0.001598] | 92.6% | 6/6 | 30/30 |
| `price_only_k9` | +0.000615 | [-0.000252, +0.001476] | [-0.000446, +0.001678] | 91.7% | 6/6 | 30/30 |
| `price_only_k27` | +0.000592 | [-0.000289, +0.001467] | [-0.000486, +0.001667] | 90.6% | 6/6 | 30/30 |

Individual fold gains (30 per candidate) range from about -0.0023 to +0.0016,
and 24 of 30 are positive for every candidate -- consistent, not concentrated
in one lucky repeat. The price-history coefficient is **negative in every
single one of the 30 fits, for all three candidates** (full sign stability,
exceeding the pre-registered bar of >=27/30) -- more stable than the bundled
`both_k3` version even needed to demonstrate, and behaviorally coherent
throughout (respondents anchor toward a reference price, so a higher price
relative to that reference always reduces choice probability).

## Decision

Per the rule fixed in `codex_price_history_preregister.md` before this ran,
promotion requires all four of: gain>0, family-3 lower bound>0, >=5/6 positive
repeats, >=27/30 negative-coefficient folds. **All three candidates pass three
of the four criteria and fail only the family-adjusted lower bound**, which is
negative for every candidate (-0.000393 to -0.000486). None is promoted.

```
"candidate","point_gain","lower_family3","positive_repeats","negative_coefficient_folds","total_folds","promote"
"price_only_k3",0.000601976710019466,-0.000393329843130451,6,30,30,FALSE
"price_only_k9",0.000615476469639146,-0.000446287800570814,6,30,30,FALSE
"price_only_k27",0.000592165244972852,-0.00048563399148866,6,30,30,FALSE
```

## Interpretation

Isolating the price-only mechanism from the bundled `both_k3` candidate was a
legitimate, materially different follow-up (not a re-run of a rejected
experiment), and it produced a clean, informative answer rather than a
repeat of the same ambiguity for a different reason: **the price-anchoring
effect's own size is genuinely about +0.0006 in pooled respondent-level CV
terms, essentially unchanged whether or not the noisy attribute-familiarity
term rides along with it.** Dropping that term neither unlocked hidden signal
that it had been masking, nor cost anything -- both point estimates and both
sign-stability profiles are close to identical to the bundled version's. That
rules out the specific hypothesis that the attribute term was suppressing the
price term's confirmability; it wasn't. The effect is real and directionally
totally consistent (a fully stable coefficient sign across 30 independent
fits is not something a pure-noise term would produce), but at this
project's ~1,135-respondent sample size, an effect of this magnitude
(~0.0006, similar order to the bootstrap SD of ~0.0004-0.0005 per candidate)
sits inside the same noise floor that has closed out every other near-miss
this session.

**Not adopted; no submission made.** This closes the price-only refinement of
the design-history lead. The remaining open direction from the brief -- a
version explicitly borrowing strength from *other*, similar versions rather
than being estimated in isolation (as opposed to a global single-population
prior, which is what `history_design_prior()` already is) -- is a materially
different mechanism from anything tested here or in the original version-
correction round, and is not attempted in this round.

## Reproduction

```powershell
& 'C:\Program Files\R\R-4.6.0\bin\Rscript.exe' R/codex_price_history_only.R
```

Resumable: progress is checkpointed to
`data_processed/codex_price_history/repeat_progress.rds` after each completed
repeat seed.
