# Dispersion-relative price-gap features -- findings

Pre-registration: `codex_relative_price_gap_preregister.md`. Implementation:
`R/codex_relative_price_gap_screen.R`, `R/codex_relative_price_gap_cv.R`,
`R/codex_relative_price_gap_repeated_cv.R`.

## Result: consistent, real, but too-small effect -- near-miss, rejected

Single-split screen (canonical `train_val_split.rds`): base m8trpg 1.159681
(exact match to the already-logged screen number, confirming the
reimplementation) vs. +price_gap_min_rel/+price_gap_max_rel 1.159421, a
small positive gain (+0.000261), individual coefficients not significant
(p=0.57, p=0.11).

Canonical 5-fold CV (seed 4821): baseline (m8trpg, exact match to the
already-logged 1.147021) vs. candidate 1.146883, **gain +0.000139, 95% CI
[-0.000110, 0.000385], win rate 86.4% -- a pre-registered near-miss**,
triggering repeated-CV escalation.

Repeated CV (6 seeds, 30 fold-fits total): **gain +0.0000906 pooled, 95% CI
[-0.000157, 0.000336], win rate 76.5%, positive in 6 of 6 individual
repeats** -- a consistent direction across every seed, but the pooled
effect shrank under repeated CV (as it usually does for near-misses in this
project) and the interval still crosses zero. **Rejected** per the
pre-registered promotion rule (`lower_95 > 0` required).

## Interpretation

A real, small, directionally consistent effect -- dividing the existing
`price_gap_min`/`price_gap_max` features by the task's own price spread
adds a small amount of information, but at roughly a fifth the size of the
smallest gain that has ever cleared this project's bar. This is the same
"diminishing returns" signature seen in nearly every near-miss logged this
project (choice-set geometry, componentwise residual boosting, low-rank
factorization before it reversed sign): a real, non-noise, correctly-signed
effect that is simply too small relative to this dataset's respondent-level
noise floor.

## Follow-up: no-refit shrinkage diagnostic (proposed by external review, closes the question)

Rather than trying another feature or model, tested whether the fitted
correction simply *overshoots* -- blending `p_alpha = (1-alpha) p0 + alpha
p1` (`p0` = m8trpg-alone OOF, `p1` = m8trpg+relative-gap OOF, both already
cached, no refitting) for `alpha in {0.25, 0.5, 0.75, 1}`, with `alpha`
chosen using **only** the canonical seed and then frozen and evaluated
purely out-of-sample on the other 5 repeated-CV seeds
(`R/codex_relative_price_gap_shrinkage.R`).

**`alpha = 1.0` (the full, unshrunk correction) is optimal on the canonical
seed** -- canonical log loss improves monotonically from `alpha=0.25`
(gain +0.0000664) through `alpha=1.0` (gain +0.0001387); there is no
overshoot to shrink away. Per the pre-registered decision rule ("if the
best canonical value is alpha=1, there is no shrinkage opportunity"), this
closes the question directly. The out-of-sample confirmation on the other
5 seeds (using the frozen `alpha=1`, i.e. identical to the original
unshrunk repeated-CV result minus the canonical seed) reproduces the same
pattern: gain +0.0000809, 95% CI **[-0.000168, 0.000327]**, positive in
5 of 5 seeds, still crossing zero.

**Conclusion: no shrinkage rescues this candidate.** The relative price-gap
effect is genuinely positive and directionally stable, but it is already
being used at its optimal (full) strength, and even at that strength it
remains too small to clear this project's promotion bar. This closes the
relative-price-gap line definitively, not just as a near-miss.

## Not adopted; no submission made

`set_context_utility_network_v14` (1.143533 CV / 1.200 public) remains the
current best model, unchanged.
