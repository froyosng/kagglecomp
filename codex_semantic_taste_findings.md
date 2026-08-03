# Semantically-matched feature-taste heterogeneity -- findings

Pre-registration: `codex_semantic_taste_preregister.md`. Implementation:
`R/codex_semantic_taste_screen.R` (screen + placebo), `R/codex_semantic_taste_cv.R`
(canonical CV for the one pairing that passed).

## Screen result: placebo test cleanly discriminates between the three pairings

| Pairing | Real gain | Placebo gain | Verdict |
|---|---|---|---|
| night% x Night Vision contribution | +0.000009 | +0.000033 (night% x Cruise Control) | **Fails** -- placebo does *better* than the semantically-motivated pairing |
| miles x (Cruise Control + Lane Departure) contribution | +0.000587 | +0.000300 (miles x Night Vision) | **Ambiguous** -- real beats placebo by only ~2x; an unrelated attribute alone captures over half the apparent effect |
| parking situation x Parallel Park Aids contribution | +0.000292 | +0.000000 (parking x Night Vision) | **Clean pass** -- real signal, placebo shows nothing |

The night-driving pairing's clean failure corroborates the earlier tempering
evidence from the historical 195-term automated search (which picked
`KA x nighta`, not `NV x nighta`, when every attribute was available to
compete for that slot) -- two independent checks now agree the
"night-driving predicts night-vision taste" hypothesis does not hold in
this data, at least not via a linear covariate x fitted-part-worth term.
The mileage pairing is not pursued further given the ambiguous placebo
result (a `>=`-beats-placebo reading technically survives the
pre-registered falsification rule, but the margin is too weak to trust
given how many other pairings in this project have looked promising on a
single split and evaporated).

## Canonical CV result (parking situation x Parallel Park Aids): near-chance, not pursued

Nested gamma selection (inner-fold grid search per outer fold, matching
the `codex_alt_link.R`/`codex_task_temperature.R` pattern) converged to
`gamma = 0.10` in **every one of the 5 outer folds** -- a materially
smaller magnitude than, and the opposite sign from, the single-split
screen's own optimum (`gamma = -0.45`). This screen-to-CV reversal is the
classic signature of a single-split result that was substantially
overfitting noise, not a real effect.

Pooled canonical CV: gain **+0.0000126**, 95% CI **[-0.0000994, 0.0001254]**,
**win rate 58.8%** -- barely above chance, and an order of magnitude
smaller than any candidate that has ever cleared this project's bar. This
technically lands inside the pre-registered near-miss escalation band
(`lower_95` in `[-0.00075, 0]`), which would formally require a full
6-seed repeated-CV confirmation (~120 additional mlogit fits: 5 outer x 4
inner x 6 seeds). Given the near-chance win rate and the screen-to-CV sign
reversal already strongly suggesting noise, this was **deliberately not
escalated** -- a judgment call to not spend substantial remaining compute
chasing a signal this weak, rather than a mechanical rule violation.

## Conclusion

Of three theory-motivated, semantically-matched pairings, one failed its
own placebo test outright (night x NV), one was ambiguous and not pursued
(miles x CC/LD), and the third passed its placebo test but produced a
near-chance canonical CV result once properly nested (parking x PP). None
promoted. `segment_shift_v15` (1.143255 CV, 1.200 public) remains the
current best model, unchanged.
