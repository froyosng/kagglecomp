# Nonlinear feature-family saturation -- findings

Pre-registration: `codex_saturation_preregister.md`. Implementation:
`R/codex_saturation_screen.R`.

## Result: fails its own placebo test -- closed

Base `segment_shift_v15`-formula validation log loss: 1.158952.

| | Best delta | Gain vs. delta=0 |
|---|---|---|
| Real functional families (parking/warning/visibility/passive/control) | 0.025 | +0.000317 |
| Random family reassignment (fixed seed, same group sizes) | 0.025 | +0.000238 |

The real grouping beats the random placebo by only **~1.3x** -- far weaker
discrimination than Candidate 1's parking x Parallel-Park-Aids pairing
(which showed an effectively infinite margin, +0.000292 vs. 0.000000). Per
the pre-registered falsification criterion ("close it if ... a random
family reassignment performs comparably"), this fires: a quadratic
within-alternative co-occurrence-count term evidently absorbs some
generic flexibility/noise regardless of whether the attributes grouped
together are functionally related, which is not evidence of genuine
feature-family redundancy or complementarity.

## Conclusion

Not escalated to CV given the weak, non-discriminating placebo margin --
consistent with this project's standing practice of not chasing a signal
that fails its own pre-registered falsification check. This closes all
three candidates from the third external-review round (alignability:
structurally absent from the data; semantic taste heterogeneity: one
placebo failure, one ambiguous, one near-chance CV result; feature-family
saturation: fails its placebo test). `segment_shift_v15` (1.143255 CV,
1.200 public) remains the current best model, unchanged.
