# Partial-profile alignability weighting -- findings

Pre-registration: `codex_alignability_preregister.md`. Implementation:
`R/codex_alignability_screen.R`.

## Result: the target phenomenon does not exist in this dataset -- structurally closed

Screen on the canonical single split: base `segment_shift_v15`-formula
validation log loss 1.158952. A coarse grid over `gamma in [-2, 2]`
produced **exactly the same log loss (1.158952) at every single point on
the grid** -- gain 0.000000 everywhere.

This is not a weak/null effect; it is a structural non-existence, verified
directly: `n_unique_per_task` (count of attributes shown by exactly 1 of
the 3 inside alternatives, per task) is **0 for all 4,313 validation
tasks**. Extending the check to the full 21,565-task training set, across
all task x attribute pairs (409,735 total): the count of alternatives
displaying a given attribute is **exclusively 0 or 3 -- never 1, never 2**
(215,650 pairs at 0, 194,085 at 3, zero elsewhere, confirmed per-attribute
for all 19 attributes individually).

## What this reveals about the design

The partial-profile mechanism decides, per task, which ~9 of the 19
attributes are "active" -- and once an attribute is active for a task, **all
three inside alternatives** receive a real (non-reference) level for it;
the remaining ~9-10 attributes are inactive (level 0) for all three
alternatives simultaneously. Attribute activity is a **task-level**, not an
**alternative-level**, coin flip. This is a stronger, more precise version
of the already-known "exactly 9 of 19 active per alternative" structural
constant (`cleaning_log.md` finding #2): it is not merely that each
alternative independently ends up with 9 active attributes, but that the
active *set* is shared identically across all three alternatives in a
task.

Consequently, every attribute that is shown at all is automatically
"fully alignable" (comparable across all three alternatives); the
"uniquely displayed, non-comparable" state this candidate modeled never
occurs. No amount of retuning `gamma` can find a signal in a display-mask
distinction that the data's own generating mechanism never produces.

## Not adopted -- closed by design, not by a failed test

This is among the cleanest closures of the session: the mechanism is
provably absent from the data, independent of any modeling choice.
`segment_shift_v15` (CV 1.143255, public 1.200) remains the current best
model, unchanged.
