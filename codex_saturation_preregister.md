# Nonlinear feature-family saturation -- pre-registration

Date: 2026-07-31. Candidate 3 from the third external-review round, run
after Candidate 2 (alignability) was structurally closed and Candidate 1
(semantic taste heterogeneity) closed near-chance. Depends on the same
external code-to-feature mapping as Candidate 1 (partially corroborated by
the independently-verified "None-level" structural check,
`codex_alignability_preregister.md`; partially tempered by Candidate 1's
own placebo failures).

## Hypothesis

Within a single alternative, multiple genuinely-present features from the
same functional family (e.g. two or more parking-assistance technologies)
may be redundant (diminishing marginal utility) or complementary
(bundling premium), a nonadditive effect the existing purely-additive
per-attribute factor coding cannot represent. A single scalar parameter on
a within-family same-alternative "redundancy count" will improve on
`segment_shift_v15`.

## Definitions

- **Present** (as opposed to merely "shown"): for every attribute except
  `CC` (which has no explicit-None level per the independently-verified
  exception), present = level is nonzero **and** not the attribute's own
  maximum level. For `CC`, present = level nonzero.
- **Families** (fixed before seeing any result): parking/manoeuvring
  (`BU`,`FA`,`PP`,`LB`); warning/intervention (`LD`,`BZ`,`FC`,`FP`,`RP`);
  visibility/information (`GN`,`NS`,`NV`,`AF`,`HU`); passive/emergency
  safety (`KA`,`SC`,`TS`); control/assistance (`CC`,`MA`).
- `K_jg` = count of present attributes in family `g` for alternative `j`.
- `R_j = sum_g choose(K_jg, 2)` -- one scalar per alternative, summed over
  the 5 families.
- `V_j(delta) = V_j^(v15) + delta * R_j`. `delta = 0` reproduces the
  existing model exactly.

## Why this is not a duplicate

- Not a linear attribute count: the already-known "9 of 19 active" fact
  makes a *linear* count degenerate (constant, already absorbed by the
  factor dummies). `R_j` is a *quadratic* (pairwise-redundancy) function
  of family-restricted presence counts, not a linear combination of
  existing dummy columns.
- Not choice-set geometry (near-miss): that measured similarity *between*
  alternatives. This measures redundancy *within* one alternative's own
  feature set.
- Not alignability (structurally closed): that concerned cross-alternative
  display comparability. This concerns within-alternative feature
  co-occurrence.

## Screening discipline

Screened on the canonical single split, one parameter (`delta`), starting
with the pooled all-family `R_j` before ever considering separate
per-family parameters (per the source's own explicit instruction: "Only
if that works should you estimate separate parking, collision and
visibility saturation parameters"). **Falsification placebo**: randomly
reassign the 19 attributes to 5 same-sized family groups (fixed seed,
computed once) and rerun the identical screen -- a real family-structure
effect must beat this random-grouping placebo.

## What would falsify this

`delta = 0` optimal; a random family reassignment performs comparably;
gains do not concentrate in alternatives with multiple present features
from one family; only a many-parameter (per-family) version shows any
effect; or the effect does not propagate into `segment_shift_v15`.
