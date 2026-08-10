# Semantically-matched feature-taste heterogeneity -- pre-registration

Date: 2026-07-31. Candidate 1 from the third external-review round, run
after Candidate 2 (alignability) was structurally closed. Depends on
trusting the source's claimed code-to-feature mapping (Night Vision =
`NV`, Adaptive Front Lighting = `AF`, Parallel Park Aids = `PP`, Backup
Aids = `BU`, Front Park Assist = `FA`, Low-Speed Braking Assist = `LB`,
Cruise Control = `CC`, Lane Departure = `LD`), which cannot be
independently confirmed from this anonymized dataset -- only the
structural "None-level" pattern was independently verified
(`codex_alignability_preregister.md`). Tempering evidence already on
record: the historical 195-term automated interaction search picked
`KA x nighta`, not `NV x nighta`, when every attribute was available to
compete for that slot (different functional form: raw level vs. fitted
part-worth, so not fully decisive).

## Hypothesis

Night-driving percentage, annual mileage, and parking situation currently
shift price sensitivity and opt-out propensity in `segment_shift_v15`, but
cannot shift the taste for the *specific* safety features those exposures
would logically make more relevant (Night Vision for night driving,
Cruise-Control/Lane-Departure for highway mileage, Parallel Park Aids for
parking situation). Adding each covariate's interaction with the
*fold-fitted part-worth contribution* of its semantically matched
feature(s) -- not raw level codes -- will improve on `segment_shift_v15`.

## Minimum viable implementation (the "most conservative screen" specifically)

Tested as three **separate**, single-parameter residual corrections (not
jointly), each against the frozen `segment_shift_v15` formula's own
fold-fitted predictions, exactly mirroring the already-established
screen-then-CV discipline:

1. `gamma_1 * z(nighta) * C_NV` (Night Vision's own fitted contribution)
2. `gamma_2 * z(milesa) * C_CC_LD` (Cruise Control + Lane Departure
   combined contribution, the two clearest "highway" features in the
   claimed group)
3. `gamma_3 * z(pparkind) * C_PP` (Parallel Park Aids' own fitted
   contribution; `pparkind` treated as continuous for this first pass,
   an approximation flagged as such)

`gamma = 0` reproduces the existing model exactly in every case. Screened
on the canonical single 80/20 split before any CV commitment.

## Why this is not a duplicate

- Not the 195-term automated interaction search: that used raw attribute
  level codes (already established as not cardinally meaningful for most
  attributes) interacted with covariates, not fitted part-worth
  contributions.
- Not the rejected low-rank factorization: that asked the data to discover
  arbitrary broad heterogeneity directions with no restriction on which
  covariate should relate to which feature. This tests only pre-specified,
  theory-motivated pairings with a strong inductive bias (3 parameters,
  not a learned factorization).
- Not any already-tested attribute x segment/covariate term (the existing
  model has none; the only prior sweep used raw levels, per above).

## Falsification (from the source, adopted verbatim)

Close it if: the semantic pairing does not beat a mismatched placebo
pairing (e.g. night% x Cruise-Control contribution instead of night% x
Night-Vision contribution); signs vary across folds; the gain concentrates
only in extreme covariate tails; or the effect does not propagate into
`segment_shift_v15`.
