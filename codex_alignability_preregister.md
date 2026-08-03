# Partial-profile alignability weighting -- pre-registration

Date: 2026-07-31. Proposed by a third external-review round, which
independently found and verified (via paperzz.com-hosted trace of Mishra
et al.'s Management Science paper on what appears to be the same or a
closely-related GM conjoint survey instrument) that this dataset's
attribute level codes carry real semantics: level 0 means "not shown in
this partial profile," while (for every attribute except `CC`) the final
positive level means "shown, explicitly declared absent" -- a
behaviorally distinct state from simply not displaying the attribute.

## Independent verification before building anything

Rather than trust the source, checked the specific, falsifiable structural
prediction directly against this project's own fitted `m8trpg`
coefficients (`data_processed/m8trpg_fulldata_model.rds`), with no
dependence on the source's claimed code-to-feature-name mapping: **for a
smooth quality ladder, coefficients across levels 1..max should not show a
sharp break at the top level; for an explicit-None level, the top level
should break whatever trend the lower levels establish.**

Result: **18 of 19 attributes show exactly this break** -- coefficients
generally positive and often increasing through the middle levels, then
dropping to near-zero or clearly negative at the final level (e.g. `BU`:
0.090, 0.237, 0.206, 0.309, 0.340, **-0.093**; `NV`: 0.288, 0.443,
**-0.113**). **The one specified exception, `CC`, is exactly the one
attribute that does not break** (0.249, 0.200, 0.267 -- no drop). This is
a specific, falsifiable prediction (which one of 19 attributes is the
exception) that panned out, verified entirely from data already on disk.
This meaningfully raises confidence in the source's other specific claims,
though the code-to-feature-name mapping itself (e.g. `NV` = Night Vision)
remains a single external attribution that cannot be independently
confirmed from this anonymized dataset alone.

**Consequently: this candidate (alignability) is prioritized above the
source's own two other proposals (semantically-matched taste
heterogeneity, functional-family saturation) specifically because it needs
*only* the level-0-means-not-shown structural fact (already established
and used throughout this project, e.g. the "9 of 19 active attributes"
finding) and does not depend on trusting which specific real-world feature
each 2-letter code represents.** A further check of the historical 195-term
automated interaction search found it selected `KA x nighta` (Knee Air
Bags), not `NV x nighta` (the semantically "obvious" pairing) -- real,
if not fully decisive, evidence against the semantic-matching candidate
specifically (different functional form was tested there: raw attribute
level x covariate, not fitted part-worth x covariate), which further
supports running the semantics-independent candidate first.

## Hypothesis

Consumer-choice research distinguishes alignable differences (a dimension
shown for multiple alternatives, inviting direct comparison) from
nonalignable/unique features (shown for only one alternative in the
choice set). The existing conditional logit weights an attribute's fitted
part-worth identically regardless of whether competitors also display that
same attribute. Letting a single scalar parameter reweight an
alternative's attribute contributions according to whether each attribute
is uniquely displayed (shown by only 1 of the 3 inside alternatives) vs.
alignable (shown by 2 or 3) will improve on `segment_shift_v15`.

## Mechanism

For each task and attribute `m`, `k_ntm = ` count of inside alternatives
(1-3) with a nonzero level on attribute `m` (0 not counted as "shown").
Using fold-fitted `m8trpg`-family coefficients (same convention as the
already-rejected salience experiment), `c_ntjm` = attribute `m`'s fitted
contribution to alternative `j`'s utility. Correction:

```
U_ntj(gamma) = U_ntj^(v15) + gamma * sum_m c_ntjm * I(k_ntm == 1)
```

`gamma = 0` reproduces the existing model exactly. `gamma < 0` downweights
uniquely-displayed attributes; `gamma > 0` amplifies them. One parameter.
The opt-out alternative needs no special case (`c_nt4m = 0` for every
attribute, so the correction is automatically zero for it).

## Why this is not a duplicate

- Not the rejected salience/focusing experiment: that reweighted by the
  *magnitude of cross-alternative spread* in a fitted quantity (a
  continuous, softmax-normalized range). This reweights by a *binary
  structural fact of the partial-profile display mask* (shown by how many
  alternatives), independent of how similar or different the fitted
  utilities happen to be.
- Not choice-set geometry (near-miss, additive similarity in raw attribute
  space): that measured similarity *between complete bundles*. This
  measures whether *one specific attribute* is comparable across
  alternatives at all.
- Not the sequence/history direction (closed): uses only the current
  task's own display mask, no cross-task information.
- Does not depend on the external source's claimed code-to-feature mapping
  at all -- only on the already-established, already-verified "level 0 =
  not shown" structural fact.

## Minimum viable implementation and screening

Screened first on the canonical single 80/20 split
(`data_processed/train_val_split.rds`), fold-fitted `m8trpg` coefficients,
coarse one-dimensional grid over `gamma` (both signs), before any
commitment to full CV.

## Falsification

Close it if: `gamma = 0` is optimal; the gain does not increase with the
number of uniquely-displayed attributes in a task; randomly reassigning
the shown/not-shown display mask across tasks (within the training fold)
performs comparably to the real mask; or the effect does not propagate
into the full `segment_shift_v15` ensemble.
