# Choice-set-dependent attribute focusing/salience -- findings

Pre-registration: `codex_salience_preregister.md`. Implementation:
`R/codex_salience_screen.R`.

## Result: clean, decisive reject -- optimum is exactly lambda = 0

Base m8trpg validation log loss reproduced exactly (1.159681, matching
every earlier screen using this same split -- a good cross-implementation
check). A coarse grid over `lambda in [0, 5]` and a mirrored grid over
`lambda in [-5, 0]` were evaluated directly (no continuous optimizer
needed, since the grid alone gives an unambiguous answer):

- **The best point in the entire 41-point grid (both directions) is
  `lambda = 0`** -- the unweighted model.
- Moving to `lambda = 0.25` alone already costs +0.00317 (1.162849 vs.
  1.159681); by `lambda = 1` the cost is +0.10 (1.262239); by `lambda = 5`
  the model is worse than the uniform-guess-adjacent range (1.728651).
- The negative-lambda side is equally monotonic and equally harmful in the
  opposite direction (would correspond to *deweighting* attributes with
  high cross-alternative contrast, an anti-focusing effect) -- also
  strictly worse than `lambda = 0` at every point tested.

This meets the pre-registered falsification criterion directly: the
optimal `lambda` is not merely `<= 0`, it is exactly `0`, in both
directions -- there is no evidence at any tested magnitude that
reweighting m8trpg's fitted attribute contributions by their task-specific
cross-alternative range improves prediction. Given the grid result is this
unambiguous (monotonic, large-magnitude harm in both directions), the
pre-registered placebo check (shuffling attribute-range assignment across
tasks) was not run -- it would only be informative if the real version
showed a plausible positive effect to compare against.

## Interpretation

m8trpg's existing per-attribute utility contributions already appear to be
correctly weighted on average; letting the model's attention to an
attribute vary with how much that attribute happens to distinguish the
three bundles in a given task does not capture real behavior here, at
least not in the exact functional form tested (a softmax-normalized
range-based reweighting applied uniformly across respondents). This closes
the specific focusing/salience mechanism as operationalized; it does not
rule out every conceivable implementation of attention-weighting (e.g. one
where the *sign or curvature* of the reweighting differs from a monotonic
softmax-in-range function), but no further variant is being pursued without
a specific new reason to expect a different functional form to behave
differently, consistent with this project's standing practice.

## Not adopted; no submission made

`set_context_utility_network_v14` (1.143533 CV / 1.200 public) remains the
current best model, unchanged. This is the tenth mechanism tested and
closed this session (five from the original coverage audit, three from
ChatGPT's first review round, two -- relative price-gap and salience --
from tracing back the historical improvement pattern and a second
ChatGPT round).
