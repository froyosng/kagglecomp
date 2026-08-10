# Alternative-specific asymmetric probability link -- findings

Pre-registration: `codex_alt_link_preregister.md` (committed before any
result was seen). Implementation: `R/codex_alt_link.R`. Second experiment of
this session, run after the dedicated hurdle model
(`codex_hurdle_model_preregister.md`/`codex_hurdle_model_findings.md`) was
independently pre-registered, implemented, and rejected.

## Result: all three families reject decisively at the canonical stage

Gradient checks (analytic vs. finite-difference, `numDeriv::grad`) passed to
~1e-9 precision for all three families before any real fold was fit. Nested
respondent-grouped canonical CV, penalty chosen per outer fold by inner CV
(`penalty_grid = c(1, 0.1, 0.01, 0.001, 0)`, ties favor stronger shrinkage):

| Family | Mean fold theta | Canonical gain | 95% CI | Verdict |
|---|---|---|---|---|
| 1: global scale (`a=0` fixed, `b` free) -- replicates the already-logged post-hoc temperature sweep | `b=0.00609` | **-0.000531** | **[-0.000873, -0.000212]** | REJECT, entirely below zero |
| 2: opt-out-specific shift (`b`, `c` free) | `b=0.00630, c=-0.00365` | **-0.000777** | **[-0.001136, -0.000438]** | REJECT, entirely below zero |
| 3: shape + scale (`a`, `b` free) | `a=0.00509, b=0.00179` | **-0.000738** | **[-0.001217, -0.000293]** | REJECT, entirely below zero |

None of the three cleared even a near-miss (`lower_95` between -0.00075 and
0) -- every one has a **decisively harmful** interval, the same pattern as
the two already-logged global recalibration experiments (post-hoc
temperature/shrinkage, global utility-scale heterogeneity), not a near-miss
crossing zero. No repeated-CV escalation was triggered (correctly, per the
pre-registered gates: none passed or near-missed).

Family 1 was pre-registered specifically as a replication sanity check
(gate #1): its result (small negative theta effect, decisively harmful,
selected penalty mostly the strongest available) closely matches the
already-logged post-hoc-calibration finding ("identity is optimal... every
deviation makes it worse monotonically") -- confirming the estimation
pipeline, penalty selection, and bootstrap machinery are implemented
correctly before trusting families 2 and 3's results.

## Interpretation

This closes the "alternative probability link" avenue more thoroughly than
the previously-logged experiments did, because it tests strictly more
flexible link shapes than a pure scalar temperature or a covariate-indexed
scale:

- Family 2 shows that even a **single global additive shift specifically on
  the opt-out log-odds** (independent of any respondent/task covariate, the
  cheapest possible test of "is the opt-out margin systematically
  mis-scaled relative to the inside bundles") is harmful, not neutral --
  v14's opt-out calibration is not just "on average right" but actively
  penalized by any uniform nudge in either direction.
- Family 3 shows that adding a **shape** parameter (reweighting the
  surprisal `-log(p)` by a power exponent before rescaling, a strictly
  larger family than pure scale) is *also* harmful, ruling out "softmax's
  problem is specifically its exponential shape, not just its temperature"
  as a live hypothesis.

Combined with the two already-rejected global recalibration experiments,
three structurally different generalizations of "adjust the link, not the
utility" (covariate-indexed scale, global scale, opt-out-specific shift,
shape+scale) all agree: v14's plain softmax on its own frozen utilities is
already at (or very near) a local optimum for this dataset, from every angle
tested so far.

## Consequence for the pre-registered third candidate (task-difficulty-conditioned local temperature)

This project's own adversarial coverage audit ranked a *task-content-
conditioned* bounded temperature (varying by choice-set difficulty features
rather than a single global constant) as the third candidate, specifically
because it had not been tested and is structurally distinct from the two
already-rejected *global*/*covariate-indexed* recalibration experiments.
Family 3's result raises the bar that candidate needs to clear: a
task-conditioned scale is a **strict generalization** of family 1's global
scale (family 1 is the special case where the temperature function is
constant), and family 3 already shows that adding shape flexibility *on top
of* scale flexibility does not help either. This does not by itself close
the task-conditioned version -- a function of genuine task-difficulty
content is a different mechanism from a shape exponent -- but it is real,
concrete evidence (not merely an assumption) that this project's link
function is unusually robust to added flexibility along several dimensions
now, and materially lowers the expected value of that follow-up relative to
where the original ranking placed it.

## Not adopted; no submission made

`set_context_utility_network_v14` (1.143533 CV / 1.200 public) remains the
current best model, unchanged.
