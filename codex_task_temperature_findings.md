# Task-content-conditioned local temperature -- findings

Pre-registration: `codex_task_temperature_preregister.md` (committed before
any result was seen). Implementation: `R/codex_task_temperature.R`. Third
experiment of this session, run after the hurdle model and the three
alternative-link families were independently pre-registered, implemented,
and rejected (`codex_hurdle_model_findings.md`, `codex_alt_link_findings.md`).

## Result: clean, decisive reject, same pattern as every link/scale variant this session

Gradient check passed to 3.5e-10 precision before any real fold was fit.
Canonical respondent-grouped CV: **gain -0.000564, 95% CI
[-0.000913, -0.000239] -- entirely below zero**, not a near-miss. No
repeated-CV escalation triggered (correctly rejected per the pre-registered
gate).

Mean fold `theta = (intercept=0.00605, gap_top2=-0.00055, price_cv=0.00029)`:
the intercept alone (0.00605) is almost identical to the alternative-link
experiment's family-1 global-scale estimate (0.00609), while both
task-difficulty coefficients are an order of magnitude smaller and
contribute essentially nothing. In every outer fold, the nested penalty
search picked the *strongest* available penalty (1.0) in 4 of 5 folds (only
fold 1 chose 0.1) -- the inner cross-validation itself is trying to shrink
this model back toward identity as hard as the grid allows, and the harm
persists anyway.

## Interpretation

This confirms, directly rather than by inference, what
`codex_alt_link_findings.md` flagged as the expected outcome: conditioning
the temperature on task-difficulty content (closeness of the leading
bundles' predicted probabilities, price coefficient of variation) does not
rescue the already-rejected global-scale adjustment -- it just reproduces
the same harmful global component with two near-zero, non-contributing
task-difficulty terms attached. This is now the **fourth** structurally
different link/scale generalization this session and prior sessions have
tested (post-hoc global temperature, covariate-indexed global utility-scale
heterogeneity, the alternative-link experiment's three families, and now a
task-content-conditioned local temperature) that all agree: v14's plain
softmax on its own frozen utilities is not just calibrated on average, it
is locally optimal along every direction tested so far, including one
specifically chosen because it was the least explored.

## Not adopted; no submission made

`set_context_utility_network_v14` (1.143533 CV / 1.200 public) remains the
current best model, unchanged. This closes all three of this session's
pre-registered candidates; see the session summary in `AGENTS.md` /
`cleaning_log.md` for the consolidated conclusion and remaining open
directions.
