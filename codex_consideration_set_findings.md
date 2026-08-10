# Conjunctive price-screening consideration-set mixture -- findings (screen stage)

Pre-registration: `codex_consideration_set_preregister.md`. Implementation:
`R/codex_consideration_set_screen.R`.

## Result: decisive screen-stage reject

Correctness check passed exactly: at `pi_n = 0` for every respondent, the
panel objective reproduces v14's own frozen panel likelihood to numerical
tolerance (confirms the mixture machinery, gate, and panel aggregation are
implemented correctly before trusting any fitted result).

Optimization was well-behaved, not multimodal: **12 of 12 random restarts
converged to the same objective value** (20.10148, one restart landed at
20.10231, a negligible difference) -- ruling out the multimodality failure
mode this project's own earlier latent-class experiments specifically
warned about.

The fitted model is nonetheless **decisively worse out of sample**:

| | Fit-set logloss | Validation logloss |
|---|---|---|
| v14 baseline | 1.139710 | 1.158825 |
| Consideration-set mixture | (panel NLL 20.10, not directly comparable) | **1.194021** |

**Screen gain (v14 - mixture) = -0.035196** -- a large, unambiguous negative,
not a near-miss. Per the pre-registered screen protocol, this rejects
without any canonical or repeated CV compute spent.

Fitted parameters: threshold `a = 1.56` (price levels 1-12) at mean income,
`b = 0.29` (threshold rises with income, directionally sensible),
`alpha0 = -1.16`, `alpha1 = -0.13` (screening-type probability falls with
income, also directionally sensible), implying a mean training-respondent
screening probability of 24%.

## Interpretation

The fitted threshold (~1.56, near the very bottom of the 1-12 price range)
means the model resolves the training panel likelihood by declaring roughly
a quarter of training respondents "screeners" who would reject nearly every
inside bundle on price grounds alone -- an aggressive, low-generalizability
solution. This is consistent with a familiar failure mode already documented
in this project: any mechanism that lets the model infer a *stable,
person-specific* behavioral pattern from a respondent's repeated 19-task
sequence (mixed logit's random coefficients, latent-class membership, and
now this screening-type panel mixture) risks fitting patterns specific to
the *training* respondents that do not transfer, precisely because test
respondents are entirely new people (`AGENTS.md`'s core structural fact).
The panel likelihood here rewards the model for correctly identifying which
*training* respondents behave like screeners using their own 19 outcomes --
but at test time, `pi_n` can only be predicted from income, a single weak
covariate, so the model pays the full cost of an aggressive, low threshold
on newly-arriving respondents without the corresponding benefit of having
actually observed their behavior.

This does not by itself refute the noncompensatory-screening hypothesis in
general -- a more identified version (e.g., a fixed, literature-derived or
smaller, better-regularized threshold; a richer or different screening-type
predictor; a higher, less aggressive fixed floor on the threshold) might
behave differently. It was not pursued further this session, consistent
with the project's standing practice of not chasing increasingly flexible
variants of a mechanism that failed its first honest test without a
specific new reason to expect a different outcome.

## Not adopted; no CV compute spent, no submission made

`set_context_utility_network_v14` (1.143533 CV / 1.200 public) remains the
current best model, unchanged.
