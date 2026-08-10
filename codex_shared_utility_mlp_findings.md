# Shared-alternative-utility MLP: screens positive, fails canonical CV

Pre-registration: `codex_shared_utility_mlp_preregister.md`, committed at
`8ac57ef` before any five-fold CV fit. Implementation: `R/codex_shared_utility_mlp.R`.
Raw output: `data_processed/codex_shared_utility_mlp/`.

## The idea

A genuinely new combination not yet tried this session: a **weight-shared**
neural network utility function, applied identically to every alternative's
own feature row (including the opt-out's fixed all-zero profile), trained on
the **exact 4-way cross-entropy** (not a ranking loss, not naive per-row
binary renormalization). Earlier rounds tested this constraint only with a
tree function class (`codex_shared_utility_screen.R`'s custom xgboost
objective, which failed cold-start: 1.218569 vs. v11's 1.160568, "m8trpg's
hand-built features were always doing the real work") and tested this
function class (neural net) only without the exchangeability constraint (the
already-adopted flat shallow/deep MLP, which stacks all 4 alternatives'
features into one row per task). This script is the missing cell in that
2x2: shared constraint x flexible function class.

Feature treatment mirrors m8trpg's own established choices as closely as
possible (one-hot attributes/price/segment/region/ppark -- reusing the
project's confirmed price-as-factor and categorical-attribute findings --
standardized continuous income/age/miles/night/gender/urbanicity/education,
and the existing price-gap/is-cheapest/is-dearest/Task_c context features),
not an arbitrary new encoding, so any result is attributable to the
architecture, not a weaker feature set.

## Screen (single split, seed 7402)

| Config | Component alone | Incremental gain vs. current |
|---|---:|---:|
| `shared_64_32` | 1.249503 | +0.001724 (weight 0.13) |
| `shared_128_64` | 1.334371 | +0.000634 (weight 0.07) |
| `shared_32` | 1.181526 | +0.001568 (weight 0.22) |

All three clear the project's screen-then-freeze gate despite the component
alone being far weaker than m8trpg (1.147021) or even rank:ndcg xgboost's own
screen number (1.193073) -- the same qualitative pattern already seen for
xgboost and the shallow MLP (weak alone, real diversity in a blend). Frozen
winner per the established rule (lowest incremental screen log loss):
`shared_64_32` (hidden 64-32, dropout 0.10, weight_decay 1e-4, lr 1e-3, 60
epochs, batch = 256 tasks, 3 seeds averaged) -- no further tuning.

Two real implementation bugs were caught by smoke-testing before the full
screen run, not left for a full CV round to surface: an assertion comparing
per-task prediction rows against the per-row `x_valid` count, and
checkpointing a raw torch `state_dict` (crashes with "external pointer is not
valid" when reloaded in a fresh `Rscript` process) instead of the plain
prediction matrix that `R/codex_torch_deep_mlp.R`'s own `fit_torch_once`
pattern already uses. Both fixed and re-verified via smoke test before any
real run.

## Canonical five-fold CV (seed 4821)

| Fold | Component alone | Blend weight |
|---:|---:|---:|
| 1 | 1.334196 | 0.09 |
| 2 | 1.241706 | 0.07 |
| 3 | 1.229595 | 0.04 |
| 4 | 1.209969 | 0.06 |
| 5 | 1.220631 | 0.05 |
| **Pooled** | **1.247219** | -- |

Fold-cross-fitted incremental blend: current (1.143686618) -> **1.143543142**,
gain **+0.000143**, respondent-bootstrap 95% CI **[-0.000567, +0.000850]**
(99% CI [-0.000782, +0.001078]), win rate 65.5%.

**This is a much weaker result than the single-split screen suggested**
(+0.001724 -> +0.000143, an 8x drop) and the fold-cross-fitted blend weights
(0.04-0.09) are notably smaller than the screen's single-split weight (0.13)
-- consistent with the screen's single 80/20 split overstating how much this
component helps once genuinely evaluated across 5 independent held-out
groups instead of one.

## Decision

Per the pre-registered stopping rule ("if the canonical CV CI already
crosses zero, report the canonical result as sufficient to reject and do
not spend the additional compute on repeated CV"): the 95% CI crosses zero
comfortably, not narrowly -- unlike the price-history near-miss earlier this
session (CI [-0.000090, +0.001564], barely missing), this interval is wide
and centered close to zero relative to its own width. **Repeated CV was not
run; this canonical result is treated as sufficient to reject.**

## Interpretation

The shared-alternative-utility constraint plus exact cross-entropy plus a
neural function class is a real, well-motivated combination that had not
been tried, and it does add a small amount of genuine ensemble diversity (a
component alone worse than m8trpg still contributes some signal in a blend,
same pattern as xgboost and the shallow MLP). But the magnitude that survives
honest cross-validation (+0.0001) is an order of magnitude smaller than the
single-split screen implied, and far too small to distinguish from noise at
this project's ~1,135-respondent sample size. This closes the "shared-utility
objective" direction the brief specifically asked to investigate: the
earlier tree-based cold-start failure and this neural cold-start result now
agree from two different function classes that the loss function/constraint
was never the missing lever -- m8trpg's hand-built feature/interaction
structure is what does the real work in this dataset, not the learner's
functional form.

**Not adopted; no submission made.**

## Reproduction

```powershell
& 'C:\Program Files\R\R-4.6.0\bin\Rscript.exe' R/codex_shared_utility_mlp.R
$env:CODEX_SHARED_MLP_STAGE = "cv"
& 'C:\Program Files\R\R-4.6.0\bin\Rscript.exe' R/codex_shared_utility_mlp.R
```
