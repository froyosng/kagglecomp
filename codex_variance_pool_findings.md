# Set-context variance pooling -- findings (screen stage)

Pre-registration: `codex_variance_pool_preregister.md`. Implementation:
`R/codex_variance_pool_screen.R`. Fifth and final piece of this session's
adversarial-modelling round, addressing the brief's item #5 (representation
gap in the set-context network).

## Result: screen-stage reject, no CV compute spent

Both architectures fit on the identical canonical single 80/20 respondent
split (`data_processed/train_val_split.rds`, seed 7402), identical
hyperparameters, identical training seed (22501):

| Architecture | Standalone val logloss | Best blend weight | Blended val logloss |
|---|---|---|---|
| Existing (mean + max pooling) | 1.335116 | 0.08 | 1.156997 |
| New (mean + max + variance pooling) | **1.371673** (worse) | 0.06 | **1.157392** (worse) |

Screen gain (existing minus variance) = **-0.000395** -- the new
architecture is worse both standalone and blended. Per the pre-registered
screen protocol ("promote to the full canonical + repeated CV protocol only
if the new architecture's blended validation log loss beats the existing
architecture's on this split; otherwise log as a screen-stage reject and
stop"), this rejects at the screen stage. No canonical or repeated CV was
run, avoiding the substantial `torch` retraining cost (full 5-fold x
3-seed x 50-epoch CV) that a promotion would have required.

## Interpretation

Adding one broadcast pooling statistic (per-embedding-coordinate variance
across the three inside alternatives) to the architecture, with the encoder
and head otherwise completely unchanged, does not help and mildly hurts
even in-sample-adjacent training behavior (training loss did improve
slightly, 0.8627 vs 0.8777, but validation loss got worse in both the
standalone and blended comparison -- a classic sign of the extra capacity
finding noise rather than signal on the training split, consistent with
the pattern already seen when the choice-set-geometry hand-built version of
similar information near-missed rather than clearly helped). This is a
clean, cheap, single-screen rejection, not a near-miss requiring further
investigation.

## Not adopted; no further compute spent

Combined with the already-near-missed choice-set-geometry features (raw
attribute-space similarity, additive to mlogit utility) and the closed
sequence-transformer rescue (cross-task history), this closes the
representation-gap question this session was able to test cheaply: neither
of the two representation gaps previously identified in the set-context
network's pooling (missing spread/variance information; the sequence/
history dimension separately closed in the transformer rescue) has produced
an improvement when concretely implemented and tested. `set_context_utility_
network_v14` (1.143533 CV / 1.200 public) remains the current best model,
unchanged.
