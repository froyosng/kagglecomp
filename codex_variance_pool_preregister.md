# Set-context variance pooling -- pre-registration (screen stage)

Date: 2026-07-31. Fifth and final piece of this session's adversarial-
modelling round, addressing the brief's item #5 (representation gap in the
set-context network) after items #1-#4 (hurdle model, three alternative-link
families, task-content-conditioned temperature, difficulty-predictability
diagnostic) were independently pre-registered/implemented/logged and all
rejected or closed as non-exploitable.

## Hypothesis

`R/codex_set_context_network_v2.R`'s encoder pools each task's three inside
alternatives' learned embeddings via **mean** and **max** only (verified by
direct code read of `set_context_net$forward()`), broadcasting both back to
every alternative slot alongside the alternative's own embedding and
`embedding - mean`. Neither statistic captures the **spread/variance across
alternatives** -- how similar or different the three bundles are to each
other in embedding space. Adding an explicit per-embedding-dimension
variance-across-alternatives pooling term, broadcast the same way mean/max
already are, will let the network use choice-set-similarity information the
current architecture cannot represent from mean+max alone, and will show an
improvement over the existing architecture even in a cheap single-split
screen.

## Why this is not a duplicate

- **Not the sequence-transformer rescue** (closed 2026-07-31): that concerned
  cross-task respondent history, not within-task alternative pooling.
- **Not simply enlarging the network**: `encoder_hidden`, `embedding_dim`,
  and `head_hidden` are left completely unchanged; the only change is one
  additional broadcast pooling statistic (variance) computed from the
  already-existing per-alternative embeddings, adding exactly one
  `embedding_dim`-sized block to the head's input (152 -> 184 for the
  existing config) and a correspondingly small number of extra first-layer
  weights. No new encoder or head layer, no additional hidden units.
- **Not the choice-set-geometry experiment** (near-miss, crosses zero,
  `R/codex_choice_set_geometry.R`): that hand-built pairwise
  similarity/distance features in the *raw attribute/price space* as
  additive terms to the mlogit utility. This experiment adds a pooling
  statistic in *learned embedding space* to a neural architecture, a
  different representation and a different model family -- but it is
  acknowledged as the closest existing near-cousin, which is exactly why
  this candidate was ranked third (lowest of the three original candidates)
  and is only pursued now as a screen, not committed to the full expensive
  nested-CV + repeated-CV protocol without evidence from a cheap first look.

## Minimum viable implementation and screening discipline

Because retraining this network is materially more expensive than every
other candidate this session (`torch`, full 5-fold x 3-seed x 50-epoch CV
for the canonical seed alone in the original build), this candidate is
**screened first** on the project's existing canonical single 80/20
respondent split (`data_processed/train_val_split.rds`, seed 7402) before
any commitment to the full nested-CV protocol -- explicitly the
"screen-first" discipline this project already uses elsewhere ("single-split
screening first (fast), CV confirmation before trusting a result").

Architecture change (`set_context_net_variance` module, otherwise identical
config to `network_config` in `codex_set_context_network_v2.R`):
`var_inside = Var(embedding[,1:3,])` across the alternative dimension
(population variance per embedding coordinate, one value per task per
embedding dimension), broadcast to all 4 alternative slots exactly as
`mean_repeated`/`max_repeated` already are, and concatenated into the head's
input alongside the existing four blocks.

Screen protocol: fit both the existing (mean+max) architecture and the new
(mean+max+var) architecture on the identical single split, same seeds, same
epochs, same everything else; compare (a) each component's standalone
validation log loss, and (b) each component's best-blend-weight validation
log loss against the same frozen flat-ensemble baseline used throughout this
project. Promote to the full canonical + repeated CV protocol only if the
new architecture's blended validation log loss beats the existing
architecture's on this split; otherwise log as a screen-stage reject and
stop (no CV compute spent).

## What would falsify this at the screen stage

The variance-pooling architecture's best-blend validation log loss failing
to beat the existing architecture's best-blend validation log loss on the
same split, same seeds.
