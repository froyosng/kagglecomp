# Pre-registration: shared-alternative-utility MLP, CV round

Fixed on branch `zhenhao`, after `a60741a`, before any five-fold CV fit.
Screen already run (single split, seed 7402, `R/codex_shared_utility_mlp.R`
"screen" stage, results in
`data_processed/codex_shared_utility_mlp/shared_utility_mlp_screen.csv`):

| Config | Component alone | Incremental gain vs. current (0.85*v11+0.15*shallow) |
|---|---:|---:|
| `shared_64_32` | 1.249503 | +0.001724 (weight 0.13) |
| `shared_128_64` | 1.334371 | +0.000634 (weight 0.07) |
| `shared_32` | 1.181526 | +0.001568 (weight 0.22) |

All three configs are net-positive in the incremental blend despite the
component alone being far weaker than m8trpg (1.147021) or even rank:ndcg
xgboost's own screen number (1.193073) -- the exchangeability constraint
plus exact 4-way cross-entropy adds diversity the ensemble can use even
though the raw component is not competitive by itself, the same qualitative
pattern already seen for xgboost (never beats the logit alone) and the
shallow MLP (weak alone, real ensemble contribution).

## Frozen decision

Per this project's established screen-then-freeze rule (see
`R/codex_torch_deep_mlp.R`'s `stage == "cv"` block): the viable config with
the lowest incremental screen log loss is promoted, unmodified. That is
**`shared_64_32`** (hidden = 64-32, dropout 0.10, weight_decay 1e-4,
lr 1e-3, 60 epochs, batch = 256 tasks, 3 seeds `9401/9402/9403` averaged).
No further architecture search, no seed-count change, no hyperparameter
retuning before or during CV.

## CV design

- Canonical respondent-grouped five-fold CV (seed 4821, `oof_ensemble_v10.rds
  $fold_of_case`), refitting `shared_64_32` from scratch inside every fold
  (fresh `shared_utility_scaler`/one-hot construction from that fold's
  training respondents only -- attribute/price/segment/region/ppark level
  sets are a fixed structural constant of the design, computed once from the
  full `train.csv`, not fold-specific, matching how `Pr_lvl2:12` and
  `attr_max` are already treated elsewhere in this project).
- Comparison architecture, matching every other candidate this session:
  `incremental = (1-w)*current + w*shared_mlp`, with `current = 0.85*(0.8*
  mlogit+0.2*original_xgb)+0.15*shallow_mlp` (the exact submitted model),
  weight chosen via fold-cross-fitted grid search (0 to 0.40 by 0.01, fit on
  the other 4 folds only).
- If the canonical CV gain's ordinary 95% respondent-bootstrap CI excludes
  zero, proceed to the same repeated-CV design as the price-history round
  (seeds `1907, 2719, 6151, 8293, 104729`, reusing cached `original_xgb`/
  `shallow_mlp` components from `data_processed/codex_repeat_cv/checkpoints/`,
  refitting only `shared_64_32` and the plain m8trpg `mlogit` component for
  the `current` baseline, since the shared-utility net is entirely new and
  was never part of that earlier round's cached fits). If the canonical CV
  CI already crosses zero, report the canonical result as sufficient to
  reject and do not spend the additional compute on repeated CV.

## Promotion rule

Same standard as every other candidate this session: canonical CV gain > 0,
ordinary 95% bootstrap CI excludes zero; if repeated CV is triggered, promotion
also requires the family-adjusted lower bound > 0 and >=5/6 positive repeats.
