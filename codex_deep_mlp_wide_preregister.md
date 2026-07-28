# Pre-registration: wider deep-MLP architecture search

This registry was fixed before running any of the new architecture
configurations. The search starts from `zhenhao` commit `5a8d601` and uses the
exact `build_mlp_matrix()` feature representation already used by the submitted
shallow MLP and the first five-config torch experiment.

## Candidate family

Exactly 24 configurations are tested: the Cartesian product of six hidden-layer
layouts and four training recipes.

Hidden-layer layouts:

1. `32-16`
2. `64-32`
3. `128-64`
4. `256-128`
5. `128-64-32`
6. `256-128-64`

Training recipes:

| Recipe | Dropout | Weight decay | Learning rate | Epochs |
|---|---:|---:|---:|---:|
| A | 0.15 | 0.0001 | 0.0005 | 16 |
| B | 0.30 | 0.0010 | 0.0010 | 12 |
| C | 0.45 | 0.0020 | 0.0010 | 12 |
| D | 0.30 | 0.0005 | 0.0020 | 8 |

Batch size is fixed at 256. Recipe B with layout `128-64` exactly reproduces
the previously selected deep architecture and acts as the anchor; the other 23
configurations are new.

## Selection and validation

- Screen all 24 configurations on the canonical seed-7402 respondent split.
- Average exactly two seeds (`13101`, `13102`) for each screen configuration.
- For each configuration, fit arithmetic weights across `v11`, the submitted
  shallow MLP, and the deep MLP on the screen split.
- Freeze the configuration with the lowest three-way screen log loss. If no
  configuration improves the fixed 15%-shallow-MLP baseline, stop.
- Confirm only that frozen configuration using canonical respondent-grouped
  five-fold CV (seed 4821), averaging exactly three seeds per fold.
- In each outer fold, learn the three arithmetic weights from the other four
  folds only. The primary endpoint is this fold-cross-fitted three-way blend
  versus the exact fixed-15% submitted OOF baseline (`1.143686618134879`).

## Multiplicity and decision rule

The architecture family size is fixed at 24. The primary respondent-clustered
bootstrap uses 100,000 resamples and reports ordinary 95%, 99%, and
Bonferroni-family-wise 95% intervals with alpha `0.05 / 24`.

A submission candidate requires all of:

1. positive point gain over the exact fixed-15% submitted baseline;
2. Bonferroni-adjusted lower bound above zero;
3. at least four of five folds improving;
4. no single fold with a loss regression larger than the total pooled gain.

Seed-bagging is run only if the frozen architecture has a positive ordinary
95% lower bound and improves at least four folds. The architecture, optimizer,
epochs, and blending procedure remain frozen during any bagging follow-up.
