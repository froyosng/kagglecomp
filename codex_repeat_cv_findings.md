# Repeated-CV confirmation and eight-component submission audit

## Verdict

**Neither near-miss is promoted, and no Kaggle submission was made.**

Both candidates improved on every one of the six pre-registered five-fold
respondent splits, so the favorable direction is not an artifact of the
canonical seed alone. However, averaging each respondent's gain across the six
splits and then resampling respondents shows that neither gain is statistically
confirmed:

- `history_both_k3`: mean gain `+0.000573818`; ordinary 95% CI
  `[-0.000239460, +0.001381368]`; family-13 adjusted lower bound
  `-0.000621118`.
- Eight-component arithmetic ensemble: mean gain `+0.001169419`; ordinary 95%
  CI `[-0.000199100, +0.002512944]`; family-6 adjusted lower bound
  `-0.000689067`.

The eight-component result is still the stronger of the two, but repeated CV
turns its canonical-split ordinary CI—which barely excluded zero—into an
ordinary CI that crosses zero. Under the decision rule committed before the new
split results were observed (at least 5/6 positive repeats and an adjusted lower
bound above zero), both candidates fail promotion.

## Repeated-CV design

The five additional split seeds were frozen in
`codex_repeated_cv_preregister.md`: `1907`, `2719`, `6151`, `8293`, and
`104729`, alongside canonical seed `4821`.

Each additional repeat is a genuine respondent-grouped five-fold refit of the
models used by the candidate, not a relabeling or reslicing of canonical OOF
predictions. For the eight-component candidate, blend weights are selected
without the held-out fold using the same fold-cross-fitted procedure as the
original result. For inference, the six gains are averaged within each of the
1,135 respondents before the established 100,000-replicate respondent
bootstrap. The 30 folds are therefore not incorrectly treated as independent
observations.

| Candidate | Split seed | Baseline | Candidate | Gain |
|---|---:|---:|---:|---:|
| history | 4821 | 1.143686618 | 1.142951450 | +0.000735168 |
| history | 1907 | 1.141815179 | 1.141377795 | +0.000437385 |
| history | 2719 | 1.142460203 | 1.141802310 | +0.000657893 |
| history | 6151 | 1.142651190 | 1.142026151 | +0.000625039 |
| history | 8293 | 1.143129118 | 1.142606217 | +0.000522901 |
| history | 104729 | 1.141130306 | 1.140665785 | +0.000464522 |
| eight-component | 4821 | 1.143789442 | 1.142111910 | +0.001677532 |
| eight-component | 1907 | 1.141910645 | 1.140498513 | +0.001412132 |
| eight-component | 2719 | 1.142559842 | 1.141991161 | +0.000568680 |
| eight-component | 6151 | 1.142771719 | 1.141977201 | +0.000794518 |
| eight-component | 8293 | 1.143079844 | 1.141274547 | +0.001805297 |
| eight-component | 104729 | 1.141263617 | 1.140505265 | +0.000758352 |

History improved in 24/30 individual folds; the eight-component blend improved
in 22/30. The more relevant pre-registered repeat-level consistency check is
6/6 positive for both.

## Eight-component submission build and outlier audit

`submission_codex_8component_candidate.csv` is a valid, reproducible submission
file with 4,997 rows, exact sample-submission IDs and column order, finite
strictly positive probabilities, and row sums equal to one. It has **not** been
submitted.

The full-data component weights were frozen from the canonical cross-fitted
mean before the repeated results were read:

| Component | Weight |
|---|---:|
| m8trpg mlogit | 0.062469 |
| original xgboost | 0.004745 |
| rank:ndcg xgboost | 0.093305 |
| retuned xgboost | 0.015559 |
| glmnet-Cox | 0.131120 |
| shallow MLP | 0.080085 |
| triple-interaction mlogit | 0.479258 |
| deep MLP | 0.133459 |

The requested outlier audit gives a second reason not to recommend the file:

- Versus the currently submitted shallow-MLP blend, the largest probability
  change is `0.362630`, at `No = 22637`.
- Across all test rows, 159 rows have a maximum class-probability change above
  0.05, 50 above 0.10, and 30 above 0.15.
- `No = 22637` belongs to the known extreme-income respondent (Case 1192,
  `incomea = 3,800,000`). All 19 of that respondent's rows change by more than
  0.05, 17 by more than 0.10, and 12 by more than 0.15.
- The triple-interaction mlogit has 47.9% ensemble weight and differs from the
  current submission by as much as `0.683103` on the maximum-change row. Its
  weighted contribution accounts for about `0.327383` of the total maximum
  shift.

This is precisely the transfer-risk pattern that the earlier
triple-interaction audit warned about. A candidate whose repeated ordinary CI
already crosses zero should not be risked when its largest test-set changes are
concentrated on an extrapolative respondent.

## Joint history plus triple candidate

The one pre-registered joint candidate—`both_k3` history priors plus
`Price x z(income) x z(mileage)` on the same mlogit component—failed its
single-split gate:

- Plain triple model: `1.157575921`
- Joint model: `1.157991764`
- Gain versus plain triple: `-0.000415843` (worse)

Per the frozen screen-then-CV rule, full five-fold CV was not run and no terms
were removed post hoc.

## Stretch bagging item

The requested respondent-bootstrap bagging of the full m8trpg model was not
rerun because it is already a logged completed negative experiment: 15
respondent-bootstrap fits per canonical fold worsened the ensemble from
`1.145094` to `1.145658`. Repeating a logged harmful candidate would violate the
project instruction not to retry exhausted ideas.

## Reproduction

- `R/codex_repeated_cv.R`: five new genuine repeated-CV refits and pooled
  respondent inference.
- `R/codex_history_triple_joint.R`: frozen joint screen and conditional CV.
- `R/codex_submit_8component.R`: full-data component fits, candidate CSV, and
  outlier diagnostics.
- `R/codex_repeat_cv_audit.R`: read-only independent artifact audit. It
  recomputes saved OOF losses and respondent gains, validates split balance and
  probability matrices, reproduces saved bootstrap summaries, reconstructs the
  candidate CSV from its components, and verifies the No 22637 sensitivity.

Run the final audit with:

```powershell
& 'C:\Program Files\R\R-4.6.0\bin\Rscript.exe' 'R/codex_repeat_cv_audit.R'
```

Expected final line: `Repeated-CV and submission artifact audit passed.`
