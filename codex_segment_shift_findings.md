# Segment-distribution shift: targeted opt-out-margin pruning -- findings

Pre-registration: `codex_segment_shift_preregister.md`. Implementation:
`R/codex_segment_shift_cv.R` (canonical + test-like populations),
`R/codex_segment_shift_repeated_cv.R` (6-seed confirmation),
`R/codex_segment_shift_propagate.R` (full v14-ensemble propagation).

## Result: PROMOTED -- the first confirmed improvement of the session

### 1. A genuinely new structural fact

Respondent-level segment counts (`segmentind`, 6 levels):

| Segment | Train n (%) | Test n (%) |
|---|---|---|
| 1 | 128 (11.3%) | 57 (21.7%) |
| 2 | 383 (33.7%) | 9 (3.4%) |
| 3 | 49 (4.3%) | **84 (31.9%)** |
| 4 | 211 (18.6%) | 16 (6.1%) |
| 5 | 58 (5.1%) | **97 (36.9%)** |
| 6 | 306 (27.0%) | **0 (0.0%)** |

Segment 6 -- 27% of training respondents -- has **zero** test representation.
Segments 3 and 5, only 9.4% of training respondents combined, make up
**68.8% of test respondents**. Segment alone reaches adversarial-validation
AUC 0.898 (in-sample; inflated by segment 6's quasi-complete separation,
since a coefficient of -19.0 (SE 606) is fit for a level with zero test
rows to misclassify), dwarfing income alone (AUC 0.654, the previously
identified main driver). This had not been characterized as its own
structural fact before -- the one prior mention of these exact segment
counts (`cleaning_log.md`, 2026-07-27) was about training-set sparsity as a
reason to be cautious *adding* new segment-specific terms, not about the
train/test distribution mismatch in the terms already deployed since `mod8`.

### 2. Coefficient audit (m8trpg, full-data refit)

| Term | Estimate | SE | p-value | Segment's test share |
|---|---|---|---|---|
| `P_seg3` | 0.1148 | 0.0171 | <1e-10 | 31.9% |
| `P_seg5` | 0.1179 | 0.0164 | <1e-12 | 36.9% |
| `In_seg3` | 0.0440 | 0.1429 | **0.758** | 31.9% |
| `In_seg5` | 0.1118 | 0.1341 | **0.404** | 36.9% |
| `In_seg2` | -0.2062 | 0.0822 | 0.012 | 3.4% |
| `In_seg4` | -0.1966 | 0.0900 | 0.029 | 6.1% |
| `In_seg6` | -0.3028 | 0.0857 | 0.0004 | 0.0% |

A striking split: segments 3/5's *price-sensitivity* deviations are large,
precise, and highly significant (no evidence of unreliability there). But
their *opt-out-margin* deviations (`In_seg3`, `In_seg5`) are statistically
indistinguishable from zero on the full 1,135-respondent training set --
and these are exactly the two segments whose loss-function weight explodes
from ~9% (training, where CV is computed) to ~69% (test, where the actual
score is computed). The other three segments' `In_seg` terms are all
individually significant, but govern segments whose test share is shrinking
or vanishing.

### 3. The fix: drop `In_seg3` and `In_seg5`

Candidate: `m8trpg` with only these two already-non-significant terms
removed (`In_seg2`, `In_seg4`, `In_seg6` untouched; all `P_seg2-6`
untouched). Evaluated via the canonical respondent-grouped 5-fold CV (seed
4821, same partition as every other component), on three populations:

| Population | n | Point gain | 95% CI | Win rate |
|---|---|---|---|---|
| All respondents | 1135 | +0.000504 | **[0.000081, 0.000966]** | 99.1% |
| Top 50% test-like | 568 | +0.000991 | [0.000154, 0.001900] | 99.0% |
| Top 30% test-like | 341 | +0.001777 | [0.000389, 0.003280] | 99.5% |

The gain is significant on the **whole population already** -- not merely a
test-like-subgroup effect -- and amplifies monotonically as the population
is restricted toward test-likeness (0.0005 -> 0.001 -> 0.0018), exactly as
the mechanistic hypothesis predicts. Test-like propensity computed via the
project's standard adversarial-validation classifier (no choice outcomes
used). Segment composition of the top-30%-test-like training respondents:
segments 1/3/4/5 all represented (128/49/104/58 of 341), segment 2 nearly
absent (2 of 341) -- consistent with segment 2 being one of the most
"non-test-like" segments per the raw counts above.

### 4. Repeated CV (6 seeds): confirmed

| Seed | Baseline (m8trpg) | Candidate | Gain |
|---|---|---|---|
| 4821 (canonical) | 1.147021 | 1.146517 | 0.000504 |
| 1907 | 1.145363 | 1.145147 | 0.000216 |
| 2719 | 1.145845 | 1.145696 | 0.000149 |
| 6151 | 1.146078 | 1.145904 | 0.000174 |
| 8293 | 1.147074 | 1.146635 | 0.000439 |
| 104729 | 1.144116 | 1.143721 | 0.000395 |

**6 of 6 positive.** Pooled (all respondents): gain +0.000313, 95% CI
**[0.0000783, 0.000584]**, win rate 99.6%. Top-30%-test-like: gain
+0.001033, CI **[0.000254, 0.001927]**. Both exclude zero. **PROMOTE** per
the pre-registered rule.

### 5. Propagated through the full v14 ensemble: confirmed again

Reconstructed the exact v14 pipeline (0.80/0.20 mlogit/xgb -> 0.85/0.15
+shallow MLP -> per-fold set-context blend weights from
`crossfit_blend()`'s own grid search, **not** the single fixed 0.111 used
only in the separate full-data test-submission build) -- verified against
the official OOF cache first (`max|reconstructed - official| = 3.7e-9`)
before trusting anything.

| Seed | Original v14-stack | Pruned v14-stack | Gain |
|---|---|---|---|
| 4821 (canonical) | 1.143533 | **1.143255** | 0.000279 |
| 1907 | 1.140296 | 1.140163 | 0.000133 |
| 2719 | 1.141606 | 1.141492 | 0.000114 |
| 6151 | 1.141472 | 1.141373 | 0.000099 |
| 8293 | 1.141557 | 1.141329 | 0.000228 |
| 104729 | 1.139426 | 1.139214 | 0.000212 |

**6 of 6 positive at the full-ensemble level too.** Pooled gain +0.0001776,
95% CI **[0.0000441, 0.0003292]**, win rate 99.6%. **PROMOTE for a full-data
build audit.**

## Why this is not a duplicate

- Not `mod11`'s pruning (different terms entirely: `P_income`, `P_night`,
  `In_income`, `In_age`, `In_miles`, `In_night`; segment terms untouched
  there, and that test never checked test-like-reweighted performance).
- Not the earlier segment-sparsity caution (about *adding* new
  segment-specific covariate slopes, not removing an already-non-significant
  deployed term).
- Not the already-rejected importance-weighted refit (a global density-ratio
  reweighting of the entire likelihood, decisively harmful). This targets
  exactly two already-statistically-weak parameters with a specific
  mechanical argument (loss-weight amplification under a confirmed
  distribution shift), not a global reweighting.
- Distinguishes cleanly from every one of today's ten rejected mechanisms:
  this is neither personalization-that-can't-transfer, nor
  recalibration-of-an-already-calibrated-output, nor added flexibility
  without new information -- it *removes* two parameters using a concrete,
  quantified, previously-uncharacterized structural fact about the data.

## Next step

Per the pre-registration, a promoted candidate still requires a full-data
build (retrain every component on all 1,135 respondents with `In_seg3`/
`In_seg5` dropped from the mlogit formula) with the project's established
MD5-lock and two-independent-run reproducibility check before any
submission is considered -- not yet done as of this writing.
