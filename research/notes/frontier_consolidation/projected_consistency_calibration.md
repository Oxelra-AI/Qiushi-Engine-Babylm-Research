# projected consistency calibration — Projected/row-centered source-view consistency calibration (successor-route evidence)

CPU-only analysis with no training, official evaluation, corpus/tokenizer change or H100 work. The init-matched minfreq50 80M screen and its cheap evaluation remained pending; their outputs were not inspected.

## Scientific Motivation
successor route comparison while minfreq runs raw-cosine pair geometry looked saturated (token-mean 80M last-layer centered margin
0.8474, top1 1.0000; clean_80M already 0.8247). That framing risks wrongly closing the
source-view consistency route. projected consistency calibration asks a sharper question: is there residual pair-specific
structure that a **fixed-subspace, row-centered** consistency loss could strengthen, and at what
gradient scale, without full-vector forcing that successor route comparison while minfreq runs warned against?

Script: `experiments/archive/frontier_consolidation/scripts/projected_consistency_calibration.py`
Output: `experiments/archive/frontier_consolidation/data/projected_consistency_calibration/projected_consistency_calibration.{json,md}`
Grounded on the frozen 100M reinvest stream (SHA 3dd19f09...) and spatial repair route status legal tokenizer
(SHA 91b775...), route portfolio and intervention assets pair-span map, actual token-mean checkpoints 20M/80M/100M.

## Key measured facts
Two distinct regimes emerge from the projection + centering sweep (proj dims 32/64/128/480):

1. **Raw pooled span cosine** confirms successor route comparison while minfreq runs: modest pair-vs-shuffle margin that only grows
   slowly (d128 raw pair-shuffle margin 0.182 at 20M -> 0.299 at 80M -> 0.303 at 100M). This is
   the near-saturated global geometry.

2. **Row-centered residual** (subtract each row's hidden mean before pooling the span) is a
   different, training-strengthened signal:
   - d128 row_centered pair-shuffle margin: 0.613 (20M) -> 0.883 (80M) -> 0.887 (100M).
   - It keeps rising with exposure, unlike the raw margin. So residual pair alignment is NOT
     saturated by the current MLM objective.
   - But `same-row-other` margin > `shuffle` margin in row_centered mode (80M d128:
     same-row-other 0.973 vs shuffle 0.883; 100M 0.978 vs 0.887). That means after centering,
     a source residual aligns with ANY rewrite from the same row about as well as with its true
     paired rewrite. Pair-SPECIFIC residual structure is weak; the model mostly encodes a shared
     row/topic residual, not a per-proposition source<->rewrite invariant.

This is the actionable gap: a row-centered projected consistency loss that pushes true-pair
residual similarity ABOVE same-row-other similarity would target the exact abstraction the model
has not formed on its own, rather than re-forcing the already-high raw geometry.

## Gradient-scale calibration (for a future single-variable trainer, if selected)
- Full-vector row-normalized loss (successor route comparison while minfreq runs) had aux/MLM all-hidden L2 ratio ~0.093 at lambda=1 and
  suggested lambda ~0.536 for a 5% budget.
- projected consistency calibration shows the scale is projection- and centering-dependent:
  - raw d480 aux/MLM L2 ~0.089 (lambda@5% ~0.56) — essentially the successor route comparison while minfreq runs full-vector case.
  - row_centered d128 aux/MLM L2 ~0.55 (lambda@5% ~0.091); row_centered d64 ~0.76 (lambda@5% ~0.066).
  - Grad cosine with MLM stays ~0.002-0.007 for all modes: the auxiliary direction is nearly
    orthogonal to MLM, so a small-weight auxiliary term should not directly cancel MLM learning.
- Practical construction target: row_centered, moderate projection (d64-128), auxiliary weight
  chosen for ~3-5% all-hidden L2 budget (lambda ~0.05-0.09), positive-only, stop-gradient
  symmetric, changed-block rows only, everything else frozen at the validated legal16k token-mean
  compact-view recipe.

## Route status after projected consistency calibration
- Source-view consistency is NOT closed by successor route comparison while minfreq runs saturation. projected consistency calibration identifies a real,
  training-unsaturated, pair-nonspecific residual gap and a concrete low-risk construction
  (row-centered projected auxiliary loss with a calibrated small weight).
- This is still not BabyLM-score evidence. It ranks source-view consistency above sequence
  (which measured at only ~2.6% active-token gain vs fixed seq256) and above global word-mean
  (closed) as the most promising successor IF minfreq50 is weak/redistributive.
- Fractional credit remains a distinct but riskier optimization hypothesis (same credit-move
  direction as failed global word-mean).

## Decision rule carried forward
1. After minfreq50 scoring completes, judge minfreq50 against wordmean failure anatomy thresholds
   (needed cheap7 +~0.70 if SuperGLUE/AoA flat) and token-mean references (70M 42.6086 / 80M 42.9486).
2. If minfreq50 is broadly positive on BLiMP/Supplement/EWoK with Entity/GlobalPIQA preserved,
   continue it to 100M/full eval.
3. If minfreq50 is flat/redistributive, the strongest prepared successor is a row-centered
   projected source-view consistency screen using the consistency route stream validation and interface collate interface and
   projected consistency calibration lambda calibration — a single-variable auxiliary loss on the legal16k token-mean
   compact-view substrate. Do NOT auto-launch sequence.
