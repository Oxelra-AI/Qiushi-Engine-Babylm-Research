# causal interface trajectory: K-scaling test design

## Prediction

The matched coefficient for equal answer-only/full alternation in the query-first
orbit-binding task with K entities is:

    w* = 1 / (3K + 5)

For K=4: w* = 1/17 (established in Step021b as preserving).

The test: does w = 1/(3K+5) preserve the L1 query-match marker at other K values?

## K values and derived constants

| K | SL | IS_POS | n_ctx_pos | w* | ATTR_POS |
|---|---|---|---|---|---|
| 2 | 12 | 9 | 7 | 1/11 ≈ 0.0909 | [5, 8] |
| 3 | 15 | 12 | 10 | 1/14 ≈ 0.0714 | [5, 8, 11] |
| 4 | 18 | 15 | 13 | 1/17 ≈ 0.0588 | [5, 8, 11, 14] |
| 6 | 24 | 21 | 19 | 1/23 ≈ 0.0435 | [5, 8, 11, 14, 17, 20] |

SL = 3K + 6,  IS_POS = 3K + 3,  n_ctx_pos = 3K + 3,  ATTR_POS = [5 + 3i for i in range(K)]

## Experimental design

For each K, two seeds (e.g., 43, 100):

1. **Preparation**: query-first answer-only training until binding acquired
   (top-4 > 0.95, B-swap > 5.0) or max_epochs.
   Expected: K=2 faster, K=6 slower than K=4.

2. **Continuation** (100 epochs):
   - direct_full (w=1.0): expected to collapse
   - static w=1/(3K+5): predicted to preserve
   - static w=0.5: predicted to partially fail (too much context)

3. **Causal intervention**: learn L1 direction from preparation, test d_only
   and orth_only redirection for all continuation models.

4. **Metrics**: direction accuracy, redirect rate, redirect margin, held transfer.

## Implementation notes

- Requires parameterizing K throughout: SL, IS_POS, ATTR_POS, weight vector, probes.
- Model: d=64, nh=2, nl=3 (same for all K). If K=6 fails to bind, try d=128.
- Vocabulary: N_ENT=10, N_ATTR=12 accommodate K≤8.
- Entity/attribute sampling: sample K from TRAIN_E without replacement.
- Preparation epochs: start with 500 for K=2,3; 600 for K=4; 800 for K=6.
- Check binding every 50 epochs; save first bound checkpoint.

## Success criteria

If w=1/(3K+5) preserves the L1 direction (accuracy > 0.9, d_only redirect > 0.9)
while direct_full erases it (accuracy < 0.35, redirect near chance), the coefficient
scaling law is confirmed. If a different w is needed, the scaling can be refined.

## Files needed

New script: scripts/k_scaling.py (parameterized by K)
Data: data/k_scaling/
Note: notes/026_k_scaling.md
