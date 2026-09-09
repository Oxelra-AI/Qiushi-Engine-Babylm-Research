# pair level relation robustness pair-level robustness of relation-cost signal

Rows are held-out compact rewrite pairs, not individual token positions. Only tokenizer-nonoverlap rewrite tokens are used; token rows for the same pair are averaged before arm contrasts. For each pair, `excess_true_cost = (true_source_NLL_A - true_source_NLL_B) - (unrelated_source_NLL_A - unrelated_source_NLL_B)`, so positive RminusC means REPEAT is worse than CLEAN in source-conditioned nonidentical use after the unrelated-source control is removed.

## Checkpoint-averaged pair distributions

| arch | seed | contrast | n pairs | mean excess | median excess | frac excess>0 | mean gain delta | median gain delta | frac gain<0 | true delta | unrelated delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D | 43022 | RminusC | 1465 | +0.785 | +0.607 | 0.661 | -0.785 | -0.607 | 0.661 | +0.484 | -0.301 |
| D | 43022 | VminusC | 1465 | -0.671 | -0.668 | 0.346 | +0.671 | +0.668 | 0.346 | -1.155 | -0.484 |
| D | 43022 | VminusR | 1465 | -1.456 | -1.199 | 0.227 | +1.456 | +1.199 | 0.227 | -1.639 | -0.183 |
| D | 43122 | RminusC | 1465 | +1.055 | +0.878 | 0.701 | -1.055 | -0.878 | 0.701 | +0.709 | -0.346 |
| D | 43122 | VminusC | 1465 | -0.889 | -0.772 | 0.315 | +0.889 | +0.772 | 0.315 | -1.352 | -0.463 |
| D | 43122 | VminusR | 1465 | -1.944 | -1.641 | 0.190 | +1.944 | +1.641 | 0.190 | -2.060 | -0.116 |
| RBT | 43022 | RminusC | 1465 | +0.396 | +0.254 | 0.659 | -0.396 | -0.254 | 0.659 | +0.497 | +0.102 |
| RBT | 43022 | VminusC | 1465 | -0.061 | -0.054 | 0.466 | +0.061 | +0.054 | 0.466 | -0.498 | -0.437 |
| RBT | 43022 | VminusR | 1465 | -0.457 | -0.329 | 0.299 | +0.457 | +0.329 | 0.299 | -0.995 | -0.538 |

## Reading

The REPEAT cost is not a token-position artifact. After averaging non-overlap tokens within each held-out rewrite pair and then across checkpoints, RminusC has positive excess true-source cost in DeBERTa seed43022, DeBERTa seed43122, and RoBERTa seed43022. The distribution is broad rather than universal: some pairs benefit from REPEAT, but about 66-70% of pairs show an excess source-use cost depending on architecture/seed. The mean remains larger than the median, so high-cost pairs contribute materially; the principle should be stated as a distributional installed tendency, not an every-example law.

VIEW's positive nonidentical source-use signal is also pair-broad in DeBERTa: VminusC has negative excess cost and positive gain on a majority of pairs in both seeds. In RoBERTa, VminusC is weak at pair level, matching the term decomposition: it separates from REPEAT but does not cleanly prove a strong VIEW-over-CLEAN conditioning benefit beyond broad compact-rewrite fit.

## Files

- Pair contrast rows: `experiments/archive/relation_learning/data/relation_decomposition/rewrite_pair_level_contrasts.csv`
- Checkpoint summary: `experiments/archive/relation_learning/data/relation_decomposition/rewrite_pair_level_by_checkpoint.csv`
- Checkpoint-averaged summary: `experiments/archive/relation_learning/data/relation_decomposition/rewrite_pair_level_late_summary.csv`
