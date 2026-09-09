# reference and margin state sampled direct-margin domain/window readout

File-only analysis of the delivered CPU-safe binding content trade predeclared predictions margin ladder. Positive margin values mean the compact-view arm has a larger correct-vs-best-distractor margin than the matched repeat arm on the sampled rows.

Input rows: 3480 (2160 Entity, 1320 EWoK).

## Entity V-R margin split by operation group

The official prediction split showed zero-operation losses and nonzero-operation gains at MAX. The direct-margin sample shows the same direction mainly at late checkpoints, but not a monotone clean law across all checkpoints.

| window | dose | all mean | zero-op mean | nonzero mean | nonzero-zero gap | frac positive all |
|---|---|---:|---:|---:|---:|---:|
| common10_80 | dose1 | +0.4197 | -0.4345 | +0.5906 | +1.0251 | 0.4958 |
| common10_80 | dose1p82 | +0.3148 | -0.3868 | +0.4551 | +0.8419 | 0.5236 |
| common10_80 | dose2p64 | +0.3605 | -1.9462 | +0.8218 | +2.7680 | 0.5194 |
| late70_80 | dose1 | +0.7982 | -0.8469 | +1.1273 | +1.9742 | 0.5111 |
| late70_80 | dose1p82 | +0.0481 | -1.8586 | +0.4294 | +2.2880 | 0.5056 |
| late70_80 | dose2p64 | +0.5007 | -3.0650 | +1.2138 | +4.2788 | 0.5333 |
| endpoint80 | dose1 | +0.8673 | -0.1729 | +1.0754 | +1.2483 | 0.5111 |
| endpoint80 | dose1p82 | +0.2796 | -1.4723 | +0.6300 | +2.1022 | 0.5333 |
| endpoint80 | dose2p64 | +0.9792 | -2.6744 | +1.7100 | +4.3844 | 0.5556 |

Entity dose slopes over rho (mean of checkpoint means):

| window | subgroup | slope/rho | R2 | MAX-minus-1x |
|---|---|---:|---:|---:|
| common10_80 | all | -0.8516 | 0.3165 | -0.0592 |
| common10_80 | zero_ops | -21.7421 | 0.7262 | -1.5116 |
| common10_80 | nonzero_ops | +3.3265 | 0.3887 | +0.2313 |
| late70_80 | all | -4.2817 | 0.1553 | -0.2976 |
| late70_80 | zero_ops | -31.9058 | 0.9974 | -2.2181 |
| late70_80 | nonzero_ops | +1.2431 | 0.0101 | +0.0865 |
| endpoint80 | all | +1.6084 | 0.0221 | +0.1119 |
| endpoint80 | zero_ops | -35.9826 | 0.9995 | -2.5015 |
| endpoint80 | nonzero_ops | +9.1266 | 0.3416 | +0.6346 |

## EWoK V-R margin by domain

EWoK sampled margins are domain-heterogeneous. The all-domain aggregate does not currently provide a stable broad mechanism by itself.

| window | dose | all-domain mean | frac positive |
|---|---|---:|---:|
| common10_80 | dose1 | +0.0599 | 0.5045 |
| common10_80 | dose1p82 | -0.2117 | 0.4909 |
| common10_80 | dose2p64 | +0.0090 | 0.5114 |
| late70_80 | dose1 | +0.1919 | 0.6000 |
| late70_80 | dose1p82 | -0.0127 | 0.4909 |
| late70_80 | dose2p64 | -0.3209 | 0.5455 |
| endpoint80 | dose1 | +0.3385 | 0.6000 |
| endpoint80 | dose1p82 | +0.0111 | 0.4545 |
| endpoint80 | dose2p64 | -0.2960 | 0.5273 |

Largest absolute MAX EWoK domain means in common10_80:

| domain subgroup | MAX mean | dose1 mean | MAX-minus-1x | n |
|---|---:|---:|---:|---:|
| social-relations | +0.6773 | +1.3797 | -0.7025 | 40 |
| spatial-relations | +0.5723 | -0.5614 | +1.1337 | 40 |
| quantitative-properties | -0.5539 | -0.2712 | -0.2827 | 40 |
| physical-interactions | -0.5434 | -0.1370 | -0.4065 | 40 |
| agent-properties | -0.1950 | +1.2885 | -1.4834 | 40 |
| social-properties | +0.1771 | -0.0717 | +0.2489 | 40 |
| physical-dynamics | -0.1392 | -0.8734 | +0.7341 | 40 |
| material-dynamics | +0.0969 | +0.0753 | +0.0216 | 40 |

## Scientific reading

- Direct sampled margins support reading the Entity V-R effect as operation-sensitive: at MAX late70_80 and endpoint80, nonzero-operation margins exceed zero-operation margins by several log-likelihood units, matching the official prediction-stratum direction.
- The same direct margins are not a clean monotone dose law across the whole 10M-80M window; signs fluctuate by checkpoint and the sample has only 15 zero-op rows per checkpoint/dose. This strengthens the need to read broad V-B/V-C from official stable-family scores rather than promoting Entity alone.
- EWoK margins are domain-heterogeneous and do not presently supply an independent broad positive carrier for V-R; this keeps the incoming ex-Entity V-B breadth ladder decisive for the companion mechanism.

## Files

- checkpoint_summaries_csv: `experiments/archive/frontier_consolidation/data/margin_domain_window_readout/checkpoint_summaries.csv`
- window_summaries_csv: `experiments/archive/frontier_consolidation/data/margin_domain_window_readout/window_summaries.csv`
- checkpoint_slopes_csv: `experiments/archive/frontier_consolidation/data/margin_domain_window_readout/checkpoint_slopes.csv`
- window_slopes_csv: `experiments/archive/frontier_consolidation/data/margin_domain_window_readout/window_slopes.csv`
- summary_json: `experiments/archive/frontier_consolidation/data/margin_domain_window_readout/margin_domain_window_summary.json`
- summary_md: `research/documents/frontier_consolidation/data/margin_domain_window_readout/margin_domain_window_summary.md`
