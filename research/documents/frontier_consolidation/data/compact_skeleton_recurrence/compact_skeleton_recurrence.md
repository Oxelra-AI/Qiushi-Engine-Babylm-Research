# source wide skeleton recurrence integrated compact-view skeleton recurrence measurement

This is a CPU text measurement only: no training, no model scoring, and no leaderboard action.

Pairs: `experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl` SHA `d2a3110c110e216180b632ce7a181acdf348b804ea19f172e16b5afeb0d8c9fc`
Selected subset: `experiments/archive/frontier_consolidation/data/reciprocal_causal_transfer_scaffold_dosematched_shuffle/selected_reciprocal_pairs.jsonl` SHA `d2729b6752561dac1483e8f7e34c71393de288fc8ed81537fce5e1c1cd6eedbd`

## Aggregate measurements
| subset | n | source words | view words | compact source-content coverage | repeat-prefix source-content coverage | compact-repeat coverage | pairs compact covers more | compact tail-content coverage | pairs with tail recovery | compact content fraction | repeat content fraction | compact-repeat content density | compact source Jaccard | repeat source Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_pairs | 12155 | 21.538708 | 13.303826 | 66.99% | 59.87% | 7.12% | 54.19% | 70.26% | 97.66% | 65.27% | 49.43% | 15.84% | 0.478008 | 0.624758 |
| reference_179_selected_subset | 6071 | 21.566793 | 13.313128 | 66.83% | 59.68% | 7.15% | 53.98% | 69.99% | 97.58% | 65.20% | 49.26% | 15.94% | 0.476289 | 0.623700 |

## Source-position deciles

### all_pairs
| decile | content positions | compact covered | repeat-prefix covered | compact-repeat |
|---:|---:|---:|---:|---:|
| 0 | 15178 | 68.15% | 100.00% | -31.85% |
| 1 | 12703 | 62.31% | 100.00% | -37.69% |
| 2 | 13625 | 61.64% | 100.00% | -38.36% |
| 3 | 12845 | 62.80% | 99.68% | -36.88% |
| 4 | 12100 | 63.16% | 94.57% | -31.41% |
| 5 | 13846 | 63.72% | 69.57% | -5.85% |
| 6 | 13301 | 63.60% | 31.69% | 31.91% |
| 7 | 12684 | 65.56% | 7.46% | 58.10% |
| 8 | 12898 | 70.62% | 0.11% | 70.51% |
| 9 | 15205 | 76.18% | 0.00% | 76.18% |

### reference_179_selected_subset
| decile | content positions | compact covered | repeat-prefix covered | compact-repeat |
|---:|---:|---:|---:|---:|
| 0 | 7563 | 68.62% | 100.00% | -31.38% |
| 1 | 6320 | 62.23% | 100.00% | -37.77% |
| 2 | 6795 | 60.63% | 100.00% | -39.37% |
| 3 | 6437 | 62.22% | 99.63% | -37.41% |
| 4 | 6045 | 63.41% | 94.39% | -30.98% |
| 5 | 6937 | 63.98% | 69.67% | -5.69% |
| 6 | 6680 | 63.05% | 31.68% | 31.38% |
| 7 | 6360 | 65.68% | 6.93% | 58.74% |
| 8 | 6437 | 70.27% | 0.14% | 70.13% |
| 9 | 7591 | 76.22% | 0.00% | 76.22% |

## Scientific reading
The compact view should not be treated as mainly a source-free non-copy paraphrase if it re-exposes source content from across the full source while the repeat arm only re-exposes the prefix. In that case, reciprocal multiview mechanism and scaffold's copied-token lift is not merely a nuisance: it is part of a data mechanism in which a short view repeats selected source-wide content keys in a denser context. The still-open question for real training is whether this source-wide skeleton recurrence, rather than generated paraphrase style or causal/MLM topology alone, carries the DeBERTa compact-view gain.

Figure: `experiments/archive/frontier_consolidation/figures/compact_skeleton_source_position_coverage.png`
Per-pair CSV: `experiments/archive/frontier_consolidation/data/compact_skeleton_recurrence/per_pair_skeleton_metrics.csv`
High-tail examples: `experiments/archive/frontier_consolidation/data/compact_skeleton_recurrence/high_tail_skeleton_examples.jsonl`
JSON: `experiments/archive/frontier_consolidation/data/compact_skeleton_recurrence/compact_skeleton_recurrence.json`
