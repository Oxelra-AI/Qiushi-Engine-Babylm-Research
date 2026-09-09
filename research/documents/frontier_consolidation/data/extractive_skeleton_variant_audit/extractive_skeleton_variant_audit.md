# source wide skeleton recurrence integrated extractive skeleton variant audit

No training/evaluation. This audits exact-length source-derived variants as possible dissection of the compact-view data mechanism.

## Aggregate geometry
| variant | n | source-content coverage | tail-content coverage | pairs with tail recovery | content fraction | Jaccard with compact | Jaccard with source | selected tail fraction | Δ source coverage vs prefix | Δ tail coverage vs compact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 12155 | 66.99% | 70.26% | 97.66% | 65.27% | 1.000000 | 0.478008 | 40.98% | 5.26% | 0.00% |
| prefix_repeat | 12155 | 61.73% | 4.85% | 17.66% | 49.43% | 0.359182 | 0.624758 | 0.00% | 0.00% | -65.40% |
| spread_even | 12155 | 64.69% | 68.08% | 99.09% | 52.17% | 0.383088 | 0.624755 | 37.45% | 2.96% | -2.17% |
| content_spread | 12155 | 92.72% | 92.16% | 100.00% | 75.96% | 0.442697 | 0.625525 | 30.52% | 30.99% | 21.90% |
| scored_source_skeleton | 12155 | 98.38% | 99.89% | 100.00% | 81.76% | 0.478539 | 0.625366 | 46.53% | 36.65% | 29.63% |
| oracle_compact_projection | 12155 | 70.04% | 67.03% | 96.86% | 57.68% | 0.702342 | 0.625705 | 35.57% | 8.31% | -3.23% |

## Source-position decile coverage

### compact
| decile | content positions | covered |
|---:|---:|---:|
| 0 | 15178 | 68.15% |
| 1 | 12703 | 62.31% |
| 2 | 13625 | 61.64% |
| 3 | 12845 | 62.80% |
| 4 | 12100 | 63.16% |
| 5 | 13846 | 63.72% |
| 6 | 13301 | 63.60% |
| 7 | 12684 | 65.56% |
| 8 | 12898 | 70.62% |
| 9 | 15205 | 76.18% |

### prefix_repeat
| decile | content positions | covered |
|---:|---:|---:|
| 0 | 15178 | 100.00% |
| 1 | 12703 | 100.00% |
| 2 | 13625 | 100.00% |
| 3 | 12845 | 99.70% |
| 4 | 12100 | 94.83% |
| 5 | 13846 | 70.97% |
| 6 | 13301 | 35.16% |
| 7 | 12684 | 12.14% |
| 8 | 12898 | 5.68% |
| 9 | 15205 | 5.85% |

### spread_even
| decile | content positions | covered |
|---:|---:|---:|
| 0 | 15178 | 62.90% |
| 1 | 12703 | 63.26% |
| 2 | 13625 | 62.05% |
| 3 | 12845 | 60.87% |
| 4 | 12100 | 61.60% |
| 5 | 13846 | 62.73% |
| 6 | 13301 | 61.99% |
| 7 | 12684 | 61.94% |
| 8 | 12898 | 59.20% |
| 9 | 15205 | 83.12% |

### content_spread
| decile | content positions | covered |
|---:|---:|---:|
| 0 | 15178 | 99.96% |
| 1 | 12703 | 97.09% |
| 2 | 13625 | 91.27% |
| 3 | 12845 | 89.99% |
| 4 | 12100 | 87.06% |
| 5 | 13846 | 86.23% |
| 6 | 13301 | 89.32% |
| 7 | 12684 | 87.48% |
| 8 | 12898 | 88.34% |
| 9 | 15205 | 97.59% |

### scored_source_skeleton
| decile | content positions | covered |
|---:|---:|---:|
| 0 | 15178 | 89.66% |
| 1 | 12703 | 95.65% |
| 2 | 13625 | 98.30% |
| 3 | 12845 | 99.37% |
| 4 | 12100 | 99.62% |
| 5 | 13846 | 99.66% |
| 6 | 13301 | 99.73% |
| 7 | 12684 | 99.91% |
| 8 | 12898 | 99.93% |
| 9 | 15205 | 100.00% |

### oracle_compact_projection
| decile | content positions | covered |
|---:|---:|---:|
| 0 | 15178 | 79.27% |
| 1 | 12703 | 73.30% |
| 2 | 13625 | 69.89% |
| 3 | 12845 | 68.07% |
| 4 | 12100 | 66.90% |
| 5 | 13846 | 65.89% |
| 6 | 13301 | 64.98% |
| 7 | 12684 | 65.93% |
| 8 | 12898 | 68.86% |
| 9 | 15205 | 65.69% |

## Scientific reading
If a source-only variant approaches compact tail coverage and content density while remaining exact-length, then a future MLM training dissection can separate source-wide skeleton recurrence from generated paraphrase style. If only the oracle projection approaches compact, then the compact generator is doing nontrivial content selection. If source-only variants recover tail content but look distributionally unnatural, the lowest-cost future test should be a short MLM screen against prefix repeat and compact before any mature run.

Figure: `experiments/archive/frontier_consolidation/figures/extractive_skeleton_variant_deciles.png`
Metrics CSV: `experiments/archive/frontier_consolidation/data/extractive_skeleton_variant_audit/extractive_skeleton_variant_metrics.csv`
Example skeletons: `experiments/archive/frontier_consolidation/data/extractive_skeleton_variant_audit/source_only_skeleton_examples.jsonl`
Variant pair JSONL files:
- compact: `experiments/archive/frontier_consolidation/data/extractive_skeleton_variant_audit/compact_pairs.jsonl` SHA `b374c63bdcba9542f661c85aaed7762876b725c7993cddafa478b37e4e5dbd51`
- prefix_repeat: `experiments/archive/frontier_consolidation/data/extractive_skeleton_variant_audit/prefix_repeat_pairs.jsonl` SHA `c874d9e3e882a0a5a2258df2e3bef1539f98184226d9aac131b1045d68ebfeae`
- spread_even: `experiments/archive/frontier_consolidation/data/extractive_skeleton_variant_audit/spread_even_pairs.jsonl` SHA `3bad14132e96821988f26e6f7ce14cbe04206baec22fd7e8170fcd24823dcd83`
- content_spread: `experiments/archive/frontier_consolidation/data/extractive_skeleton_variant_audit/content_spread_pairs.jsonl` SHA `19710c09a7a70fed3283def98d827be09ce3e3f7a149fb0e22ce22ee12a463a5`
- scored_source_skeleton: `experiments/archive/frontier_consolidation/data/extractive_skeleton_variant_audit/scored_source_skeleton_pairs.jsonl` SHA `a69b3912c56bf1079f409bddce0807a93e97b7f53b75f0fb199d041177b828e7`
- oracle_compact_projection: `experiments/archive/frontier_consolidation/data/extractive_skeleton_variant_audit/oracle_compact_projection_pairs.jsonl` SHA `a1e33f8cf837df15144f5f95929821edad733f2c533ef98eab67882dd586bc53`
JSON: `experiments/archive/frontier_consolidation/data/extractive_skeleton_variant_audit/extractive_skeleton_variant_audit.json`
