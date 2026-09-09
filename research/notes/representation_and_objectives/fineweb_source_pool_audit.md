# babysteps public method reading FineWeb source-pool audit

This CPU-only audit quantifies available FineWeb source sentences across representation_and_objectives and frontier_consolidation before any Qwen generation or H100 training.

## Per-pool scale

| pool | rows | unique norms | words | unique docs | entities/row mean | numbers/row mean | relation/row mean | bad-hint rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A01_step011_544_stratified | 544 | 544 | 14610 | 411 | 1.463 | 0.450 | 1.062 | 4 |
| A01_step010_balanced_source_by_rewrite | 32836 | 32819 | 703975 | 5283 | 1.059 | 0.330 | 0.816 | 191 |
| A01_step010_strict_factual_expository | 14385 | 14380 | 332669 | 4566 | 1.362 | 0.417 | 1.297 | 96 |
| A02_step011_high_precision | 9517 | 9512 | 214771 | 3913 | 1.296 | 0.377 | 1.258 | 14 |
| A02_step011_medium_repaired | 21465 | 21456 | 469887 | 5018 | 0.925 | 0.283 | 1.235 | 34 |
| A02_step011_high_anchor | 3985 | 3983 | 97605 | 2268 | 2.187 | 0.551 | 1.372 | 6 |

## Deduplicated priority pool

All loaded rows: 82732; unique normalized sentences: 33512; cross-pool duplicated normalized sentences: 21790.
Priority unique pool: 33512 rows, 725008 source words, 5293 docs.

Cumulative clean budgets (priority order high-anchor -> high-precision -> strict factual -> medium -> balanced):

| rows | source words | paired words @0.9 rewrite/source | unique docs |
|---:|---:|---:|---:|
| 512 | 13871 | 26355 | 462 |
| 1024 | 26829 | 50975 | 839 |
| 2048 | 51027 | 96951 | 1425 |
| 4096 | 100521 | 190990 | 2331 |
| 8192 | 188569 | 358281 | 3658 |
| 12000 | 281600 | 535040 | 4276 |
| 16000 | 371913 | 706635 | 4705 |
| 20000 | 447998 | 851196 | 4966 |
| 33512 | 725008 | 1377515 | 5293 |

## Interpretation

The audited high-anchor/high-precision material is sufficient for a small source-by-rewrite generation and training probe but does not by itself justify a 1.5-2M-word high-quality rewrite block. Inflating to that size would require lower-quality balanced/medium material or heavy repetition, changing the mechanism. The clean next experiment should therefore begin with generation/faithfulness measurement on the high-anchor/high-precision pool, then materialize a three-arm contrast at the actually accepted unique-word scale: protected slot control, FineWeb source repetition, and FineWeb source+accepted rewrite.

JSON: `experiments/archive/representation_and_objectives/data/fineweb_source_pool_audit/fineweb_source_pool_audit.json`
Priority sources: `experiments/archive/representation_and_objectives/data/fineweb_source_pool_audit/fineweb_priority_unique_sources.jsonl`
Samples: `experiments/archive/representation_and_objectives/data/fineweb_source_pool_audit/fineweb_priority_samples.json`
