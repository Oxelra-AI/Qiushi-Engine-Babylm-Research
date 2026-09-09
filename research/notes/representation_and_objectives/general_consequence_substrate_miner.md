# fw globalpiqa relevant substrate — General consequence/contrast substrate miner

## Purpose

The GlobalPIQA margin reader shows the lineage usually puts the correct four-choice physical/temporal/spatial/affordance answer at rank 3 or 4 on the hardest rows. This CPU-only miner asks whether existing allowed reservoirs contain enough broad consequence/contrast material for a later strict-compliant low-cost probe. It does not read official evaluation items and does not construct a final training corpus.

## Reservoir summary

| reservoir | scanned sentences | scanned words | physical-causal words | spatial-causal words | temporal-quant words | affordance-consequence words | explicit-contrast words |
|---|---:|---:|---:|---:|---:|---:|---:|
| `compact_experience_aligned_10m_rows` | 560054 | 8077251 | 240832 | 438020 | 449468 | 181415 | 81220 |
| `fw_frozen_sources_38167` | 38317 | 878412 | 32423 | 42853 | 83072 | 32668 | 11159 |
| `fw_compact_pair_sources` | 22885 | 495351 | 15829 | 19531 | 36961 | 17960 | 5632 |
| `fw_compact_rewrites` | 22640 | 319932 | 6949 | 6591 | 18386 | 7622 | 3453 |
| `fw_breadth_whole_sentence_companions` | 12208 | 319392 | 12243 | 18203 | 32721 | 12392 | 4798 |

Deduplicated high-scoring candidate sentences: 16818 totaling 522132 words. This is a candidate reservoir for a later probe, not a selected training arm.

## Scientific use

- If the ongoing compact-vs-breadth official vectors show GlobalPIQA_parallel or EWoK movement, reuse the margin reader and this reservoir summary to interpret whether the movement came from dense same-proposition restatement or added consequence substrate.
- If the running FW arms do not move the hard rows, the next efficient route is a small factorial probe that swaps a controlled 50k–200k word slice from this general reservoir and/or adds a corpus-derived paired contrastive objective. It should be tested from shared checkpoints before any 100M commitment.
- Because many GlobalPIQA_parallel hard rows are spatial/direction and time/counting, a future route should keep physical, spatial, temporal/quantitative, and affordance channels separate rather than calling them one undifferentiated common-sense bucket.

## Files

- JSON: `experiments/archive/representation_and_objectives/data/general_consequence_substrate/general_consequence_substrate_miner.json`
- summary CSV: `experiments/archive/representation_and_objectives/data/general_consequence_substrate/general_consequence_substrate_summary.csv`
- candidate JSONL: `experiments/archive/representation_and_objectives/data/general_consequence_substrate/general_consequence_candidates.jsonl`
