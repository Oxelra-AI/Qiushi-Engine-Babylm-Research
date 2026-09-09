# earlier analysis context-isolation masking construction check

Records: `experiments/archive/relation_learning/data/context_isolation_screen_all100M/sentence_records.jsonl`; n=651; max_len=256; mask_prob=0.15.

| tokenizer | same target seq | same masked labels | row whole-span masked | target trunc | mean target tokens | mean masked |
|---|---:|---:|---:|---:|---:|---:|
| compact_experience_off43022 | 0.9462 | 0.9601 | 0.0015 | 35 | 22.17 | 3.33 |
| representation_frontier_studies_crv_c43022 | 0.9570 | 0.9631 | 0.0000 | 28 | 22.21 | 3.34 |
| adapter_base43022 | 0.9570 | 0.9631 | 0.0000 | 28 | 22.21 | 3.34 |

Summary JSON: `experiments/archive/relation_learning/data/context_isolation_mask_check/summary.json`
