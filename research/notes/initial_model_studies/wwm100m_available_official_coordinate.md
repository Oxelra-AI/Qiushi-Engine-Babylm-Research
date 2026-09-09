# wwm100m available official coordinate — WWM 100M available official coordinate

Evidence JSON: `experiments/archive/initial_model_studies/data/wwm100m_available_official_coordinate.json`

Final checkpoint: `chck_100M`, backend `mlm`.

| column/task | score | provenance |
|---|---:|---|
| BLiMP | 55.92 | recovered_from_step72_failed_runner |
| Supplement | 52.03 | recovered_from_step72_failed_runner |
| Entity Tracking | 16.76 | fresh_run |
| COMPS | 51.66 | fresh_run |
| GlobalPIQA parallel | 17.48 | fresh_run |
| GlobalPIQA nonparallel | 48.00 | fresh_run |
| GlobalPIQA mean | 32.74 | derived |
| Reading eye | 11.68 | fresh_run |
| Reading self-paced | 3.82 | fresh_run |

EWoK is intentionally missing because `evaluation_data/full_eval/ewok_filtered` exists but contains no JSONL files after official download; AoA and (Super)GLUE are not included in this runner.
