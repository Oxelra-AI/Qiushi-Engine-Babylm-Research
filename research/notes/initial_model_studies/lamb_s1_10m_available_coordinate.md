# lamb s1 10m available coordinate — LAMB reference-default S1 10M available coordinate

Evidence JSON: `experiments/archive/initial_model_studies/data/lamb_s1_10m_available_coordinate.json`

Direct checkpoint: `experiments/archive/initial_model_studies/training/runs/babylm_lamb_refdefault_s1_10M/hf_model/chck_10M`

| column/task | score |
|---|---:|
| BLiMP | 51.75 |
| Supplement | 51.63 |
| Entity | 16.56 |
| COMPS | 50.33 |
| GlobalPIQA parallel | 18.45 |
| GlobalPIQA nonparallel | 54.00 |
| GlobalPIQA mean | 36.23 |
| Reading eye | 9.32 |
| Reading self-paced | 4.22 |
| Reading mean | 6.77 |

This is a 10M LAMB early-dynamics screen on official data with baseline16k tokenizer, S1 12x384 shape, and flat WWM; exact leader data remains gated and this does not substitute for the possible full 100M LAMB trajectory.
