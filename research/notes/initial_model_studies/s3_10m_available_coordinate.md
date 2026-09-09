# s3 10m available coordinate — S3 official40k-under-12x384 10M available coordinate

Evidence JSON: `experiments/archive/initial_model_studies/data/s3_10m_available_coordinate.json`

Direct checkpoint: `experiments/archive/initial_model_studies/training/runs/babylm_s3_12x384_official40k_flatwwm_10M/hf_model/chck_10M`

| column/task | score |
|---|---:|
| BLiMP | 54.34 |
| Supplement | 52.50 |
| Entity | 17.77 |
| COMPS | 50.02 |
| GlobalPIQA parallel | 22.33 |
| GlobalPIQA nonparallel | 52.00 |
| GlobalPIQA mean | 37.16 |
| Reading eye | 2.18 |
| Reading self-paced | 0.16 |
| Reading mean | 1.17 |

This is a 10M legal official40k-under-12x384 screen on official data with flat WWM; exact leader data remains gated and this tokenizer is official-corpus ByteLevel-BPE, not the leader SentencePiece tokenizer.
