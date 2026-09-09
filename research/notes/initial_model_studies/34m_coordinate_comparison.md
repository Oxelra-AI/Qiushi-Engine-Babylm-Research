# 34m coordinate comparison — 34M coordinate comparison

Evidence JSON: `experiments/archive/initial_model_studies/data/34m_coordinate_comparison.json`

DeBERTa-v2 is a batch-256 rescue run after batch-512 OOM; differences against BERT34 mix architecture/package with update geometry.

| column | 10.7M BERT | 34M BERT b512 | DeBERTa-v2 b256 | cap Δ | DeBERTa-b256 Δ vs BERT34 |
|---|---:|---:|---:|---:|---:|
| BLiMP | 55.92 | 53.68 | 66.76 | -2.24 | +13.08 |
| Supplement | 52.03 | 50.91 | 59.88 | -1.12 | +8.97 |
| Entity | 16.76 | 17.12 | 22.62 | +0.36 | +5.50 |
| COMPS | 51.66 | 49.84 | 52.19 | -1.82 | +2.35 |
| GlobalPIQA parallel | 17.48 | 20.39 | 24.27 | +2.91 | +3.88 |
| GlobalPIQA nonparallel | 48.00 | 50.00 | 47.00 | +2.00 | -3.00 |
| GlobalPIQA mean | 32.74 | 35.20 | 35.63 | +2.45 | +0.44 |
| Reading eye | 11.68 | 10.33 | 9.95 | -1.35 | -0.38 |
| Reading self-paced | 3.82 | 3.79 | 5.29 | -0.03 | +1.50 |
