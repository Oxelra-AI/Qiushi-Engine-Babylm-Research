# bert8x512 b256 available coordinate — Update-geometry comparison

Evidence JSON: `experiments/archive/initial_model_studies/data/update_geometry_coordinate_comparison.json`

| column | 10.7M BERT | 34M BERT b512 | 34M BERT b256 | DeBERTa b256 | BERT update Δ | DeBERTa-vs-BERT at b256 |
|---|---:|---:|---:|---:|---:|---:|
| BLiMP | 55.92 | 53.68 | 54.25 | 66.76 | +0.57 | +12.51 |
| Supplement | 52.03 | 50.91 | 50.91 | 59.88 | +0.00 | +8.97 |
| Entity | 16.76 | 17.12 | 16.59 | 22.62 | -0.53 | +6.03 |
| COMPS | 51.66 | 49.84 | 50.11 | 52.19 | +0.27 | +2.08 |
| GlobalPIQA parallel | 17.48 | 20.39 | 20.39 | 24.27 | +0.00 | +3.88 |
| GlobalPIQA nonparallel | 48.00 | 50.00 | 52.00 | 47.00 | +2.00 | -5.00 |
| GlobalPIQA mean | 32.74 | 35.20 | 36.20 | 35.63 | +1.00 | -0.56 |
| Reading eye | 11.68 | 10.33 | 10.42 | 9.95 | +0.09 | -0.47 |
| Reading self-paced | 3.82 | 3.79 | 3.79 | 5.29 | +0.00 | +1.50 |
