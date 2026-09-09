# aligned vs shuffled 10m available comparison — aligned vs shuffled 10M available-coordinate comparison

Comparison JSON: `experiments/archive/initial_model_studies/data/aligned_vs_shuffled_10m_available_comparison.json`

Aligned JSON: `experiments/archive/initial_model_studies/data/aligned_10m_available_coordinate.json`
Shuffled JSON: `experiments/archive/initial_model_studies/data/shuffled_10m_available_coordinate.json`

| column/task | aligned | shuffled | aligned - shuffled |
|---|---:|---:|---:|
| BLiMP | 53.65 | 51.83 | +1.82 |
| Supplement | 44.02 | 43.84 | +0.18 |
| Entity | 16.00 | 16.42 | -0.42 |
| COMPS | 50.03 | 50.18 | -0.15 |
| GlobalPIQA parallel | 10.68 | 13.59 | -2.91 |
| GlobalPIQA nonparallel | 51.00 | 52.00 | -1.00 |
| GlobalPIQA mean | 30.84 | 32.80 | -1.96 |
| Reading eye | 4.38 | 4.16 | +0.22 |
| Reading self-paced | 1.05 | 1.07 | -0.02 |
| Reading mean | 2.71 | 2.62 | +0.10 |

EWoK is evaluated by the companion full-EWoK script. The primary mechanism test is aligned-minus-shuffled on Entity and EWoK under identical text multisets.
