# debertav2 b256 coordinate — DeBERTa-v2 b256 near-complete coordinate

Evidence JSON: `experiments/archive/initial_model_studies/data/debertav2_b256_coordinate.json`

EWoK full is gated (missing); AoA is a degenerate official 0. No true official Overall yet.

| column | DeBERTa-v2 b256 | 2026 leader | gap |
|---|---:|---:|---:|
| BLiMP | 66.76 | 67.20 | -0.44 |
| BLiMP Supplement | 59.88 | 56.01 | +3.87 |
| EWoK | missing (fast-interim 46.18) | 56.07 | n/a |
| Entity Tracking | 22.62 | 28.45 | -5.83 |
| COMPS | 52.19 | 53.57 | -1.38 |
| (Super)GLUE | 68.02 | 69.79 | -1.77 |
| GlobalPIQA | 35.63 | 39.67 | -4.04 |
| Reading | 7.62 | 5.42 | +2.20 |
| AoA | 0.00 (degenerate) | 0.00 | +0.00 |

## (Super)GLUE subtasks

| task | accuracy | correct/n |
|---|---:|---:|
| boolq | 69.42 | 1135/1635 |
| multirc | 67.00 | 1624/2424 |
| rte | 61.87 | 86/139 |
| wsc | 67.31 | 35/52 |
| mrpc | 78.92 | 161/204 |
| qqp | 76.16 | 15395/20215 |
| mnli | 55.48 | 2723/4908 |

Mean (Super)GLUE = 68.02

Non-official sensitivity Overall (8 cols, EWoK missing) = 39.09
Non-official sensitivity Overall (9 cols using fast-interim EWoK 46.18) = 39.88
2026 leader Overall = 41.80

These sensitivity Overalls are NOT official leaderboard numbers.
