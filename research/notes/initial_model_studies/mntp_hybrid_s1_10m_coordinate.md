# mntp hybrid s1 10m coordinate — deterministic MNTP hybrid S1 10M coordinate

Evidence JSON: `experiments/archive/initial_model_studies/data/mntp_hybrid_s1_10m_coordinate.json`
Run: `experiments/archive/initial_model_studies/training/runs/babylm_mntp_hybrid_s1_10M`

Matched against leadershape micro split alignment S1 10M: same data/order/exposure/updates/schedule/init; only every 16th batch uses MNTP.

| column/task | MNTP 10M | S1 10M | delta | public leader | MNTP-leader |
|---|---:|---:|---:|---:|---:|
| BLiMP | 50.78 | 53.37 | -2.59 | 67.20 | -16.42 |
| Supplement | 54.19 | 52.34 | +1.85 | 56.04 | -1.85 |
| EWoK full | 50.48 | 50.60 | -0.12 | 56.07 | -5.59 |
| Entity | 17.39 | 17.73 | -0.34 | 28.45 | -11.06 |
| COMPS | 49.73 | 50.34 | -0.61 | 53.57 | -3.84 |
| GlobalPIQA mean | 34.75 | 35.21 | -0.46 | 39.66 | -4.91 |
| Reading mean | 7.40 | 8.35 | -0.95 | 5.42 | +1.97 |

Objective counts: `{'wwm': 230, 'mntp': 15}`, predicted tokens: `{'wwm': 2032237, 'mntp': 133272}`.
Loss: 9.7532 → 4.2581 (S1 10M loss_last 4.0523).
