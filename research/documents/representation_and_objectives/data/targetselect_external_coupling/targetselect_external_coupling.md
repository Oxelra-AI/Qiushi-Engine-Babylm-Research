# wholeword copied control plan target-selective early external coupling

This summarizes the repaired target selective source absent result 20M official-style readout. It is not a route decision; the original compact-view advantage matured late.

## cheap7-style scores

| arm | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | equal7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 60.190 | 57.900 | 48.020 | 17.940 | 50.340 | 33.225 | 8.040 | 39.379 |
| drop_abs | 60.700 | 58.500 | 50.810 | 18.000 | 50.510 | 34.240 | 8.015 | 40.111 |
| drop_copied_tok | 60.560 | 59.250 | 49.400 | 18.630 | 50.560 | 32.765 | 8.560 | 39.961 |

## Deltas

| contrast | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | equal7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| drop_abs_minus_full | +0.510 | +0.600 | +2.790 | +0.060 | +0.170 | +1.015 | -0.025 | +0.731 |
| drop_copied_tok_minus_full | +0.370 | +1.350 | +1.380 | +0.690 | +0.220 | -0.460 | +0.520 | +0.581 |
| drop_abs_minus_drop_copied_tok | +0.140 | -0.750 | +1.410 | -0.630 | -0.050 | +1.475 | -0.545 | +0.150 |

## EWoK domain deltas: drop_abs minus token copied-drop

| domain | n | delta pp | full | drop_abs | drop_copied_tok |
|---|---:|---:|---:|---:|---:|
| physical-dynamics | 120 | +13.333 | 32.50 | 51.67 | 38.33 |
| material-dynamics | 770 | +8.701 | 48.96 | 54.16 | 45.45 |
| social-relations | 1548 | +0.969 | 50.00 | 50.97 | 50.00 |
| physical-relations | 818 | +0.733 | 48.90 | 49.76 | 49.02 |
| spatial-relations | 490 | +0.612 | 42.86 | 45.71 | 45.10 |
| social-properties | 328 | +0.305 | 47.87 | 48.17 | 47.87 |
| agent-properties | 2210 | -0.181 | 50.59 | 50.27 | 50.45 |
| social-interactions | 294 | -0.680 | 53.40 | 53.40 | 54.08 |
| material-properties | 170 | -1.176 | 49.41 | 50.00 | 51.18 |
| physical-interactions | 556 | -2.338 | 49.64 | 51.98 | 54.32 |
| quantitative-properties | 314 | -4.777 | 54.14 | 52.87 | 57.64 |

JSON: `experiments/archive/representation_and_objectives/data/targetselect_external_coupling/targetselect_external_coupling.json`
