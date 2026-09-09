# earlier analysis — fast Supplement late checkpoint grid

Existing checkpoint ladders only; fast Supplement slices only; no pretraining, corpus modification, SuperGLUE, AoA, or full official evaluation.

## Reinvest candidate rows
| exposure M | reinvest 43022 | reinvest 43122 | mean | min seed | gap 43122-43022 | TE43022 | TE43122 | DiD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 60.800 | 63.200 | 62.000 | 60.800 | +2.400 | -4.000 | 3.200 | 7.200 |
| 45 | 63.200 | 65.600 | 64.400 | 63.200 | +2.400 | 1.200 | 4.000 | 2.800 |
| 50 | 62.400 | 64.400 | 63.400 | 62.400 | +2.000 | -3.200 | 2.400 | 5.600 |
| 55 | 65.600 | 64.400 | 65.000 | 64.400 | -1.200 | -0.800 | 1.200 | 2.000 |
| 60 | 62.800 | 64.800 | 63.800 | 62.800 | +2.000 | -0.400 | 0.400 | 0.800 |
| 65 | 64.800 | 63.600 | 64.200 | 63.600 | -1.200 | 0.400 | -0.800 | -1.200 |
| 70 | 64.000 | 63.200 | 63.600 | 63.200 | -0.800 | -1.600 | -3.600 | -2.000 |
| 75 | 62.800 | 63.600 | 63.200 | 62.800 | +0.800 | -1.600 | -2.800 | -1.200 |
| 80 | 66.000 | 65.200 | 65.600 | 65.200 | -0.800 | 3.200 | -0.800 | -4.000 |
| 85 | 65.600 | 62.400 | 64.000 | 62.400 | -3.200 | 4.000 | -1.200 | -5.200 |
| 90 | 67.600 | 63.600 | 65.600 | 63.600 | -4.000 | 2.800 | -0.400 | -3.200 |
| 95 | 66.400 | 63.600 | 65.000 | 63.600 | -2.800 | 2.400 | -0.800 | -3.200 |
| 100 | 66.400 | 63.200 | 64.800 | 63.200 | -3.200 | 2.000 | -2.000 | -4.000 |

## Best fast-Supplement exposures
- max_reinvest_43022: {'exposure_m': 90, 'reinvest_43022': 67.6, 'reinvest_43122': 63.6, 'reinvest_mean': 65.6, 'reinvest_min_seed': 63.6, 'seed_gap_43122_minus_43022': -3.999999999999993, 'TE_43022': 2.799999999999997, 'TE_43122': -0.3999999999999986, 'DiD': -3.1999999999999957}
- max_reinvest_43122: {'exposure_m': 45, 'reinvest_43022': 63.2, 'reinvest_43122': 65.6, 'reinvest_mean': 64.4, 'reinvest_min_seed': 63.2, 'seed_gap_43122_minus_43022': 2.3999999999999915, 'TE_43022': 1.2000000000000028, 'TE_43122': 3.999999999999993, 'DiD': 2.79999999999999}
- max_reinvest_mean: {'exposure_m': 80, 'reinvest_43022': 66.0, 'reinvest_43122': 65.2, 'reinvest_mean': 65.6, 'reinvest_min_seed': 65.2, 'seed_gap_43122_minus_43022': -0.7999999999999972, 'TE_43022': 3.200000000000003, 'TE_43122': -0.7999999999999972, 'DiD': -4.0}
- max_reinvest_min_seed: {'exposure_m': 80, 'reinvest_43022': 66.0, 'reinvest_43122': 65.2, 'reinvest_mean': 65.6, 'reinvest_min_seed': 65.2, 'seed_gap_43122_minus_43022': -0.7999999999999972, 'TE_43022': 3.200000000000003, 'TE_43122': -0.7999999999999972, 'DiD': -4.0}
- min_abs_seed_gap: {'exposure_m': 70, 'reinvest_43022': 64.0, 'reinvest_43122': 63.2, 'reinvest_mean': 63.6, 'reinvest_min_seed': 63.2, 'seed_gap_43122_minus_43022': -0.7999999999999972, 'TE_43022': -1.5999999999999943, 'TE_43122': -3.5999999999999943, 'DiD': -2.0}

## Scientific read
- Seed43122 fast Supplement best is 65.600 at 45M versus 63.200 at 100M; this says whether earlier stopping can actually raise the weak seed's Supplement before broad task evaluation.
- Best across-seed minimum is 65.200 at 80M; if this occurs far before 100M, broad balance must be checked because Entity/EWoK/Reading may move differently.
- This grid only selects or rejects candidate stopping exposures for further narrow broad-surface checks; it cannot by itself establish Overall or official full-Supplement improvement.

Machine-readable output: `experiments/archive/frontier_consolidation/data/supplement_late_grid/supplement_late_grid.json`
