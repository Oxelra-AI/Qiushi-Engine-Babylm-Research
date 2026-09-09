# earlier analysis scale1.75 split mature evaluation

## Scores

| exposure | arm | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 70M | spatial repair route status | 42.6086 | 65.390 | 59.310 | 50.470 | 26.980 | 51.820 | 35.550 | 8.740 |
| 70M | scale1.75 | 42.6700 | 67.550 | 60.430 | 48.550 | 27.370 | 52.070 | 34.680 | 8.040 |
| 80M | spatial repair route status | 42.9486 | 66.110 | 60.660 | 51.010 | 27.060 | 51.930 | 35.580 | 8.290 |
| 80M | scale1.75 | 43.8121 | 68.110 | 62.620 | 49.240 | 28.200 | 52.110 | 38.105 | 8.300 |

## Deltas

| exposure | cheap7 Δ | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 70M | +0.0614 | +2.160 | +1.120 | -1.920 | +0.390 | +0.250 | -0.870 | -0.700 |
| 80M | +0.8636 | +2.000 | +1.960 | -1.770 | +1.140 | +0.180 | +2.525 | +0.010 |

## Interpretation

- 70M: cheap7 Δ +0.0614, columns {'BLiMP': 2.1599999999999966, 'Supplement': 1.1199999999999974, 'EWoK': -1.9200000000000017, 'Entity': 0.39000000000000057, 'COMPS': 0.25, 'GlobalPIQA': -0.8699999999999974, 'Reading': -0.6999999999999993}
- 80M: cheap7 Δ +0.8636, columns {'BLiMP': 2.0, 'Supplement': 1.9600000000000009, 'EWoK': -1.769999999999996, 'Entity': 1.1400000000000006, 'COMPS': 0.17999999999999972, 'GlobalPIQA': 2.5250000000000057, 'Reading': 0.010000000000001563}
- Next required readout: pairwise item-flip family analysis at 70M/80M. Only launch 100M if losing families recover while gains persist; otherwise fixed scale1.75 is localized redistribution.
