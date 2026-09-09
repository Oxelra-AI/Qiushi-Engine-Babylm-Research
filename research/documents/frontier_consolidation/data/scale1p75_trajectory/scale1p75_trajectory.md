# scale1p75 pre80m state scale1.75 trajectory

## Scores

| exposure | arm | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20M | spatial repair route status | 39.6636 | 59.690 | 55.450 | 50.730 | 18.650 | 50.260 | 34.195 | 8.670 |
| 20M | scale1p75 | 40.3057 | 59.610 | 56.270 | 52.220 | 18.220 | 49.970 | 37.150 | 8.700 |
| 30M | spatial repair route status | 41.1943 | 63.080 | 57.860 | 48.430 | 22.610 | 51.170 | 37.120 | 8.090 |
| 30M | scale1p75 | 41.1571 | 62.870 | 58.520 | 49.410 | 22.320 | 50.690 | 36.210 | 8.080 |
| 40M | spatial repair route status | 42.2214 | 63.190 | 57.600 | 48.230 | 26.810 | 52.440 | 38.605 | 8.675 |
| 40M | scale1p75 | 41.7464 | 63.970 | 57.180 | 48.410 | 28.430 | 51.800 | 34.650 | 7.785 |
| 50M | spatial repair route status | 41.9729 | 63.800 | 59.110 | 48.460 | 26.260 | 51.610 | 36.580 | 7.990 |
| 50M | scale1p75 | 42.3843 | 65.640 | 59.070 | 47.730 | 29.050 | 52.340 | 35.105 | 7.755 |

## Deltas scale1.75 minus spatial repair route status

| exposure | cheap7 Δ | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20M | +0.6421 | -0.080 | +0.820 | +1.490 | -0.430 | -0.290 | +2.955 | +0.030 |
| 30M | -0.0371 | -0.210 | +0.660 | +0.980 | -0.290 | -0.480 | -0.910 | -0.010 |
| 40M | -0.4750 | +0.780 | -0.420 | +0.180 | +1.620 | -0.640 | -3.955 | -0.890 |
| 50M | +0.4114 | +1.840 | -0.040 | -0.730 | +2.790 | +0.730 | -1.475 | -0.235 |

## Interpretation

- cheap7 delta trajectory 20/30/40/50M = [0.6421, -0.0371, -0.475, 0.4114]
- The 50M advantage is not an isolated one-checkpoint artifact; it remains positive after the exact reproduced 20M prefix.
- Entity and BLiMP are positive at 50M, unlike the scale1.75 20M Entity weakness.
- Remaining 50M negative columns below -0.5 are {'EWoK': -0.730000000000004, 'GlobalPIQA': -1.4749999999999943}; mature continuation should test whether these recover or deepen.
