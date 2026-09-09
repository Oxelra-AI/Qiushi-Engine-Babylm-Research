# adapter scale sweep plan train-time scaled adapter 20M merged

| row | cheap7 | Δ vs spatial repair route status | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial repair route status legal 20M | 39.6636 | +0.0000 | 59.690 | 55.450 | 50.730 | 18.650 | 50.260 | 34.195 | 8.670 |
| adapter matched horizon plan trained scale1 evaluated scale1 | 39.3893 | -0.2743 | 60.380 | 56.230 | 49.490 | 18.650 | 50.440 | 32.225 | 8.310 |
| adapter matched horizon plan scale1 checkpoint evaluated at scale1.75 | 39.7550 | +0.0914 | 60.560 | 57.120 | 50.740 | 18.470 | 50.460 | 32.710 | 8.225 |
| adapter matched horizon plan scale1 checkpoint evaluated at scale2.00 | 39.8193 | +0.1557 | 60.610 | 57.350 | 51.080 | 18.460 | 50.330 | 32.710 | 8.195 |
| train-time adapter scale 1.75 | 40.3057 | +0.6421 | 59.610 | 56.270 | 52.220 | 18.220 | 49.970 | 37.150 | 8.700 |
| train-time adapter scale 2.00 | 39.7229 | +0.0593 | 59.920 | 56.210 | 50.540 | 17.450 | 50.460 | 35.120 | 8.360 |

## Interpretation
- A train-time scaled adapter 20M arm exceeds spatial repair route status 20M cheap7, so residual amplitude is a trajectory-level signal rather than only post-hoc endpoint sensitivity.
