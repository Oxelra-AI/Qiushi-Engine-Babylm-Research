# adapter scale sweep plan adapter scale-sweep summary

Inference-time adapter scaling is treated as an endpoint sensitivity map, not as a clean branch-vs-backbone causal separation.

spatial repair route status/disabled 20M cheap7: 39.6636; adapter matched horizon plan live scale=1 cheap7: 39.3893.

| scale | cheap7 | Δ vs spatial repair route status | Δ vs live1 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 39.4486 | -0.2150 | +0.0593 | 60.170 | 55.900 | 49.390 | 18.770 | 50.360 | 33.195 | 8.355 |
| 0.75 | 39.4550 | -0.2086 | +0.0657 | 60.310 | 56.270 | 49.320 | 18.830 | 50.410 | 32.710 | 8.335 |
| 1.00 | 39.3893 | -0.2743 | +0.0000 | 60.380 | 56.230 | 49.490 | 18.650 | 50.440 | 32.225 | 8.310 |
| 1.25 | 39.5779 | -0.0857 | +0.1886 | 60.480 | 56.890 | 49.660 | 18.530 | 50.480 | 32.725 | 8.280 |
| 1.50 | 39.6429 | -0.0207 | +0.2536 | 60.550 | 56.890 | 50.170 | 18.430 | 50.500 | 32.710 | 8.250 |
| 1.75 | 39.7550 | +0.0914 | +0.3657 | 60.560 | 57.120 | 50.740 | 18.470 | 50.460 | 32.710 | 8.225 |
| 2.00 | 39.8193 | +0.1557 | +0.4300 | 60.610 | 57.350 | 51.080 | 18.460 | 50.330 | 32.710 | 8.195 |
| 2.50 | 39.6586 | -0.0050 | +0.2693 | 60.730 | 56.450 | 51.060 | 18.290 | 50.230 | 32.710 | 8.140 |

## Interpretation
- At least one inference scale exceeds spatial repair route status 20M cheap7; adapter amplitude control deserves construction/training follow-up.
