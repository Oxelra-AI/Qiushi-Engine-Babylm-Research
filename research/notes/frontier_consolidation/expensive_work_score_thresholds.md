# expensive work score thresholds score calibration for expensive route continuation

CPU-only calibration using completed legal endpoint and cheap 70/80M reference scores; no training/evaluation.

## Current legal score gap
- Current best fully legal endpoint: Overall 41.257771; target 41.800; needed Overall gain 0.542229.
- Current cheap7 endpoint mean: 43.005724; SuperGLUE 70.279868; AoA 0.000000.
- Required total nine-column score gain: 4.880062.

## Cheap7 gain needed at 100M
| assumed SuperGLUE gain | assumed AoA gain | needed cheap7 mean gain | target cheap7 endpoint mean |
|---:|---:|---:|---:|
| -2.0 | 0.0 | 0.9829 | 43.9886 |
| -2.0 | 5.0 | 0.2686 | 43.2743 |
| -1.0 | 0.0 | 0.8400 | 43.8457 |
| -1.0 | 5.0 | 0.1257 | 43.1314 |
| 0.0 | 0.0 | 0.6972 | 43.7029 |
| 0.0 | 5.0 | -0.0171 | 42.9886 |
| 0.5 | 0.0 | 0.6257 | 43.6314 |
| 0.5 | 5.0 | -0.0886 | 42.9172 |
| 1.0 | 0.0 | 0.5543 | 43.5600 |
| 1.0 | 5.0 | -0.1600 | 42.8457 |
| 2.0 | 0.0 | 0.4114 | 43.4172 |
| 2.0 | 5.0 | -0.3028 | 42.7029 |
| 3.0 | 0.0 | 0.2686 | 43.2743 |
| 3.0 | 5.0 | -0.4457 | 42.5600 |

## Current token-mean maturity reference
| exposure | token-mean reinvest mean7 | clean mean7 | treatment Δ mean7 | endpoint100M - exposure mean7 |
|---:|---:|---:|---:|---:|
| 20 | 39.6636 | 40.4914 | -0.8279 | NA |
| 70 | 42.6086 | 41.3164 | 1.2921 | 0.3972 |
| 80 | 42.9486 | 41.6021 | 1.3464 | 0.0572 |

## Interpreting an 80M cheap-screen delta if SuperGLUE/AoA are flat
| cheap7 mean gain assumed | projected Overall | projected margin vs 41.8 | SuperGLUE gain needed if AoA flat |
|---:|---:|---:|---:|
| -1.00 | 40.4800 | -1.3200 | 11.8801 |
| -0.50 | 40.8689 | -0.9311 | 8.3801 |
| 0.00 | 41.2578 | -0.5422 | 4.8801 |
| 0.25 | 41.4522 | -0.3478 | 3.1301 |
| 0.50 | 41.6467 | -0.1533 | 1.3801 |
| 0.70 | 41.8022 | 0.0022 | -0.0199 |
| 1.00 | 42.0355 | 0.2355 | -2.1199 |
| 1.50 | 42.4244 | 0.6244 | -5.6199 |
| 2.00 | 42.8133 | 1.0133 | -9.1199 |

## Scientific reading
- If SuperGLUE and AoA remain at the current legal endpoint values, the seven cheap columns need about +0.697 mean points at the 100M endpoint to reach 41.8.
- A cheap-screen 80M gain much below +0.5 mean7 would require a large unmeasured SuperGLUE/AoA gain and should not by itself justify a 100M/full continuation.
- A continuation is stronger when the gain is broad and includes BLiMP/Supplement/EWoK without sacrificing Entity or GlobalPIQA, because prior larger-vocabulary evidence improved some language columns while harming those columns.
- AoA has repeatedly scored 0.0 or a negative thresholded value in this line; do not rely on AoA to close the gap unless a full trajectory proves it.

JSON: `experiments/archive/frontier_consolidation/data/expensive_work_score_thresholds/expensive_work_score_thresholds.json`
