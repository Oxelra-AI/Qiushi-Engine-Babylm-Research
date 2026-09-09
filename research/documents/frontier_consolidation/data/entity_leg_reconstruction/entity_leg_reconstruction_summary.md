# entity leg reconstruction and second basin plan Entity-leg reconstruction before fresh tables

This readout uses only existing stable-family score files; it runs no model inference.

## Entity V-R dose curve on common 10M-80M window

| dose | rho | Entity V-R | contribution to cheap6 | contribution to cheap5 | effect/rho | ratio vs 1x |
|---:|---:|---:|---:|---:|---:|---:|
| 1x | 0.042352 | +0.5587 | +0.0931 | +0.1117 | 13.19 | 1.00 |
| 1.82x | 0.077120 | +1.0662 | +0.1777 | +0.2132 | 13.83 | 1.91 |
| 2.64x/MAX | 0.111872 | +2.0625 | +0.3437 | +0.4125 | 18.44 | 3.69 |

The finite slopes are 14.60 score-points/rho from 1x to 1.82x and 28.67 from 1.82x to MAX. A through-origin fit gives 16.62 score-points/rho; ordinary least squares gives slope 21.63, intercept -0.439, R^2 0.966 on the three points.

## Noise-scale reconciliation

At 1x, the Entity leg contributes +0.0931 to cheap6. The 1x two-seed trajectory-mean cheap6 treatment difference is +0.1133, with pointwise mean absolute difference 0.4199. This explains why minimum-dose six-family comparisons could miss the carrier even when Entity itself is moving.
For cheap5 the 1x Entity contribution is +0.1117, while the trajectory-mean cheap5 seed difference is +0.3014; at MAX the Entity contribution grows to +0.4125.
The 1x Entity family itself has a two-seed signed trajectory difference +0.3030, pointwise mean absolute difference 1.1370, and pointwise max absolute difference 2.7200; this is why the second-basin MAX measurement must be read directly, not inferred from one basin.

## EWoK absolute level and relative movement

| dose | window | n | mean EWoK V-R | view mean | repeat mean | fraction of paired checkpoints with both scores in [49,51] |
|---:|---|---:|---:|---:|---:|---:|
| 1x | common_10M_80M | 8 | -0.8825 | 49.383 | 50.265 | 0.25 |
| 1x | available_10M_100M | 10 | -0.8730 | 49.567 | 50.440 | 0.20 |
| 1.82x | common_10M_80M | 8 | -1.1650 | 48.788 | 49.953 | 0.12 |
| 1.82x | available_10M_100M | 10 | -1.1790 | 48.829 | 50.008 | 0.20 |
| 2.64x/MAX | common_10M_80M | 8 | -0.6062 | 49.653 | 50.259 | 0.75 |
| 2.64x/MAX | available_10M_100M | 10 | -0.4110 | 49.782 | 50.193 | 0.80 |

EWoK is negative over the common 10M-80M relative leg, but MAX recovers at 90M/100M (+0.47 and +0.27). Across view/repeat ladders its absolute scores hover near chance, so the current first-basin science is safer as a dose-scaled Entity/state-tracking gain than as a settled Entity-versus-EWoK trade.

## Frozen quantitative prediction

The fresh second-basin MAX view-minus-repeat Entity score should be near +2 score points on the same stable-family readout, much closer to the first-basin MAX value (+2.0625) than to the first-basin 1x value (+0.5588). MAX breadth should fall well short of that Entity lift if source-conditioned re-expression is the carrier; if generic non-duplicate same-population companion text is sufficient, breadth should approach the compact-view Entity level. EWoK should be read as near-chance context unless the coming full-ladder or margin output shows stable movement away from 50.

## Files

- family_vr_common10_80_by_dose_csv: `experiments/archive/frontier_consolidation/data/entity_leg_reconstruction/family_vr_common10_80_by_dose.csv`
- entity_vr_dose_curve_csv: `experiments/archive/frontier_consolidation/data/entity_leg_reconstruction/entity_vr_dose_curve_common10_80.csv`
- ewok_absolute_ladder_csv: `experiments/archive/frontier_consolidation/data/entity_leg_reconstruction/ewok_absolute_ladder_view_repeat.csv`
- ewok_absolute_summary_csv: `experiments/archive/frontier_consolidation/data/entity_leg_reconstruction/ewok_absolute_summary_by_dose_arm.csv`
- ewok_vr_by_checkpoint_csv: `experiments/archive/frontier_consolidation/data/entity_leg_reconstruction/ewok_vr_by_checkpoint.csv`
- ewok_vr_summary_csv: `experiments/archive/frontier_consolidation/data/entity_leg_reconstruction/ewok_vr_summary_by_dose.csv`
- figure: `{'path': 'experiments/archive/frontier_consolidation/data/entity_leg_reconstruction/entity_ewok_dose_reconstruction.png', 'created': True}`
