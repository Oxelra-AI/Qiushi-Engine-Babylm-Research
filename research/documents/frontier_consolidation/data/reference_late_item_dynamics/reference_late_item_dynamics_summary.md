# chck84 late item dynamics synthesis reference late item dynamics around `chck_84M`

CPU/file-only analysis of already completed selected cheap-task prediction payloads for the scale1.75 seed43022 reference trajectory. It avoids pending managed-task outputs and does not run model inference.

Usable endpoints: chck_70M, chck_72M, chck_74M, chck_76M, chck_78M, chck_80M, chck_82M, chck_84M, chck_86M, chck_88M, chck_90M, chck_92M, chck_100M. Skipped: none. Classification items: 170,722.

## Score curve

| endpoint | cheap7 | cheap6 no GP | cheap5 no GP/Reading | relation/state | syntax/surface | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chck_70M | 42.670835 | 44.002705 | 51.195247 | 37.956928 | 59.815309 | 67.5573 | 60.4318 | 48.5477 | 27.3662 | 52.0733 | 34.6796 | 8.0400 |
| chck_72M | 42.938753 | 43.989224 | 51.180069 | 37.708081 | 59.763361 | 67.3240 | 60.9575 | 48.5960 | 26.8202 | 52.2027 | 36.6359 | 8.0350 |
| chck_74M | 43.449941 | 44.590465 | 51.919558 | 38.758843 | 60.017125 | 67.8472 | 62.0459 | 50.1355 | 27.3821 | 52.1870 | 36.6068 | 7.9450 |
| chck_76M | 43.582083 | 44.658871 | 51.958645 | 37.746730 | 60.119056 | 68.2928 | 64.0617 | 49.1218 | 26.3717 | 51.9453 | 37.1214 | 8.1600 |
| chck_78M | 43.702947 | 44.718972 | 51.986766 | 38.390826 | 60.186245 | 68.0279 | 62.7797 | 49.0351 | 27.7466 | 52.3446 | 37.6068 | 8.3800 |
| chck_80M | 43.814599 | 44.765899 | 52.059079 | 38.720744 | 60.114833 | 68.1170 | 62.6242 | 49.2431 | 28.1984 | 52.1127 | 38.1068 | 8.3000 |
| chck_82M | 43.959634 | 45.023294 | 52.397953 | 39.184748 | 60.341230 | 68.4913 | 62.9378 | 50.0555 | 28.3140 | 52.1912 | 37.5777 | 8.1500 |
| chck_84M | 44.123626 | 45.124004 | 52.517804 | 39.324289 | 60.228378 | 68.2512 | 63.4837 | 50.0735 | 28.5751 | 52.2056 | 38.1214 | 8.1550 |
| chck_86M | 43.771493 | 45.044088 | 52.432906 | 39.361439 | 60.382987 | 68.4735 | 62.6757 | 50.1403 | 28.5825 | 52.2925 | 36.1359 | 8.1000 |
| chck_88M | 43.709408 | 45.054988 | 52.417986 | 39.267565 | 60.363460 | 68.4891 | 62.8279 | 49.3959 | 29.1393 | 52.2378 | 35.6359 | 8.2400 |
| chck_90M | 43.270640 | 44.873999 | 52.193799 | 38.627600 | 60.412809 | 68.5885 | 62.8882 | 49.0860 | 28.1692 | 52.2372 | 33.6505 | 8.2750 |
| chck_92M | 43.368627 | 44.824078 | 52.123894 | 38.394493 | 60.488765 | 68.6528 | 62.8530 | 48.9900 | 27.7990 | 52.3247 | 34.6359 | 8.3250 |
| chck_100M | 43.543183 | 44.782581 | 52.075097 | 38.272321 | 60.467835 | 68.6318 | 62.8952 | 49.0801 | 27.4645 | 52.3039 | 36.1068 | 8.3200 |

## Local peak shape

`chck_84M` relative to the mean of its immediate scored neighbors (`chck_82M`, `chck_86M`):

- cheap7: +0.258062
- cheap6_no_globalpiqa: +0.090312
- cheap5_no_globalpiqa_reading: +0.102375
- relation_state: +0.051195
- BLiMP: -0.231231
- Supplement: +0.676944
- EWoK: -0.024428
- Entity: +0.126819
- COMPS: -0.036230
- GlobalPIQA: +1.264563
- Reading: +0.030000

`chck_82M` relative to the mean of `chck_80M` and `chck_84M` is included in JSON for comparison.

## 82→84 gain persistence and 84→86 reversal

| column | n | net 82→84 | gains 82→84 | losses 82→84 | gains lost by 86 | gains persist through 92 | net 84→86 | mean late flips | frac any late flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | -139 | 1015 | 1154 | 0.354 | 0.448 | 127 | 0.162 | 0.086 |
| Supplement | 5218 | -10 | 51 | 61 | 0.392 | 0.353 | 17 | 0.112 | 0.062 |
| EWoK | 7618 | -3 | 237 | 240 | 0.354 | 0.422 | 28 | 0.295 | 0.154 |
| Entity | 6780 | 11 | 134 | 123 | 0.425 | 0.276 | 10 | 0.182 | 0.096 |
| COMPS | 91028 | 82 | 3001 | 2919 | 0.352 | 0.441 | -2 | 0.325 | 0.172 |
| GlobalPIQA | 203 | 1 | 5 | 4 | 0.200 | 0.400 | -4 | 0.172 | 0.113 |

## Subtask-unit bootstrap lens

### 84_minus_82

- cheap7: median +0.1633, 90% interval [-0.3956, +0.7247], P(delta<=0)=0.261
- cheap6_no_globalpiqa: median +0.1015, 90% interval [-0.0526, +0.2607], P(delta<=0)=0.138
- cheap5_no_globalpiqa_reading: median +0.1208, 90% interval [-0.0642, +0.3118], P(delta<=0)=0.140
- relation_state: median +0.1418, 90% interval [-0.1682, +0.4401], P(delta<=0)=0.221
- syntax_surface: median -0.1162, 90% interval [-0.3138, +0.0885], P(delta<=0)=0.822

### 86_minus_84

- cheap7: median -0.3555, 90% interval [-0.5865, -0.1193], P(delta<=0)=0.996
- cheap6_no_globalpiqa: median -0.0843, 90% interval [-0.2638, +0.1087], P(delta<=0)=0.766
- cheap5_no_globalpiqa_reading: median -0.0901, 90% interval [-0.3056, +0.1414], P(delta<=0)=0.741
- relation_state: median +0.0358, 90% interval [-0.2617, +0.3237], P(delta<=0)=0.416
- syntax_surface: median +0.1569, 90% interval [-0.0606, +0.3737], P(delta<=0)=0.125

### 100_minus_84

- cheap7: median -0.5757, 90% interval [-1.1370, -0.0296], P(delta<=0)=0.965
- cheap6_no_globalpiqa: median -0.3375, 90% interval [-0.6444, -0.0565], P(delta<=0)=0.978
- cheap5_no_globalpiqa_reading: median -0.4380, 90% interval [-0.8063, -0.1008], P(delta<=0)=0.985
- relation_state: median -1.0429, 90% interval [-1.6455, -0.5096], P(delta<=0)=0.999
- syntax_surface: median +0.2497, 90% interval [-0.1915, +0.6407], P(delta<=0)=0.181

This bootstrap is a subtask-resampling stability lens for the deterministic benchmark aggregate, not an alternative official metric; Reading is included as an exact scalar in cheap6/cheap7, not resampled.

## Interpretation

`chck_84M` improves cheap7 over `chck_82M` by +0.163992 and then falls by -0.352133 at `chck_86M`; its local excess over the 82/86 neighbor mean is +0.258062. The peak remains visible without GlobalPIQA/Reading (cheap5 local excess +0.102375) but the relation/state local excess is +0.051195. The item table shows large churn relative to net movement; use this as evidence of late competence allocation rather than smooth monotone acquisition.

The pending SuperGLUE result will decide endpoint arithmetic, and seed43122 scoring is still needed before treating the late phase as robust beyond the protected seed/mask stream.

## Files

- `experiments/archive/frontier_consolidation/data/reference_late_item_dynamics/late_item_dynamics.csv` (23366360 bytes, sha256 `a054ceca0098434e…`)
- `experiments/archive/frontier_consolidation/data/reference_late_item_dynamics/score_curve.csv` (3034 bytes, sha256 `96e78acb08c43333…`)
- `experiments/archive/frontier_consolidation/data/reference_late_item_dynamics/subtask_score_curve.csv` (34409 bytes, sha256 `88208fff498cb127…`)
- `experiments/archive/frontier_consolidation/data/reference_late_item_dynamics/column_dynamics_summary.csv` (2296 bytes, sha256 `9a0c2d6dde7ee518…`)
- `experiments/archive/frontier_consolidation/data/reference_late_item_dynamics/column_subtask_dynamics_summary.csv` (30834 bytes, sha256 `0610543cbd488d38…`)
- `experiments/archive/frontier_consolidation/data/reference_late_item_dynamics/column_subgroup_dynamics_summary.csv` (12158 bytes, sha256 `900f91662419b56d…`)
- JSON summary: `experiments/archive/frontier_consolidation/data/reference_late_item_dynamics/reference_late_item_dynamics_summary.json`
