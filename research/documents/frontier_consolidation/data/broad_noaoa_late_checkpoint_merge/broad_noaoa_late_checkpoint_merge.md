# existing ladder checkpoint route state — merged broad no-AoA late-checkpoint screen

This merge combines the two split GPU evaluation roots for selected existing compact_view_reinvest checkpoints. It is not a full official score because SuperGLUE, AoA, and full current official collation are absent.

Summary JSON: `experiments/archive/frontier_consolidation/data/broad_noaoa_late_checkpoint_merge/broad_noaoa_late_checkpoint_merge.json`


## Candidate checkpoint scores
| target | BLiMP | Supp | EWoK-fast | Entity full | COMPS | GPIQA | Reading | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r43022_80M | 66.630 | 66.000 | 51.090 | 27.600 | 51.780 | 35.135 | 8.160 | 43.771 |
| r43022_90M | 66.800 | 67.600 | 53.360 | 27.620 | 52.050 | 35.120 | 8.170 | 44.389 |
| r43122_45M | 62.700 | 65.600 | 48.000 | 25.530 | 51.470 | 38.550 | 8.900 | 42.964 |
| r43122_80M | 65.400 | 65.200 | 49.730 | 26.350 | 51.470 | 36.590 | 8.935 | 43.382 |

## Known 100M fast references
| reference | BLiMP | Supp | EWoK-fast | Entity full | COMPS | GPIQA | Reading | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r43022_100M_fast | 66.630 | 66.400 | 53.090 | 27.750 | 51.970 | 35.620 | 8.240 | 44.243 |
| r43122_100M_fast | 65.750 | 63.200 | 49.360 | 26.290 | 51.540 | 35.135 | 8.865 | 42.877 |

## Contrasts vs own 100M fast reference
- **r43022_80M_minus_r43022_100M_fast**: equal7_full_entity -0.472, BLiMP +0.000, Supplement -0.400, EWoK -2.000, Entity_full -0.150, COMPS -0.190, GlobalPIQA_mean -0.485, Reading -0.080
- **r43022_90M_minus_r43022_100M_fast**: equal7_full_entity +0.146, BLiMP +0.170, Supplement +1.200, EWoK +0.270, Entity_full -0.130, COMPS +0.080, GlobalPIQA_mean -0.500, Reading -0.070
- **r43122_45M_minus_r43122_100M_fast**: equal7_full_entity +0.087, BLiMP -3.050, Supplement +2.400, EWoK -1.360, Entity_full -0.760, COMPS -0.070, GlobalPIQA_mean +3.415, Reading +0.035
- **r43122_80M_minus_r43122_100M_fast**: equal7_full_entity +0.505, BLiMP -0.350, Supplement +2.000, EWoK +0.370, Entity_full +0.060, COMPS -0.070, GlobalPIQA_mean +1.455, Reading +0.070
- **r43122_80M_minus_r43022_80M**: equal7_full_entity -0.389, BLiMP -1.230, Supplement -0.800, EWoK -1.360, Entity_full -1.250, COMPS -0.310, GlobalPIQA_mean +1.455, Reading +0.775
- **r43022_90M_minus_r43022_80M**: equal7_full_entity +0.618, BLiMP +0.170, Supplement +1.600, EWoK +2.270, Entity_full +0.020, COMPS +0.270, GlobalPIQA_mean -0.015, Reading +0.010
- **r43122_45M_minus_r43122_80M**: equal7_full_entity -0.418, BLiMP -2.700, Supplement +0.400, EWoK -1.730, Entity_full -0.820, COMPS +0.000, GlobalPIQA_mean +1.960, Reading -0.035

## Scientific read
- At 80M, equal7_full_entity seed-min 43.382 vs known 100M seed-min 42.877, seed-mean 43.576 vs 43.560; seed43022 delta -0.4721 and seed43122 delta 0.505.
- r43022_80M vs r43022_100M_fast: Supplement delta -0.4, equal7_full_entity delta -0.4721, notable broad losses ['EWoK']; Supplement-only gain does not exist in this fast surface.
- r43122_80M vs r43122_100M_fast: Supplement delta 2.0, equal7_full_entity delta 0.505, notable broad losses none >=0.5; Supplement-only gain exists in this fast surface.
- r43122_45M vs r43122_100M_fast: Supplement delta 2.4, equal7_full_entity delta 0.0871, notable broad losses ['EWoK', 'Entity_full', 'BLiMP']; Supplement-only gain exists in this fast surface.
- r43022_90M vs r43022_100M_fast: Supplement delta 1.2, equal7_full_entity delta 0.1457, notable broad losses ['GlobalPIQA_mean']; Supplement-only gain exists in this fast surface.
- The 80M stopping candidate does not preserve the broad fast surface strongly enough by itself; checkpoint averaging should only proceed if it directly addresses the measured losses, otherwise return to mechanism construction rather than a broad stopping/full-eval route.
