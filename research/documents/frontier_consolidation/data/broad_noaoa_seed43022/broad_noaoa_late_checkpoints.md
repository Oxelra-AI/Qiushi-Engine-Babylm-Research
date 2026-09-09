# existing ladder checkpoint route state — broad no-AoA late-checkpoint screen

Existing compact_view_reinvest checkpoints only; no training, corpus changes, SuperGLUE, AoA, or official leaderboard collation.

Summary JSON: `experiments/archive/frontier_consolidation/data/broad_noaoa_seed43022/broad_noaoa_late_checkpoints_summary.json`

## Scores
| target | BLiMP | Supp | EWoK-fast | Entity fast | Entity full | COMPS | GPIQA | Reading | equal7 fast | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r43022_80M | 66.630 | 66.000 | 51.090 | 28.770 | 27.600 | 51.780 | 35.135 | 8.160 | 43.938 | 43.771 |
| r43022_90M | 66.800 | 67.600 | 53.360 | 27.780 | 27.620 | 52.050 | 35.120 | 8.170 | 44.411 | 44.389 |

## Known 100M fast references (not reevaluated here)
| reference | BLiMP | Supp | EWoK-fast | Entity full | COMPS | GPIQA | Reading | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r43022_100M_fast | 66.630 | 66.400 | 53.090 | 27.750 | 51.970 | 35.620 | 8.240 | 44.243 |
| r43122_100M_fast | 65.750 | 63.200 | 49.360 | 26.290 | 51.540 | 35.135 | 8.865 | 42.877 |

## Contrasts
- **r43022_80M_minus_r43022_100M_fast**: equal7_full_entity -0.472, equal7_mean -0.351, BLiMP +0.000, Supplement -0.400, EWoK -2.000, Entity_full -0.150, COMPS -0.190, GlobalPIQA_mean -0.485, Reading -0.080
- **r43022_90M_minus_r43022_100M_fast**: equal7_full_entity +0.146, equal7_mean +0.123, BLiMP +0.170, Supplement +1.200, EWoK +0.270, Entity_full -0.130, COMPS +0.080, GlobalPIQA_mean -0.500, Reading -0.070
- **r43022_90M_minus_r43022_80M**: equal7_full_entity +0.618, equal7_mean +0.474, BLiMP +0.170, Supplement +1.600, EWoK +2.270, Entity_full +0.020, COMPS +0.270, GlobalPIQA_mean -0.015, Reading +0.010

## Scientific read
- This screen can select candidate existing checkpoints for official or averaging follow-up, but it cannot establish an Overall score because SuperGLUE, AoA, and current official EWoK/full collation are absent.
- Continue the stopping/averaging route only if a candidate improves Supplement without erasing EWoK, Entity, GlobalPIQA, Reading, and the broad no-AoA balance relative to its 100M reference.
