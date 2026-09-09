# existing ladder checkpoint route state — broad no-AoA late-checkpoint screen

Existing compact_view_reinvest checkpoints only; no training, corpus changes, SuperGLUE, AoA, or official leaderboard collation.

Summary JSON: `experiments/archive/frontier_consolidation/data/broad_noaoa_seed43122/broad_noaoa_late_checkpoints_summary.json`

## Scores
| target | BLiMP | Supp | EWoK-fast | Entity fast | Entity full | COMPS | GPIQA | Reading | equal7 fast | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r43122_45M | 62.700 | 65.600 | 48.000 | 27.100 | 25.530 | 51.470 | 38.550 | 8.900 | 43.189 | 42.964 |
| r43122_80M | 65.400 | 65.200 | 49.730 | 26.930 | 26.350 | 51.470 | 36.590 | 8.935 | 43.465 | 43.382 |

## Known 100M fast references (not reevaluated here)
| reference | BLiMP | Supp | EWoK-fast | Entity full | COMPS | GPIQA | Reading | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r43022_100M_fast | 66.630 | 66.400 | 53.090 | 27.750 | 51.970 | 35.620 | 8.240 | 44.243 |
| r43122_100M_fast | 65.750 | 63.200 | 49.360 | 26.290 | 51.540 | 35.135 | 8.865 | 42.877 |

## Contrasts
- **r43122_45M_minus_r43122_100M_fast**: equal7_full_entity +0.087, equal7_mean +0.256, BLiMP -3.050, Supplement +2.400, EWoK -1.360, Entity_full -0.760, COMPS -0.070, GlobalPIQA_mean +3.415, Reading +0.035
- **r43122_80M_minus_r43122_100M_fast**: equal7_full_entity +0.505, equal7_mean +0.532, BLiMP -0.350, Supplement +2.000, EWoK +0.370, Entity_full +0.060, COMPS -0.070, GlobalPIQA_mean +1.455, Reading +0.070
- **r43122_45M_minus_r43122_80M**: equal7_full_entity -0.418, equal7_mean -0.276, BLiMP -2.700, Supplement +0.400, EWoK -1.730, Entity_full -0.820, COMPS +0.000, GlobalPIQA_mean +1.960, Reading -0.035

## Scientific read
- This screen can select candidate existing checkpoints for official or averaging follow-up, but it cannot establish an Overall score because SuperGLUE, AoA, and current official EWoK/full collation are absent.
- Continue the stopping/averaging route only if a candidate improves Supplement without erasing EWoK, Entity, GlobalPIQA, Reading, and the broad no-AoA balance relative to its 100M reference.
