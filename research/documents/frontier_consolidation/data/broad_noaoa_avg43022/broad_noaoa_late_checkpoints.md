# existing ladder checkpoint route state — broad no-AoA late-checkpoint screen

Existing compact_view_reinvest checkpoints only; no training, corpus changes, SuperGLUE, AoA, or official leaderboard collation.

Summary JSON: `experiments/archive/frontier_consolidation/data/broad_noaoa_avg43022/broad_noaoa_late_checkpoints_summary.json`

## Scores
| target | BLiMP | Supp | EWoK-fast | Entity fast | Entity full | COMPS | GPIQA | Reading | equal7 fast | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| avg43022_90_100 | 66.730 | 66.800 | 53.550 | 28.210 | 27.850 | 52.070 | 34.620 | 8.210 | 44.313 | 44.261 |

## Known 100M fast references (not reevaluated here)
| reference | BLiMP | Supp | EWoK-fast | Entity full | COMPS | GPIQA | Reading | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r43022_100M_fast | 66.630 | 66.400 | 53.090 | 27.750 | 51.970 | 35.620 | 8.240 | 44.243 |
| r43122_100M_fast | 65.750 | 63.200 | 49.360 | 26.290 | 51.540 | 35.135 | 8.865 | 42.877 |

## Contrasts
- **avg43022_90_100_minus_r43022_100M_fast**: equal7_full_entity +0.019, equal7_mean +0.024, BLiMP +0.100, Supplement +0.400, EWoK +0.460, Entity_full +0.100, COMPS +0.100, GlobalPIQA_mean -1.000, Reading -0.030

## Scientific read
- This screen can select candidate existing checkpoints for official or averaging follow-up, but it cannot establish an Overall score because SuperGLUE, AoA, and current official EWoK/full collation are absent.
- Continue the stopping/averaging route only if a candidate improves Supplement without erasing EWoK, Entity, GlobalPIQA, Reading, and the broad no-AoA balance relative to its 100M reference.
