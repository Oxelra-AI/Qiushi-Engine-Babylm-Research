# existing ladder checkpoint route state — checkpoint-average broad no-AoA merge

Arithmetic averaged weights from existing compact_view_reinvest checkpoints; no training or corpus modification. Not a full official score because SuperGLUE, AoA, and current official collation are absent.

Summary JSON: `experiments/archive/frontier_consolidation/data/checkpoint_average_noaoa_merge/checkpoint_average_noaoa_merge.json`


## Averaged-model scores
| target | BLiMP | Supp | EWoK-fast | Entity full | COMPS | GPIQA | Reading | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| avg43022_90_100 | 66.730 | 66.800 | 53.550 | 27.850 | 52.070 | 34.620 | 8.210 | 44.261 |
| avg43122_80_100 | 65.340 | 64.800 | 50.270 | 26.190 | 51.600 | 36.090 | 8.915 | 43.315 |

## Contrasts
- **avg43022_90_100_minus_r43022_100M_fast**: equal7_full_entity +0.019, BLiMP +0.100, Supplement +0.400, EWoK +0.460, Entity_full +0.100, COMPS +0.100, GlobalPIQA_mean -1.000, Reading -0.030
- **avg43022_90_100_minus_r43022_90M**: equal7_full_entity -0.127, BLiMP -0.070, Supplement -0.800, EWoK +0.190, Entity_full +0.230, COMPS +0.020, GlobalPIQA_mean -0.500, Reading +0.040
- **avg43122_80_100_minus_r43122_100M_fast**: equal7_full_entity +0.438, BLiMP -0.410, Supplement +1.600, EWoK +0.910, Entity_full -0.100, COMPS +0.060, GlobalPIQA_mean +0.955, Reading +0.050
- **avg43122_80_100_minus_r43122_80M**: equal7_full_entity -0.067, BLiMP -0.060, Supplement -0.400, EWoK +0.540, Entity_full -0.160, COMPS +0.130, GlobalPIQA_mean -0.500, Reading -0.020
- **avg43122_80_100_minus_avg43022_90_100**: equal7_full_entity -0.946, BLiMP -1.390, Supplement -2.000, EWoK -3.280, Entity_full -1.660, COMPS -0.470, GlobalPIQA_mean +1.470, Reading +0.705

## Scientific read
- avg43022_90_100: vs own 100M fast reference equal7_full_entity 0.0186, Supplement 0.4, EWoK 0.46, Entity_full 0.1, GlobalPIQA_mean -1.0; vs motivating raw r43022_90M equal7_full_entity -0.1271, Supplement -0.8, EWoK 0.19, GlobalPIQA_mean -0.5.
- avg43122_80_100: vs own 100M fast reference equal7_full_entity 0.4379, Supplement 1.6, EWoK 0.91, Entity_full -0.1, GlobalPIQA_mean 0.955; vs motivating raw r43122_80M equal7_full_entity -0.0671, Supplement -0.4, EWoK 0.54, GlobalPIQA_mean -0.5.
- Checkpoint averaging improved the measured broad fast surface in both seeds without an EWoK loss large enough to reject the route; next evidence should be a narrow official-coordinate check on the better average or raw checkpoint, starting with full Supplement/EWoK and then SuperGLUE/AoA only if the no-AoA surface remains competitive.
