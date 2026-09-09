# density eval repair density-overlay no-AoA task-family screen

Summary JSON: `experiments/archive/frontier_consolidation/data/density_noaoa_eval_compact_core/density_noaoa_eval_summary.json`

This screen asks whether FineWeb second views on a clean-Qwen row-holdout base improve BabyLM task families before any full official evaluation.

| target | BLiMP | Supp | EWoK | Entity fast | Entity full | COMPS | GPIQA | Reading | equal7 fast | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact_repeat_core | 67.090 | 58.000 | 48.640 | 25.110 | 25.420 | 51.090 | 34.105 | 8.020 | 41.722 | 41.766 |
| compact_view_core | 66.930 | 65.600 | 51.550 | 27.300 | 27.850 | 52.180 | 35.135 | 8.250 | 43.849 | 43.928 |

## Mechanism contrasts

- **compact_view_core_minus_compact_repeat_core**: equal7_mean +2.127, equal7_full_entity +2.161, BLiMP -0.160, Supplement +7.600, EWoK +2.910, Entity +2.190, Entity_full +2.430, COMPS +1.090, GlobalPIQA_mean +1.030, Reading +0.230
- **compact_repeat_core_minus_visible_leader_columns**: BLiMP -0.110, Supplement +1.990, EWoK -7.430, Entity -3.340, COMPS -2.480, GlobalPIQA_mean -5.565, Reading +2.600
- **compact_view_core_minus_visible_leader_columns**: BLiMP -0.270, Supplement +9.590, EWoK -4.520, Entity -1.150, COMPS -1.390, GlobalPIQA_mean -4.535, Reading +2.830
