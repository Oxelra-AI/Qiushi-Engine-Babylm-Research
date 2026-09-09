# density eval repair density-overlay no-AoA task-family screen

Summary JSON: `experiments/archive/frontier_consolidation/data/density_noaoa_eval_reinvest/density_noaoa_eval_summary.json`

This screen asks whether FineWeb second views on a clean-Qwen row-holdout base improve BabyLM task families before any full official evaluation.

| target | BLiMP | Supp | EWoK | Entity fast | Entity full | COMPS | GPIQA | Reading | equal7 fast | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact_view_core | 66.930 | 65.600 | 51.550 | 27.300 | 27.850 | 52.180 | 35.135 | 8.250 | 43.849 | 43.928 |
| compact_view_reinvest | 66.630 | 66.400 | 53.090 | 28.070 | 27.750 | 51.970 | 35.620 | 8.240 | 44.289 | 44.243 |

## Mechanism contrasts

- **compact_view_reinvest_minus_compact_view_core**: equal7_mean +0.439, equal7_full_entity +0.315, BLiMP -0.300, Supplement +0.800, EWoK +1.540, Entity +0.770, Entity_full -0.100, COMPS -0.210, GlobalPIQA_mean +0.485, Reading -0.010
- **compact_view_core_minus_visible_leader_columns**: BLiMP -0.270, Supplement +9.590, EWoK -4.520, Entity -1.150, COMPS -1.390, GlobalPIQA_mean -4.535, Reading +2.830
- **compact_view_reinvest_minus_visible_leader_columns**: BLiMP -0.570, Supplement +10.390, EWoK -2.980, Entity -0.380, COMPS -1.600, GlobalPIQA_mean -4.050, Reading +2.820
