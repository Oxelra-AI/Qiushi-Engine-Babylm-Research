# density eval repair density-overlay no-AoA task-family screen

Summary JSON: `experiments/archive/frontier_consolidation/data/density_noaoa_eval_retry/density_noaoa_eval_summary.json`

This screen asks whether FineWeb second views on a clean-Qwen row-holdout base improve BabyLM task families before any full official evaluation.

| target | BLiMP | Supp | EWoK | Entity fast | Entity full | COMPS | GPIQA | Reading | equal7 fast | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| near_repeat | 67.190 | 62.400 | 49.450 | 23.250 | 23.560 | 51.780 | 34.150 | 7.745 | 42.281 | 42.325 |
| near_view | 66.820 | 63.200 | 49.180 | 27.970 | 27.130 | 51.170 | 34.135 | 8.520 | 42.999 | 42.879 |

## Mechanism contrasts

- **near_view_minus_near_repeat**: equal7_mean +0.719, equal7_full_entity +0.554, BLiMP -0.370, Supplement +0.800, EWoK -0.270, Entity +4.720, Entity_full +3.570, COMPS -0.610, GlobalPIQA_mean -0.015, Reading +0.775
- **near_repeat_minus_visible_leader_columns**: BLiMP -0.010, Supplement +6.390, EWoK -6.620, Entity -5.200, COMPS -1.790, GlobalPIQA_mean -5.520, Reading +2.325
- **near_view_minus_visible_leader_columns**: BLiMP -0.380, Supplement +7.190, EWoK -6.890, Entity -0.480, COMPS -2.400, GlobalPIQA_mean -5.535, Reading +3.100
