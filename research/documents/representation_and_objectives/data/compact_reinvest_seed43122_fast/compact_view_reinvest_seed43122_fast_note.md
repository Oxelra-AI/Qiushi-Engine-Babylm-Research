# compact reinvest full eval summary compact_view_reinvest_seed43122 fast no-AoA screen

Summary JSON: `experiments/archive/representation_and_objectives/data/compact_reinvest_seed43122_fast/compact_view_reinvest_seed43122_fast_summary.json`

This is an independent-seed task-family screen. It does not include SuperGLUE or AoA.

| target | BLiMP | Supp | EWoK | Entity fast | Entity full | COMPS | GPIQA | Reading | equal7 fast | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact_view_reinvest_seed43122 | 65.750 | 63.200 | 49.360 | 26.680 | 26.290 | 51.540 | 35.135 | 8.865 | 42.933 | 42.877 |
| a02_seed43022_compact_view_reinvest | 66.630 | 66.400 | 53.090 | 28.070 | 27.750 | 51.970 | 35.620 | 8.240 | 44.289 | 44.243 |
| a02_seed43022_compact_view_core | 66.930 | 65.600 | 51.550 | 27.300 | 27.850 | 52.180 | 35.135 | 8.250 | 43.849 | 43.928 |

## Contrasts

- **compact_view_reinvest_seed43122_minus_a02_seed43022_compact_view_reinvest**: equal7_mean -1.356, equal7_full_entity -1.366, BLiMP -0.880, Supplement -3.200, EWoK -3.730, Entity -1.390, Entity_full -1.460, COMPS -0.430, GlobalPIQA_mean -0.485, Reading +0.625
- **compact_view_reinvest_seed43122_minus_a02_seed43022_compact_view_core**: equal7_mean -0.916, equal7_full_entity -1.051, BLiMP -1.180, Supplement -2.400, EWoK -2.190, Entity -0.620, Entity_full -1.560, COMPS -0.640, GlobalPIQA_mean +0.000, Reading +0.615
- **compact_view_reinvest_seed43122_minus_visible_leader_card**: BLiMP -1.450, Supplement +7.190, EWoK -6.710, Entity -1.770, COMPS -2.030, GlobalPIQA_mean -4.535, Reading +3.445
- **compact_view_reinvest_seed43122_minus_compact_experience_clean_qwen_full**: BLiMP -1.090, Supplement +0.360, EWoK -0.830, Entity +0.920, COMPS -0.240, GlobalPIQA_mean -1.485, Reading +1.105
