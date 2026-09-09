# repaired pilot eval — Repaired pilot fast evaluation

Summary JSON: `experiments/archive/frontier_consolidation/data/repaired_pilot_eval/repaired_pilot_eval_summary.json`

This uses the official-compatible fast sentence-zero-shot and Reading calls to decide whether the repaired leader-style training dynamics deserve larger H100 runs. It does not include SuperGLUE or AoA.

## Fast-screen table

| target | BLiMP | Supp | EWoK | Entity fast | Entity full | COMPS | GPIQA | Reading | equal7 fast | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stagewise_qwen10_12x384_40k_seed44011 | 53.950 | 55.600 | 48.180 | 17.290 | 16.350 | 49.440 | 35.635 | 6.795 | 38.127 | 37.993 |
| pairfit_qwen10_12x384_40k_seed44011 | 54.860 | 56.800 | 48.360 | 17.300 | 17.430 | 49.390 | 33.725 | 6.750 | 38.169 | 38.188 |

## Contrasts

- **stagewise_qwen10_12x384_40k_seed44011_minus_leader_card_columns**: BLiMP -13.250, Supplement -0.410, EWoK -7.890, Entity -11.160, COMPS -4.130, GlobalPIQA_mean -4.035, Reading +1.375
- **pairfit_qwen10_12x384_40k_seed44011_minus_leader_card_columns**: BLiMP -12.340, Supplement +0.790, EWoK -7.710, Entity -11.150, COMPS -4.180, GlobalPIQA_mean -5.945, Reading +1.330
