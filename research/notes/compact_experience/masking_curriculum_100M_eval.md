# masking curriculum 100M eval — Regime-correct masking-curriculum 100M fast evaluation

Summary JSON: `experiments/archive/compact_experience/data/curriculum_100M_eval/curriculum_100M_eval_summary.json`

This is a mechanism-screen evaluation of the four 100M-exposure official-corpus runs. It is not official Overall: SuperGLUE and AoA are absent, most zero-shot columns use fast subsets, and COMPS uses `full_eval/comps`.

## chck_100M scores

| target | BLiMP | Supplement | EWoK | Entity | COMPS | GPIQA mean | Reading | weighted fast proxy | loss_last | late mode/eff mask |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| wwm_fixed__chck_100M | 63.490 | 66.000 | 49.450 | 20.560 | 50.730 | 32.740 | 7.445 | 31.249 | 3.263 | wwm/0.149 |
| wwm_to_token__chck_100M | 64.430 | 60.800 | 50.550 | 18.660 | 50.690 | 36.150 | 6.985 | 31.010 | 2.772 | token/0.150 |
| amlm_hard__chck_100M | 66.220 | 60.800 | 46.640 | 19.900 | 51.360 | 37.580 | 6.855 | 31.125 | 3.549 | wwm/0.153 |
| amlm_hard_switch__chck_100M | 65.970 | 60.400 | 49.000 | 20.350 | 51.170 | 33.635 | 6.720 | 30.896 | 3.225 | token/0.147 |

## chck_100M contrasts

| contrast | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | weighted fast proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wwm_to_token_minus_wwm_fixed | 0.940 | -5.200 | 1.100 | -1.900 | -0.040 | 3.410 | -0.460 | -0.239 |
| amlm_hard_minus_wwm_fixed | 2.730 | -5.200 | -2.810 | -0.660 | 0.630 | 4.840 | -0.590 | -0.124 |
| amlm_hard_switch_minus_wwm_fixed | 2.480 | -5.600 | -0.450 | -0.210 | 0.440 | 0.895 | -0.725 | -0.353 |
| amlm_hard_switch_minus_amlm_hard | -0.250 | -0.400 | 2.360 | 0.450 | -0.190 | -3.945 | -0.135 | -0.228 |
| amlm_hard_minus_wwm_to_token | 1.790 | 0.000 | -3.910 | 1.240 | 0.670 | 1.430 | -0.130 | 0.114 |
| amlm_hard_switch_minus_wwm_to_token | 1.540 | -0.400 | -1.550 | 1.690 | 0.480 | -2.515 | -0.265 | -0.114 |

Weighted fast proxy = (3/28) * (BLiMP + Supplement + EWoK + Entity + COMPS + GlobalPIQA_mean) + (1/8) * Reading. It is a screening statistic only.

## Dynamics paths

- `wwm_fixed`: metrics `experiments/archive/compact_experience/training/runs/wwm_fixed_100M_seed43/scientific_metrics.json`, dynamics `experiments/archive/compact_experience/training/runs/wwm_fixed_100M_seed43/dynamics_traces.jsonl`
- `wwm_to_token`: metrics `experiments/archive/compact_experience/training/runs/wwm_to_token_100M_seed43/scientific_metrics.json`, dynamics `experiments/archive/compact_experience/training/runs/wwm_to_token_100M_seed43/dynamics_traces.jsonl`
- `amlm_hard`: metrics `experiments/archive/compact_experience/training/runs/amlm_hard_100M_seed43/scientific_metrics.json`, dynamics `experiments/archive/compact_experience/training/runs/amlm_hard_100M_seed43/dynamics_traces.jsonl`
- `amlm_hard_switch`: metrics `experiments/archive/compact_experience/training/runs/amlm_hard_switch_100M_seed43/scientific_metrics.json`, dynamics `experiments/archive/compact_experience/training/runs/amlm_hard_switch_100M_seed43/dynamics_traces.jsonl`
