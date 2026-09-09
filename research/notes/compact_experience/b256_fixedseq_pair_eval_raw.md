# masking curriculum 100M eval — Regime-correct masking-curriculum 100M fast evaluation

Summary JSON: `experiments/archive/compact_experience/data/b256_pair_eval/b256_pair_100M_eval_summary.json`

This is a mechanism-screen evaluation of the four 100M-exposure official-corpus runs. It is not official Overall: SuperGLUE and AoA are absent, most zero-shot columns use fast subsets, and COMPS uses `full_eval/comps`.

## chck_100M scores

| target | BLiMP | Supplement | EWoK | Entity | COMPS | GPIQA mean | Reading | weighted fast proxy | loss_last | late mode/eff mask |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| wwm_fixed__chck_100M | 67.360 | 57.200 | 50.910 | 24.950 | 52.720 | 39.580 | 8.230 | 32.392 | 2.593 | wwm/0.150 |
| wwm_to_token__chck_100M | 67.630 | 61.200 | 50.450 | 23.780 | 52.180 | 39.580 | 7.885 | 32.573 | 2.431 | token/0.150 |

## chck_100M contrasts

| contrast | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | weighted fast proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wwm_to_token_minus_wwm_fixed | 0.270 | 4.000 | -0.460 | -1.170 | -0.540 | 0.000 | -0.345 | 0.182 |

Weighted fast proxy = (3/28) * (BLiMP + Supplement + EWoK + Entity + COMPS + GlobalPIQA_mean) + (1/8) * Reading. It is a screening statistic only.

## Dynamics paths

- `wwm_fixed`: metrics `experiments/archive/compact_experience/training/runs/wwm_fixed_100M_b256_seq256_seed43/scientific_metrics.json`, dynamics `experiments/archive/compact_experience/training/runs/wwm_fixed_100M_b256_seq256_seed43/dynamics_traces.jsonl`
- `wwm_to_token`: metrics `experiments/archive/compact_experience/training/runs/wwm_to_token_100M_b256_seq256_seed43/scientific_metrics.json`, dynamics `experiments/archive/compact_experience/training/runs/wwm_to_token_100M_b256_seq256_seed43/dynamics_traces.jsonl`
