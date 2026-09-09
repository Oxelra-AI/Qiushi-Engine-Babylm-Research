# cadence all arms fast profile cadence fixed-256 reference vs combined schedule

JSON: `experiments/archive/initial_model_studies/data/cadence_fixed_vs_combined_fast_profile.json`
Log: `research/notes/initial_model_studies/cadence_fixed_vs_combined_fast_profile.log`

## Matching and training summaries

Config equal: True
Param counts: {'fixed256_ref': 34471264, 'combined': 34471264}
Word exposure: {'fixed256_ref': 1000000, 'combined': 1000000}
Steps: {'fixed256_ref': 49, 'combined': 49}
Log summaries: `{"fixed256_ref": {"n_log_rows": 49, "active_tokens_total": 1335322, "masked_targets_total": 199522, "active_tokens_mean": 27251.469387755104, "masked_targets_mean": 4071.877551020408, "unique_seq_lens": [256], "mask_prob_minmax": [0.15, 0.15]}, "combined": {"n_log_rows": 49, "active_tokens_total": 1122911, "masked_targets_total": 244267, "active_tokens_mean": 22916.551020408162, "masked_targets_mean": 4985.040816326531, "unique_seq_lens": [128, 256, 512], "mask_prob_minmax": [0.1531, 0.3]}}`

## Fast profile: combined - fixed256_ref

| BLiMP | Supplement | EWoK | Entity | COMPS | Reading |
|---:|---:|---:|---:|---:|---:|
| +0.18 | +4.00 | -3.54 | -0.44 | -0.09 | -0.120 |

## Raw scores

| model | BLiMP | Supplement | EWoK | Entity | COMPS | Reading |
|---|---:|---:|---:|---:|---:|---:|
| fixed256_ref | 56.54 | 47.60 | 52.27 | 16.85 | 50.03 | 5.995 |
| combined | 56.72 | 51.60 | 48.73 | 16.41 | 49.94 | 5.875 |
