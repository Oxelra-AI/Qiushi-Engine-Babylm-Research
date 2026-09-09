# cadence all arms interpretation current-best direct checkpoint recheck

JSON: `experiments/archive/initial_model_studies/data/current_best_direct_checkpoint_recheck.json`
Log: `research/notes/initial_model_studies/current_best_direct_checkpoint_recheck.log`

Exact checkpoint directories were passed as `--model_path_or_name`; no local parent+revision loading.

## Scores

| checkpoint | BLiMP | Supplement | EWoK | Entity | COMPS | Reading | mean6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| chck_80M | 66.77 | 65.60 | 49.64 | 20.76 | 53.00 | 7.300 | 43.845 |
| chck_100M | 67.34 | 65.20 | 49.64 | 21.24 | 53.11 | 7.330 | 43.977 |

## chck_80M - chck_100M

| BLiMP | Supplement | EWoK | Entity | COMPS | Reading | mean6 |
|---:|---:|---:|---:|---:|---:|---:|
| -0.57 | +0.40 | +0.00 | -0.48 | -0.11 | -0.030 | -0.132 |
