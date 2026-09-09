# cont profile and prefix probe WWM + strict prefix-continuation validation

JSON: `experiments/archive/initial_model_studies/data/cont_profile_and_prefix_probe.json`
Rows: `experiments/archive/initial_model_studies/data/cont_prefix_probe_rows.csv`

## Fast profile: continuation - matched WWM

| BLiMP | Supplement | EWoK | Entity | COMPS | Reading |
|---:|---:|---:|---:|---:|---:|
| -0.56 | -0.80 | +0.28 | -0.48 | -0.66 | -0.155 |

## Training/visibility

Visibility status: PASS; later suffix change max diff 0.0; earlier suffix 0.011602148413658142; prefix removal 0.4068615138530731.
Endpoint: WWM 6.276214599609375; continuation model WWM 5.992833614349365; continuation loss 5.979063510894775.

## Matched prefix probe aggregates

| metric | mean | median | pos frac | n |
|---|---:|---:|---:|---:|

### aggregate_all

| metric | mean | median | pos frac | n |
|---|---:|---:|---:|---:|
| cont_minus_wwm_spec_same_vs_cross | -0.73458 | -0.72997 | 0.200 | 240 |
| cont_minus_wwm_spec_deleted_vs_cross | -0.73453 | -0.72958 | 0.212 | 240 |

### aggregate_low_wwm_sensitivity

| metric | mean | median | pos frac | n |
|---|---:|---:|---:|---:|
| cont_minus_wwm_spec_same_vs_cross | -0.45883 | -0.00001 | 0.267 | 60 |
| cont_minus_wwm_spec_deleted_vs_cross | -0.45883 | -0.00001 | 0.300 | 60 |

### aggregate_high_wwm_sensitivity

| metric | mean | median | pos frac | n |
|---|---:|---:|---:|---:|
| cont_minus_wwm_spec_same_vs_cross | -0.68037 | -0.74889 | 0.167 | 60 |
| cont_minus_wwm_spec_deleted_vs_cross | -0.68023 | -0.74888 | 0.200 | 60 |
