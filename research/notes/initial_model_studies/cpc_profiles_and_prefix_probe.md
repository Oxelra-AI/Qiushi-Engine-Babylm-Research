# cpc profiles and prefix probe CPC fast profiles and prefix-sensitivity probe

JSON: `experiments/archive/initial_model_studies/data/cpc_profiles_and_prefix_probe.json`
Rows: `experiments/archive/initial_model_studies/data/cpc_prefix_probe_rows.csv`

## Fast profile deltas vs matched WWM

| model | BLiMP | Supplement | EWoK | Entity | COMPS | Reading |
|---|---:|---:|---:|---:|---:|---:|
| cpc_same | +0.54 | -1.60 | +2.27 | +1.10 | -0.30 | -0.260 |
| cpc_cross | +0.75 | -0.40 | +1.54 | -0.70 | -0.41 | +0.030 |

## Training margin endpoints

- cpc_same: mode=None, same=1.0, cross=None, cpc_loss=0.5018, active=1.0000, lp_diff=-0.00179
- cpc_cross: mode=cross, same=0.0, cross=1.0, cpc_loss=0.5018, active=1.0000, lp_diff=-0.00183

## Probe aggregate_all

| metric | mean | median | pos frac | n |
|---|---:|---:|---:|---:|
| same_minus_wwm_spec_same_vs_cross | -0.65829 | -0.53875 | 0.263 | 240 |
| cross_minus_wwm_spec_same_vs_cross | +0.00033 | +0.00004 | 0.633 | 240 |

## Probe aggregate_low_wwm_sensitivity

| metric | mean | median | pos frac | n |
|---|---:|---:|---:|---:|
| same_minus_wwm_spec_same_vs_cross | -0.33456 | -0.00001 | 0.450 | 60 |
| cross_minus_wwm_spec_same_vs_cross | +0.00004 | +0.00000 | 0.517 | 60 |

## Probe aggregate_high_wwm_sensitivity

| metric | mean | median | pos frac | n |
|---|---:|---:|---:|---:|
| same_minus_wwm_spec_same_vs_cross | -0.83927 | -0.90540 | 0.117 | 60 |
| cross_minus_wwm_spec_same_vs_cross | +0.00059 | +0.00030 | 0.600 | 60 |
