# prefix target census v2 fixed-position prefix-dependent target census

Rows: `experiments/archive/initial_model_studies/data/prefix_target_census_v2_fixedpos_rows.csv`
Summary: `experiments/archive/initial_model_studies/data/prefix_target_census_v2_fixedpos_summary.json`

This repairs the v1 artifact where replacement/deleted prefixes changed target absolute position. All variants now keep suffix and target at the same positions.

| exposure | n | mean stable Δ(full-same) | median | ≥0.02 | ≥0.05 | ≥0.10 | ≥0.20 | neg≤-0.05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 40M | 600 | +0.04370 | +0.00000 | 0.3017 | 0.2583 | 0.2133 | 0.1400 | 0.2117 |
| 80M | 600 | +0.08325 | +0.00000 | 0.3233 | 0.2733 | 0.2433 | 0.1617 | 0.1683 |
| 100M | 600 | +0.08166 | +0.00000 | 0.3283 | 0.2867 | 0.2433 | 0.1650 | 0.1833 |
