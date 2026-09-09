# prefix target census prefix-dependent target census

Rows: `experiments/archive/initial_model_studies/data/prefix_target_census_rows.csv`
Summary: `experiments/archive/initial_model_studies/data/prefix_target_census_summary.json`

Cases: 600 (300 exact 1M target, 300 broad official target requested).

## Sign-stable true-prefix over same-source-near fractions

| exposure | n | mean stable Δ | median | ≥0.02 | ≥0.05 | ≥0.10 | ≥0.20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 40M | 600 | +0.08122 | +0.00000 | 0.3133 | 0.2867 | 0.2400 | 0.1700 |
| 80M | 600 | +0.09528 | +0.00000 | 0.3367 | 0.2917 | 0.2517 | 0.1783 |
| 100M | 600 | +0.10539 | +0.00000 | 0.3400 | 0.2917 | 0.2533 | 0.2000 |

## Initial interpretation prompt

A usable cross-sentence objective target population would require a nontrivial fraction of targets where `full - same_source_near` is positive with the same sign across seed42 and seed43 at the same exposure. Cross-source differences alone do not count. Inspect rows before using any target set for training.
