# identity orbit randomization and role coordinate full-surface family-alias control

Device `cuda`, elapsed 44.19 s. Same expanded full-sentence task, family-stable aliases only.

| mode | arm | k/template | fit | train | probe-held | probe-trainfam | held-template |
|---|---:|---:|---:|---:|---:|---:|---:|
| family_alias | zero | 0 | 5/5 | 0.997 | 0.559 | 0.645 | 0.567 |
| family_alias | true | 8 | 5/5 | 1.000 | 0.552 | 0.620 | 0.544 |
| family_alias | shuffled | 8 | 5/5 | 0.998 | 0.423 | 0.439 | 0.575 |
| family_alias | exposure | 8 | 5/5 | 0.997 | 0.492 | 0.550 | 0.505 |

Summary JSON: `experiments/archive/representation_and_objectives/data/full_surface_family_alias_control/full_surface_family_alias_control_summary.json`
