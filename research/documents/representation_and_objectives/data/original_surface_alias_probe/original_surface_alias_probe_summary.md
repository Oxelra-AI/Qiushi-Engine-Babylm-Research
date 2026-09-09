# identity orbit randomization and role coordinate original-surface alias probe

Device `cuda`, elapsed 55.23 s. Uses actual paired world pilot result score-ablated sentences and assigned templates.

| mode | arm | k/template | fit | train | probe-held | held-template |
|---|---:|---:|---:|---:|---:|---:|
| fixed_names | zero | 0 | 0/5 | 0.558 | 0.505 | 0.498 |
| fixed_names | true | 8 | 0/5 | 0.558 | 0.500 | 0.499 |
| fixed_names | shuffled | 8 | 0/5 | 0.718 | 0.474 | 0.518 |
| fixed_names | exposure | 8 | 0/5 | 0.758 | 0.472 | 0.520 |
| per_item_alias | zero | 0 | 5/5 | 0.999 | 0.500 | 0.543 |
| per_item_alias | true | 8 | 5/5 | 0.997 | 0.538 | 0.518 |
| per_item_alias | shuffled | 8 | 5/5 | 0.997 | 0.515 | 0.539 |
| per_item_alias | exposure | 8 | 5/5 | 0.996 | 0.505 | 0.527 |

Summary JSON: `experiments/archive/representation_and_objectives/data/original_surface_alias_probe/original_surface_alias_probe_summary.json`
