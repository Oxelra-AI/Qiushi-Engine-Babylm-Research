# strict ordered target surface strict ordered target surface

Rows: `experiments/archive/initial_model_studies/data/strict_ordered_target_surface_rows.csv`
Summary: `experiments/archive/initial_model_studies/data/strict_ordered_target_surface_summary.json`

Selection: seed42@80M; heldout: seed43@100M; threshold 0.05.

| surface | selected | selected fraction | heldout survivors | survivor fraction of all | survivor fraction of selected |
|---|---:|---:|---:|---:|---:|
| random same + block | 157 | 0.1308 | 64 | 0.0533 | 0.4076 |
| local-neighbor same + block | 153 | 0.1275 | 68 | 0.0567 | 0.4444 |

The local-neighbor surface is the stricter one: it asks whether full prefix beats a nearby contiguous same-source chunk and block-shuffled true prefix, with independent heldout measurement. Representative heldout local survivors are stored in the JSON.
