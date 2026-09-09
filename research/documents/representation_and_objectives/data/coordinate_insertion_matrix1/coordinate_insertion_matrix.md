# role coordinate collision and route coordinate insertion matrix

Train condition: unique entry tags only. Evaluation inserts extra context material while keeping direct tag queries.

Best train acc: 1.000

Train fit by kind: `{"base_stable_anchor|ctx=unique|q=direct_tag": 0.999609375, "base_stable_train|ctx=unique|q=direct_tag": 1.0, "sparse_changed_focal|ctx=unique|q=direct_tag": 1.0, "sparse_stable_focal|ctx=unique|q=direct_tag": 1.0}`

| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| insert_entrydup_heldChanged_heldStable_directTag_heldHyp | 0.688 | 0.750 | 0.000 | 1.000 | 1.000 | -6.68 | 24.92 |
| insert_filler_heldChanged_heldStable_directTag_heldHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 25.42 | 25.69 |
| insert_labeldup_heldChanged_heldStable_directTag_heldHyp | 0.750 | 0.500 | 0.500 | 1.000 | 1.000 | 4.90 | 25.29 |
| insert_roledecl_heldChanged_heldStable_directTag_heldHyp | 0.688 | 0.250 | 0.500 | 1.000 | 1.000 | 1.64 | 23.85 |
| insert_roledecl_heldChanged_heldStable_directTag_roleSwap_heldHyp | 0.688 | 0.250 | 0.500 | 1.000 | 1.000 | -0.69 | 25.41 |
| insert_slotdecl_heldChanged_heldStable_directTag_heldHyp | 0.750 | 1.000 | 0.000 | 1.000 | 1.000 | -10.92 | 26.42 |
| insert_unique_heldChanged_heldStable_directTag_heldHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 24.55 | 25.34 |
