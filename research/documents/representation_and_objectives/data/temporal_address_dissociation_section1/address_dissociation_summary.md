# temporal addressability bridge temporal address dissociation

This run uses the earlier analysis fixed-meaning ATP temporal-change bridge with one seed and the balanced-temporal non-oracle arm. It tests whether section success survives randomized record order, held query paraphrases without exact Background/Update repetition, and arbitrary per-example tag addresses.

## Mode: section_rolepara

Best train accuracy: 1.000

Train fit by kind: `{"base_stable_anchor": 1.0, "base_stable_train": 1.0, "sparse_changed_focal": 1.0, "sparse_stable_focal": 1.0}`

| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| section_rolepara_heldChanged_heldStable_heldHyp | 0.741 | 0.506 | 0.456 | 1.000 | 1.000 | -0.93 | 21.47 |
| section_rolepara_heldChanged_heldStable_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 17.61 | 21.59 |
| section_rolepara_heldStable_heldStable_heldHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 13.80 | 21.95 |
| section_rolepara_heldStable_heldStable_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 23.19 | 24.25 |
| section_rolepara_trainChanged_heldStable_heldHyp | 0.733 | 0.481 | 0.450 | 1.000 | 1.000 | -0.32 | 21.74 |
| section_rolepara_trainChanged_heldStable_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 18.20 | 21.15 |

