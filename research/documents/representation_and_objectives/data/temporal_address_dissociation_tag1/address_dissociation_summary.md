# temporal addressability bridge temporal address dissociation

This run uses the earlier analysis fixed-meaning ATP temporal-change bridge with one seed and the balanced-temporal non-oracle arm. It tests whether section success survives randomized record order, held query paraphrases without exact Background/Update repetition, and arbitrary per-example tag addresses.

## Mode: arbitrary_tag

Best train accuracy: 0.999

Train fit by kind: `{"base_stable_anchor": 1.0, "base_stable_train": 1.0, "sparse_changed_focal": 0.9765625, "sparse_stable_focal": 1.0}`

| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| arbitrary_tag_heldChanged_heldStable_heldHyp | 0.975 | 0.944 | 0.956 | 1.000 | 1.000 | 9.90 | 20.20 |
| arbitrary_tag_heldChanged_heldStable_heldHyp_swappedFocalTags | 0.531 | 0.056 | 0.069 | 1.000 | 1.000 | -9.25 | 19.92 |
| arbitrary_tag_heldChanged_heldStable_trainHyp | 0.995 | 0.988 | 0.994 | 1.000 | 1.000 | 20.45 | 23.17 |
| arbitrary_tag_heldStable_heldStable_heldHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 20.20 | 19.76 |
| arbitrary_tag_heldStable_heldStable_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 23.19 | 23.20 |
| arbitrary_tag_trainChanged_heldStable_heldHyp | 0.963 | 0.969 | 0.881 | 1.000 | 1.000 | 8.55 | 20.13 |
| arbitrary_tag_trainChanged_heldStable_trainHyp | 0.997 | 1.000 | 0.988 | 1.000 | 1.000 | 20.35 | 23.14 |

