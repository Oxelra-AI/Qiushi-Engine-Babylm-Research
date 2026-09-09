# earlier analysis fixed-meaning temporal-change bridge

The experiment uses source-attested ATP ranking snapshots. Focal worlds can change higher-ranked participant between the first and later snapshots; secondary worlds are stable. Sparse training labels query only focal before/after state except in the oracle arm. Evaluation asks for focal update and secondary preservation under train/held wording.

## Construction

Base rows: 5120 with labels {'0': 2560, '1': 2560}

- balanced_temporal: rows=512 queries={'focal_after': 256, 'focal_before': 256} labels={'0': 256, '1': 256} changed={'False': 256, 'True': 256}

## Arm: balanced_temporal (train_acc 0.979, 1 seeds)

| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| arbitrary_tag_heldChanged_heldStable_heldHyp | 0.738 | 0.500 | 0.450 | 1.000 | 1.000 | -0.14 | 9.09 |
| arbitrary_tag_heldChanged_heldStable_heldHyp_swappedFocalTags | 0.742 | 0.431 | 0.537 | 1.000 | 1.000 | 0.08 | 9.68 |
| arbitrary_tag_heldChanged_heldStable_noTagHyp | 0.750 | 0.475 | 0.525 | 1.000 | 1.000 | 0.14 | 8.91 |
| arbitrary_tag_heldChanged_heldStable_trainHyp | 0.828 | 0.650 | 0.662 | 1.000 | 1.000 | 0.63 | 11.62 |
| arbitrary_tag_heldStable_heldStable_heldHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 11.94 | 12.05 |
| arbitrary_tag_heldStable_heldStable_noTagHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 11.59 | 11.56 |
| arbitrary_tag_heldStable_heldStable_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 12.69 | 12.52 |
| arbitrary_tag_trainChanged_heldStable_heldHyp | 0.761 | 0.506 | 0.537 | 1.000 | 1.000 | 0.51 | 8.84 |
| arbitrary_tag_trainChanged_heldStable_trainHyp | 0.838 | 0.694 | 0.656 | 1.000 | 1.000 | 0.68 | 11.64 |

