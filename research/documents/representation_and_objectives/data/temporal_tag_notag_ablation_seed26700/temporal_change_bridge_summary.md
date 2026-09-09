# earlier analysis fixed-meaning temporal-change bridge

The experiment uses source-attested ATP ranking snapshots. Focal worlds can change higher-ranked participant between the first and later snapshots; secondary worlds are stable. Sparse training labels query only focal before/after state except in the oracle arm. Evaluation asks for focal update and secondary preservation under train/held wording.

## Construction

Base rows: 5120 with labels {'0': 2560, '1': 2560}

- balanced_temporal: rows=512 queries={'focal_after': 256, 'focal_before': 256} labels={'0': 256, '1': 256} changed={'False': 256, 'True': 256}

## Arm: balanced_temporal (train_acc 0.999, 1 seeds)

| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| arbitrary_tag_heldChanged_heldStable_heldHyp | 0.975 | 0.944 | 0.956 | 1.000 | 1.000 | 9.90 | 20.20 |
| arbitrary_tag_heldChanged_heldStable_heldHyp_swappedFocalTags | 0.531 | 0.056 | 0.069 | 1.000 | 1.000 | -9.25 | 19.92 |
| arbitrary_tag_heldChanged_heldStable_noTagHyp | 0.750 | 0.450 | 0.550 | 1.000 | 1.000 | 0.30 | 16.75 |
| arbitrary_tag_heldChanged_heldStable_trainHyp | 0.995 | 0.988 | 0.994 | 1.000 | 1.000 | 20.45 | 23.17 |
| arbitrary_tag_heldStable_heldStable_heldHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 20.20 | 19.76 |
| arbitrary_tag_heldStable_heldStable_noTagHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 16.94 | 16.79 |
| arbitrary_tag_heldStable_heldStable_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 23.19 | 23.20 |
| arbitrary_tag_trainChanged_heldStable_heldHyp | 0.963 | 0.969 | 0.881 | 1.000 | 1.000 | 8.55 | 20.13 |
| arbitrary_tag_trainChanged_heldStable_trainHyp | 0.997 | 1.000 | 0.988 | 1.000 | 1.000 | 20.35 | 23.14 |

