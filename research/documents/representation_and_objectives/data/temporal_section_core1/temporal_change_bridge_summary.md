# earlier analysis fixed-meaning temporal-change bridge

The experiment uses source-attested ATP ranking snapshots. Focal worlds can change higher-ranked participant between the first and later snapshots; secondary worlds are stable. Sparse training labels query only focal before/after state except in the oracle arm. Evaluation asks for focal update and secondary preservation under train/held wording.

## Construction

Base rows: 5120 with labels {'0': 2560, '1': 2560}

- balanced_temporal: rows=512 queries={'focal_after': 256, 'focal_before': 256} labels={'0': 256, '1': 256} changed={'False': 256, 'True': 256}
- oracle_secondary: rows=1024 queries={'focal_after': 256, 'focal_before': 256, 'secondary_after': 256, 'secondary_before': 256} labels={'0': 512, '1': 512} changed={'False': 512, 'True': 512}

## Arm: balanced_temporal (train_acc 1.000, 1 seeds)

| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldChanged_heldStable_heldCtx_heldHyp | 0.728 | 0.613 | 0.525 | 0.863 | 0.912 | 1.01 | 7.29 |
| heldChanged_heldStable_heldCtx_trainHyp | 0.775 | 0.550 | 0.769 | 0.894 | 0.887 | 8.37 | 15.31 |
| heldChanged_heldStable_laterFirst_trainHyp | 0.983 | 1.000 | 0.931 | 1.000 | 1.000 | 10.14 | 29.83 |
| heldChanged_heldStable_secondaryFirst_trainHyp | 0.991 | 1.000 | 0.963 | 1.000 | 1.000 | 12.87 | 30.56 |
| heldChanged_heldStable_trainCtx_heldHyp | 0.934 | 0.950 | 0.787 | 1.000 | 1.000 | 6.67 | 19.18 |
| heldChanged_heldStable_trainCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 23.10 | 30.10 |
| heldStable_heldStable_heldCtx_trainHyp | 0.945 | 0.881 | 0.988 | 0.919 | 0.994 | 14.95 | 18.06 |
| heldStable_heldStable_trainCtx_heldHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 22.00 | 20.05 |
| heldStable_heldStable_trainCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 30.84 | 30.71 |
| trainChanged_heldStable_trainCtx_trainHyp | 0.998 | 1.000 | 0.994 | 1.000 | 1.000 | 20.48 | 30.43 |

## Arm: oracle_secondary (train_acc 1.000, 1 seeds)

| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldChanged_heldStable_heldCtx_heldHyp | 0.695 | 0.625 | 0.700 | 0.819 | 0.637 | 5.09 | 1.37 |
| heldChanged_heldStable_heldCtx_trainHyp | 0.911 | 0.925 | 0.762 | 0.969 | 0.988 | 2.16 | 4.73 |
| heldChanged_heldStable_laterFirst_trainHyp | 0.977 | 0.906 | 1.000 | 1.000 | 1.000 | 20.02 | 36.24 |
| heldChanged_heldStable_secondaryFirst_trainHyp | 0.833 | 0.331 | 1.000 | 1.000 | 1.000 | 14.36 | 36.22 |
| heldChanged_heldStable_trainCtx_heldHyp | 0.992 | 0.969 | 1.000 | 1.000 | 1.000 | 15.83 | 33.52 |
| heldChanged_heldStable_trainCtx_trainHyp | 0.998 | 1.000 | 0.994 | 1.000 | 1.000 | 15.52 | 33.18 |
| heldStable_heldStable_heldCtx_trainHyp | 0.959 | 1.000 | 0.912 | 0.975 | 0.950 | 7.48 | 5.52 |
| heldStable_heldStable_trainCtx_heldHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 35.41 | 33.44 |
| heldStable_heldStable_trainCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 34.81 | 32.78 |
| trainChanged_heldStable_trainCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 15.90 | 32.86 |

