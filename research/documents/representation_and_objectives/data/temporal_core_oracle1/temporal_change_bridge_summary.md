# earlier analysis fixed-meaning temporal-change bridge

The experiment uses source-attested ATP ranking snapshots. Focal worlds can change higher-ranked participant between the first and later snapshots; secondary worlds are stable. Sparse training labels query only focal before/after state except in the oracle arm. Evaluation asks for focal update and secondary preservation under train/held wording.

## Construction

Base rows: 5120 with labels {'0': 2560, '1': 2560}

- stable_only: rows=512 queries={'focal_after': 128, 'focal_before': 128, 'neutral_mention': 256} labels={'0': 256, '1': 256} changed={'False': 256, 'True': 256}
- balanced_temporal: rows=512 queries={'focal_after': 256, 'focal_before': 256} labels={'0': 256, '1': 256} changed={'False': 256, 'True': 256}
- oracle_secondary: rows=1024 queries={'focal_after': 256, 'focal_before': 256, 'secondary_after': 256, 'secondary_before': 256} labels={'0': 512, '1': 512} changed={'False': 512, 'True': 512}

## Arm: balanced_temporal (train_acc 0.984, 1 seeds)

| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldChanged_heldStable_heldCtx_heldHyp | 0.592 | 0.556 | 0.438 | 0.706 | 0.669 | -0.28 | 3.52 |
| heldChanged_heldStable_heldCtx_trainHyp | 0.602 | 0.444 | 0.456 | 0.756 | 0.750 | -0.54 | 5.22 |
| heldChanged_heldStable_laterFirst_trainHyp | 0.798 | 0.925 | 0.281 | 1.000 | 0.988 | -1.27 | 9.24 |
| heldChanged_heldStable_secondaryFirst_trainHyp | 0.800 | 0.963 | 0.237 | 1.000 | 1.000 | -2.05 | 8.52 |
| heldChanged_heldStable_trainCtx_heldHyp | 0.750 | 0.787 | 0.219 | 1.000 | 0.994 | -2.62 | 4.40 |
| heldChanged_heldStable_trainCtx_trainHyp | 0.784 | 0.938 | 0.200 | 1.000 | 1.000 | -2.16 | 8.49 |
| heldStable_heldStable_heldCtx_trainHyp | 0.755 | 0.762 | 0.756 | 0.750 | 0.750 | 3.70 | 2.71 |
| heldStable_heldStable_trainCtx_heldHyp | 0.997 | 1.000 | 1.000 | 0.994 | 0.994 | 14.15 | 12.42 |
| heldStable_heldStable_trainCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 17.44 | 17.53 |
| trainChanged_heldStable_trainCtx_trainHyp | 0.784 | 1.000 | 0.138 | 1.000 | 1.000 | -2.59 | 8.41 |

## Arm: oracle_secondary (train_acc 0.979, 1 seeds)

| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldChanged_heldStable_heldCtx_heldHyp | 0.622 | 0.537 | 0.456 | 0.750 | 0.744 | -0.92 | 3.66 |
| heldChanged_heldStable_heldCtx_trainHyp | 0.616 | 0.537 | 0.475 | 0.725 | 0.725 | -0.19 | 3.94 |
| heldChanged_heldStable_laterFirst_trainHyp | 0.750 | 0.619 | 0.381 | 1.000 | 1.000 | 0.00 | 15.51 |
| heldChanged_heldStable_secondaryFirst_trainHyp | 0.752 | 0.394 | 0.613 | 1.000 | 1.000 | -0.05 | 9.91 |
| heldChanged_heldStable_trainCtx_heldHyp | 0.756 | 0.487 | 0.537 | 1.000 | 1.000 | 0.03 | 11.82 |
| heldChanged_heldStable_trainCtx_trainHyp | 0.750 | 0.375 | 0.625 | 1.000 | 1.000 | 0.03 | 15.73 |
| heldStable_heldStable_heldCtx_trainHyp | 0.728 | 0.750 | 0.750 | 0.706 | 0.706 | 1.24 | 1.47 |
| heldStable_heldStable_trainCtx_heldHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 13.92 | 14.48 |
| heldStable_heldStable_trainCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 16.03 | 17.43 |
| trainChanged_heldStable_trainCtx_trainHyp | 0.750 | 0.394 | 0.606 | 1.000 | 1.000 | 0.01 | 15.95 |

## Arm: stable_only (train_acc 1.000, 1 seeds)

| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldChanged_heldStable_heldCtx_heldHyp | 0.642 | 0.544 | 0.400 | 0.825 | 0.800 | 0.73 | 7.36 |
| heldChanged_heldStable_heldCtx_trainHyp | 0.561 | 0.544 | 0.456 | 0.625 | 0.619 | -1.31 | 9.39 |
| heldChanged_heldStable_laterFirst_trainHyp | 0.750 | 0.669 | 0.331 | 1.000 | 1.000 | -4.60 | 21.30 |
| heldChanged_heldStable_secondaryFirst_trainHyp | 0.748 | 0.562 | 0.431 | 1.000 | 1.000 | 1.95 | 21.95 |
| heldChanged_heldStable_trainCtx_heldHyp | 0.725 | 0.375 | 0.606 | 0.956 | 0.963 | 0.32 | 3.59 |
| heldChanged_heldStable_trainCtx_trainHyp | 0.750 | 0.319 | 0.681 | 1.000 | 1.000 | 5.32 | 21.33 |
| heldStable_heldStable_heldCtx_trainHyp | 0.619 | 0.606 | 0.613 | 0.631 | 0.625 | 3.31 | 2.32 |
| heldStable_heldStable_trainCtx_heldHyp | 0.998 | 1.000 | 0.994 | 1.000 | 1.000 | 5.21 | 3.69 |
| heldStable_heldStable_trainCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 26.79 | 26.81 |
| trainChanged_heldStable_trainCtx_trainHyp | 0.750 | 0.431 | 0.569 | 1.000 | 1.000 | 2.14 | 21.43 |

