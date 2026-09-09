# earlier analysis fixed-meaning temporal-change bridge

The experiment uses source-attested ATP ranking snapshots. Focal worlds can change higher-ranked participant between the first and later snapshots; secondary worlds are stable. Sparse training labels query only focal before/after state except in the oracle arm. Evaluation asks for focal update and secondary preservation under train/held wording.

## Construction

Base rows: 1920 with labels {'0': 960, '1': 960}

- exposure: rows=384 queries={'neutral_mention': 384} labels={'0': 192, '1': 192} changed={'False': 192, 'True': 192}
- stable_only: rows=384 queries={'focal_after': 96, 'focal_before': 96, 'neutral_mention': 192} labels={'0': 192, '1': 192} changed={'False': 192, 'True': 192}
- changed_only: rows=384 queries={'focal_after': 96, 'focal_before': 96, 'neutral_mention': 192} labels={'0': 192, '1': 192} changed={'False': 192, 'True': 192}
- balanced_temporal: rows=384 queries={'focal_after': 192, 'focal_before': 192} labels={'0': 192, '1': 192} changed={'False': 192, 'True': 192}

## Arm: balanced_temporal (train_acc 0.901, 1 seeds)

| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldChanged_heldStable_dirCtx_heldHyp | 0.558 | 0.658 | 0.467 | 0.467 | 0.642 | -0.22 | 0.18 |
| heldChanged_heldStable_dirCtx_trainHyp | 0.642 | 0.750 | 0.250 | 0.783 | 0.783 | -1.54 | 1.67 |
| heldChanged_heldStable_heldCtx_heldHyp | 0.627 | 0.567 | 0.542 | 0.767 | 0.633 | 0.05 | 0.05 |
| heldChanged_heldStable_heldCtx_trainHyp | 0.606 | 0.508 | 0.492 | 0.717 | 0.708 | -0.01 | 1.03 |
| heldChanged_heldStable_laterFirst_heldHyp | 0.733 | 0.592 | 0.392 | 0.967 | 0.983 | -0.13 | 1.25 |
| heldChanged_heldStable_laterFirst_trainHyp | 0.746 | 0.533 | 0.458 | 0.992 | 1.000 | -0.23 | 2.07 |
| heldChanged_heldStable_nonDirCtx_trainHyp | 0.292 | 0.508 | 0.492 | 0.075 | 0.092 | -0.52 | -2.55 |
| heldChanged_heldStable_secondaryFirst_heldHyp | 0.756 | 0.508 | 0.550 | 0.975 | 0.992 | -0.02 | 1.45 |
| heldChanged_heldStable_secondaryFirst_trainHyp | 0.752 | 0.475 | 0.533 | 1.000 | 1.000 | -0.07 | 2.00 |
| heldChanged_heldStable_trainCtx_heldHyp | 0.748 | 0.467 | 0.525 | 1.000 | 1.000 | 0.08 | 1.41 |
| heldChanged_heldStable_trainCtx_trainHyp | 0.746 | 0.433 | 0.567 | 0.992 | 0.992 | 0.09 | 2.12 |
| heldChanged_trainStable_dirCtx_heldHyp | 0.550 | 0.733 | 0.317 | 0.500 | 0.650 | -0.36 | 0.23 |
| heldChanged_trainStable_dirCtx_trainHyp | 0.627 | 0.617 | 0.383 | 0.750 | 0.758 | -0.82 | 2.14 |
| heldChanged_trainStable_heldCtx_heldHyp | 0.606 | 0.508 | 0.550 | 0.783 | 0.583 | 0.18 | 0.11 |
| heldChanged_trainStable_heldCtx_trainHyp | 0.619 | 0.492 | 0.500 | 0.750 | 0.733 | -0.00 | 1.12 |
| heldChanged_trainStable_laterFirst_heldHyp | 0.738 | 0.600 | 0.367 | 0.992 | 0.992 | -0.17 | 1.20 |
| heldChanged_trainStable_laterFirst_trainHyp | 0.742 | 0.492 | 0.492 | 0.992 | 0.992 | -0.17 | 2.12 |
| heldChanged_trainStable_nonDirCtx_trainHyp | 0.275 | 0.558 | 0.433 | 0.058 | 0.050 | -0.45 | -3.09 |
| heldChanged_trainStable_secondaryFirst_heldHyp | 0.760 | 0.550 | 0.492 | 1.000 | 1.000 | -0.04 | 1.20 |
| heldChanged_trainStable_secondaryFirst_trainHyp | 0.752 | 0.500 | 0.508 | 1.000 | 1.000 | -0.05 | 2.15 |
| heldChanged_trainStable_trainCtx_heldHyp | 0.735 | 0.400 | 0.558 | 0.992 | 0.992 | 0.22 | 1.26 |
| heldChanged_trainStable_trainCtx_trainHyp | 0.746 | 0.525 | 0.475 | 0.992 | 0.992 | 0.09 | 2.18 |
| heldStable_heldStable_dirCtx_heldHyp | 0.631 | 0.550 | 0.650 | 0.567 | 0.758 | 0.31 | 0.46 |
| heldStable_heldStable_dirCtx_trainHyp | 0.694 | 0.583 | 0.583 | 0.808 | 0.800 | 0.76 | 2.35 |
| heldStable_heldStable_heldCtx_heldHyp | 0.692 | 0.733 | 0.758 | 0.708 | 0.567 | 0.51 | -0.04 |
| heldStable_heldStable_heldCtx_trainHyp | 0.779 | 0.783 | 0.800 | 0.767 | 0.767 | 1.18 | 1.18 |
| heldStable_heldStable_laterFirst_heldHyp | 0.979 | 0.992 | 0.983 | 0.983 | 0.958 | 1.37 | 1.18 |
| heldStable_heldStable_laterFirst_trainHyp | 0.996 | 1.000 | 1.000 | 0.992 | 0.992 | 2.00 | 2.07 |
| heldStable_heldStable_nonDirCtx_trainHyp | 0.079 | 0.083 | 0.092 | 0.083 | 0.058 | -2.72 | -3.11 |
| heldStable_heldStable_secondaryFirst_heldHyp | 0.992 | 0.992 | 0.992 | 0.983 | 1.000 | 1.13 | 1.30 |
| heldStable_heldStable_secondaryFirst_trainHyp | 0.998 | 1.000 | 1.000 | 0.992 | 1.000 | 2.11 | 2.15 |
| heldStable_heldStable_trainCtx_heldHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.40 | 1.31 |
| heldStable_heldStable_trainCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.64 | 1.97 |
| trainChanged_heldStable_dirCtx_heldHyp | 0.529 | 0.725 | 0.325 | 0.458 | 0.608 | -0.43 | 0.17 |
| trainChanged_heldStable_dirCtx_trainHyp | 0.646 | 0.558 | 0.450 | 0.783 | 0.792 | -0.11 | 2.45 |
| trainChanged_heldStable_heldCtx_heldHyp | 0.600 | 0.517 | 0.508 | 0.758 | 0.617 | -0.10 | 0.03 |
| trainChanged_heldStable_heldCtx_trainHyp | 0.594 | 0.575 | 0.417 | 0.692 | 0.692 | -0.07 | 0.84 |
| trainChanged_heldStable_laterFirst_heldHyp | 0.765 | 0.550 | 0.517 | 0.992 | 1.000 | 0.01 | 1.34 |
| trainChanged_heldStable_laterFirst_trainHyp | 0.750 | 0.433 | 0.567 | 1.000 | 1.000 | 0.08 | 2.03 |
| trainChanged_heldStable_nonDirCtx_trainHyp | 0.269 | 0.633 | 0.342 | 0.050 | 0.050 | -1.14 | -3.52 |
| trainChanged_heldStable_secondaryFirst_heldHyp | 0.754 | 0.608 | 0.425 | 0.992 | 0.992 | -0.33 | 1.35 |
| trainChanged_heldStable_secondaryFirst_trainHyp | 0.740 | 0.467 | 0.525 | 0.983 | 0.983 | -0.03 | 1.99 |
| trainChanged_heldStable_trainCtx_heldHyp | 0.733 | 0.550 | 0.425 | 0.983 | 0.975 | -0.11 | 1.33 |
| trainChanged_heldStable_trainCtx_trainHyp | 0.750 | 0.475 | 0.525 | 1.000 | 1.000 | 0.15 | 2.04 |

## Arm: changed_only (train_acc 0.839, 1 seeds)

| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldChanged_heldStable_dirCtx_heldHyp | 0.596 | 0.600 | 0.408 | 0.683 | 0.692 | -0.14 | 0.35 |
| heldChanged_heldStable_dirCtx_trainHyp | 0.523 | 0.542 | 0.475 | 0.542 | 0.533 | -0.66 | 0.84 |
| heldChanged_heldStable_heldCtx_heldHyp | 0.400 | 0.517 | 0.492 | 0.292 | 0.300 | 0.02 | -0.10 |
| heldChanged_heldStable_heldCtx_trainHyp | 0.492 | 0.525 | 0.458 | 0.492 | 0.492 | 0.10 | 0.01 |
| heldChanged_heldStable_laterFirst_heldHyp | 0.402 | 0.567 | 0.417 | 0.300 | 0.325 | -0.05 | 0.03 |
| heldChanged_heldStable_laterFirst_trainHyp | 0.515 | 0.458 | 0.558 | 0.533 | 0.508 | 0.09 | 0.15 |
| heldChanged_heldStable_nonDirCtx_trainHyp | 0.446 | 0.475 | 0.492 | 0.392 | 0.425 | 0.20 | -0.13 |
| heldChanged_heldStable_secondaryFirst_heldHyp | 0.383 | 0.450 | 0.533 | 0.283 | 0.267 | 0.02 | -0.01 |
| heldChanged_heldStable_secondaryFirst_trainHyp | 0.544 | 0.517 | 0.475 | 0.592 | 0.592 | -0.04 | 0.07 |
| heldChanged_heldStable_trainCtx_heldHyp | 0.348 | 0.425 | 0.492 | 0.217 | 0.258 | 0.05 | -0.04 |
| heldChanged_heldStable_trainCtx_trainHyp | 0.544 | 0.542 | 0.442 | 0.600 | 0.592 | -0.10 | 0.16 |
| heldChanged_trainStable_dirCtx_heldHyp | 0.552 | 0.408 | 0.642 | 0.575 | 0.583 | 0.21 | 0.21 |
| heldChanged_trainStable_dirCtx_trainHyp | 0.579 | 0.450 | 0.567 | 0.650 | 0.650 | 0.40 | 1.14 |
| heldChanged_trainStable_heldCtx_heldHyp | 0.408 | 0.525 | 0.492 | 0.342 | 0.275 | 0.02 | -0.08 |
| heldChanged_trainStable_heldCtx_trainHyp | 0.540 | 0.533 | 0.483 | 0.567 | 0.575 | -0.02 | 0.09 |
| heldChanged_trainStable_laterFirst_heldHyp | 0.377 | 0.475 | 0.517 | 0.242 | 0.275 | -0.03 | -0.07 |
| heldChanged_trainStable_laterFirst_trainHyp | 0.517 | 0.475 | 0.533 | 0.525 | 0.533 | -0.01 | 0.27 |
| heldChanged_trainStable_nonDirCtx_trainHyp | 0.512 | 0.467 | 0.542 | 0.533 | 0.508 | -0.14 | -0.28 |
| heldChanged_trainStable_secondaryFirst_heldHyp | 0.371 | 0.583 | 0.450 | 0.208 | 0.242 | -0.00 | -0.07 |
| heldChanged_trainStable_secondaryFirst_trainHyp | 0.519 | 0.458 | 0.525 | 0.550 | 0.542 | -0.08 | 0.20 |
| heldChanged_trainStable_trainCtx_heldHyp | 0.367 | 0.442 | 0.483 | 0.267 | 0.275 | -0.00 | -0.04 |
| heldChanged_trainStable_trainCtx_trainHyp | 0.481 | 0.517 | 0.500 | 0.458 | 0.450 | 0.02 | 0.11 |
| heldStable_heldStable_dirCtx_heldHyp | 0.631 | 0.600 | 0.600 | 0.667 | 0.658 | 0.20 | 0.36 |
| heldStable_heldStable_dirCtx_trainHyp | 0.494 | 0.433 | 0.433 | 0.550 | 0.558 | 0.50 | 0.88 |
| heldStable_heldStable_heldCtx_heldHyp | 0.417 | 0.425 | 0.450 | 0.383 | 0.408 | 0.02 | 0.03 |
| heldStable_heldStable_heldCtx_trainHyp | 0.515 | 0.508 | 0.508 | 0.525 | 0.517 | 0.15 | 0.09 |
| heldStable_heldStable_laterFirst_heldHyp | 0.356 | 0.317 | 0.375 | 0.367 | 0.367 | -0.04 | -0.00 |
| heldStable_heldStable_laterFirst_trainHyp | 0.602 | 0.667 | 0.675 | 0.525 | 0.542 | 0.22 | 0.16 |
| heldStable_heldStable_nonDirCtx_trainHyp | 0.546 | 0.567 | 0.567 | 0.525 | 0.525 | -0.10 | -0.17 |
| heldStable_heldStable_secondaryFirst_heldHyp | 0.283 | 0.292 | 0.342 | 0.267 | 0.233 | -0.04 | -0.05 |
| heldStable_heldStable_secondaryFirst_trainHyp | 0.537 | 0.500 | 0.475 | 0.583 | 0.592 | 0.18 | 0.35 |
| heldStable_heldStable_trainCtx_heldHyp | 0.352 | 0.317 | 0.325 | 0.367 | 0.400 | -0.01 | 0.06 |
| heldStable_heldStable_trainCtx_trainHyp | 0.554 | 0.583 | 0.592 | 0.517 | 0.525 | 0.15 | 0.24 |
| trainChanged_heldStable_dirCtx_heldHyp | 0.560 | 0.525 | 0.450 | 0.650 | 0.617 | 0.04 | 0.19 |
| trainChanged_heldStable_dirCtx_trainHyp | 0.525 | 0.508 | 0.500 | 0.550 | 0.542 | 0.01 | 0.96 |
| trainChanged_heldStable_heldCtx_heldHyp | 0.433 | 0.625 | 0.392 | 0.350 | 0.367 | -0.06 | -0.06 |
| trainChanged_heldStable_heldCtx_trainHyp | 0.550 | 0.475 | 0.525 | 0.583 | 0.617 | 0.17 | -0.08 |
| trainChanged_heldStable_laterFirst_heldHyp | 0.433 | 0.508 | 0.525 | 0.317 | 0.383 | -0.02 | 0.05 |
| trainChanged_heldStable_laterFirst_trainHyp | 0.492 | 0.458 | 0.525 | 0.492 | 0.492 | 0.01 | 0.08 |
| trainChanged_heldStable_nonDirCtx_trainHyp | 0.527 | 0.508 | 0.492 | 0.550 | 0.558 | 0.09 | -0.06 |
| trainChanged_heldStable_secondaryFirst_heldHyp | 0.425 | 0.508 | 0.467 | 0.350 | 0.375 | -0.01 | -0.01 |
| trainChanged_heldStable_secondaryFirst_trainHyp | 0.481 | 0.458 | 0.550 | 0.467 | 0.450 | -0.02 | -0.00 |
| trainChanged_heldStable_trainCtx_heldHyp | 0.365 | 0.475 | 0.492 | 0.225 | 0.267 | 0.02 | -0.03 |
| trainChanged_heldStable_trainCtx_trainHyp | 0.496 | 0.492 | 0.517 | 0.483 | 0.492 | -0.11 | -0.03 |

## Arm: exposure (train_acc 0.535, 1 seeds)

| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldChanged_heldStable_dirCtx_heldHyp | 0.481 | 0.633 | 0.367 | 0.442 | 0.483 | -0.01 | -0.03 |
| heldChanged_heldStable_dirCtx_trainHyp | 0.490 | 0.400 | 0.583 | 0.492 | 0.483 | 0.00 | 0.00 |
| heldChanged_heldStable_heldCtx_heldHyp | 0.540 | 0.458 | 0.483 | 0.583 | 0.633 | -0.01 | 0.03 |
| heldChanged_heldStable_heldCtx_trainHyp | 0.533 | 0.442 | 0.533 | 0.583 | 0.575 | -0.01 | 0.00 |
| heldChanged_heldStable_laterFirst_heldHyp | 0.546 | 0.433 | 0.533 | 0.625 | 0.592 | 0.13 | 0.16 |
| heldChanged_heldStable_laterFirst_trainHyp | 0.525 | 0.508 | 0.492 | 0.550 | 0.550 | -0.04 | 0.02 |
| heldChanged_heldStable_nonDirCtx_trainHyp | 0.498 | 0.475 | 0.525 | 0.492 | 0.500 | 0.03 | -0.02 |
| heldChanged_heldStable_secondaryFirst_heldHyp | 0.569 | 0.533 | 0.592 | 0.550 | 0.600 | 0.07 | -0.01 |
| heldChanged_heldStable_secondaryFirst_trainHyp | 0.475 | 0.358 | 0.633 | 0.450 | 0.458 | 0.07 | -0.01 |
| heldChanged_heldStable_trainCtx_heldHyp | 0.485 | 0.492 | 0.500 | 0.458 | 0.492 | -0.07 | -0.06 |
| heldChanged_heldStable_trainCtx_trainHyp | 0.442 | 0.525 | 0.467 | 0.392 | 0.383 | 0.02 | -0.06 |
| heldChanged_trainStable_dirCtx_heldHyp | 0.542 | 0.500 | 0.592 | 0.533 | 0.542 | -0.01 | 0.04 |
| heldChanged_trainStable_dirCtx_trainHyp | 0.442 | 0.333 | 0.667 | 0.383 | 0.383 | 0.02 | -0.00 |
| heldChanged_trainStable_heldCtx_heldHyp | 0.442 | 0.425 | 0.667 | 0.325 | 0.350 | 0.04 | -0.04 |
| heldChanged_trainStable_heldCtx_trainHyp | 0.458 | 0.550 | 0.458 | 0.417 | 0.408 | -0.00 | -0.00 |
| heldChanged_trainStable_laterFirst_heldHyp | 0.504 | 0.550 | 0.425 | 0.475 | 0.567 | -0.02 | -0.00 |
| heldChanged_trainStable_laterFirst_trainHyp | 0.485 | 0.592 | 0.417 | 0.467 | 0.467 | -0.05 | -0.10 |
| heldChanged_trainStable_nonDirCtx_trainHyp | 0.469 | 0.483 | 0.508 | 0.442 | 0.442 | -0.02 | -0.01 |
| heldChanged_trainStable_secondaryFirst_heldHyp | 0.471 | 0.508 | 0.508 | 0.442 | 0.425 | 0.13 | -0.08 |
| heldChanged_trainStable_secondaryFirst_trainHyp | 0.483 | 0.433 | 0.575 | 0.458 | 0.467 | -0.02 | -0.11 |
| heldChanged_trainStable_trainCtx_heldHyp | 0.490 | 0.458 | 0.458 | 0.500 | 0.542 | -0.01 | 0.01 |
| heldChanged_trainStable_trainCtx_trainHyp | 0.494 | 0.408 | 0.592 | 0.492 | 0.483 | 0.05 | -0.00 |
| heldStable_heldStable_dirCtx_heldHyp | 0.406 | 0.433 | 0.442 | 0.367 | 0.383 | 0.04 | -0.03 |
| heldStable_heldStable_dirCtx_trainHyp | 0.544 | 0.625 | 0.608 | 0.475 | 0.467 | -0.00 | 0.01 |
| heldStable_heldStable_heldCtx_heldHyp | 0.481 | 0.400 | 0.433 | 0.508 | 0.583 | -0.04 | -0.05 |
| heldStable_heldStable_heldCtx_trainHyp | 0.556 | 0.525 | 0.533 | 0.575 | 0.592 | -0.00 | 0.00 |
| heldStable_heldStable_laterFirst_heldHyp | 0.456 | 0.525 | 0.475 | 0.408 | 0.417 | -0.06 | -0.31 |
| heldStable_heldStable_laterFirst_trainHyp | 0.521 | 0.575 | 0.575 | 0.467 | 0.467 | 0.02 | 0.02 |
| heldStable_heldStable_nonDirCtx_trainHyp | 0.460 | 0.450 | 0.442 | 0.475 | 0.475 | -0.00 | -0.01 |
| heldStable_heldStable_secondaryFirst_heldHyp | 0.481 | 0.642 | 0.583 | 0.358 | 0.342 | 0.11 | -0.04 |
| heldStable_heldStable_secondaryFirst_trainHyp | 0.565 | 0.583 | 0.592 | 0.542 | 0.542 | 0.08 | -0.03 |
| heldStable_heldStable_trainCtx_heldHyp | 0.540 | 0.592 | 0.667 | 0.442 | 0.458 | -0.01 | -0.06 |
| heldStable_heldStable_trainCtx_trainHyp | 0.502 | 0.567 | 0.550 | 0.433 | 0.458 | -0.01 | -0.08 |
| trainChanged_heldStable_dirCtx_heldHyp | 0.512 | 0.617 | 0.425 | 0.500 | 0.508 | -0.01 | 0.08 |
| trainChanged_heldStable_dirCtx_trainHyp | 0.485 | 0.608 | 0.400 | 0.458 | 0.475 | -0.01 | 0.01 |
| trainChanged_heldStable_heldCtx_heldHyp | 0.402 | 0.542 | 0.417 | 0.267 | 0.383 | 0.02 | -0.05 |
| trainChanged_heldStable_heldCtx_trainHyp | 0.540 | 0.400 | 0.592 | 0.583 | 0.583 | 0.01 | 0.01 |
| trainChanged_heldStable_laterFirst_heldHyp | 0.533 | 0.308 | 0.667 | 0.592 | 0.567 | 0.19 | 0.05 |
| trainChanged_heldStable_laterFirst_trainHyp | 0.521 | 0.467 | 0.533 | 0.542 | 0.542 | 0.09 | -0.01 |
| trainChanged_heldStable_nonDirCtx_trainHyp | 0.473 | 0.533 | 0.458 | 0.450 | 0.450 | 0.00 | 0.00 |
| trainChanged_heldStable_secondaryFirst_heldHyp | 0.404 | 0.442 | 0.483 | 0.325 | 0.367 | -0.08 | -0.05 |
| trainChanged_heldStable_secondaryFirst_trainHyp | 0.540 | 0.450 | 0.550 | 0.575 | 0.583 | 0.01 | 0.05 |
| trainChanged_heldStable_trainCtx_heldHyp | 0.500 | 0.492 | 0.500 | 0.450 | 0.558 | -0.06 | -0.00 |
| trainChanged_heldStable_trainCtx_trainHyp | 0.496 | 0.392 | 0.608 | 0.492 | 0.492 | 0.11 | -0.02 |

## Arm: stable_only (train_acc 0.990, 1 seeds)

| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldChanged_heldStable_dirCtx_heldHyp | 0.579 | 0.525 | 0.408 | 0.692 | 0.692 | -2.06 | 1.28 |
| heldChanged_heldStable_dirCtx_trainHyp | 0.619 | 0.400 | 0.600 | 0.725 | 0.750 | 2.18 | 6.64 |
| heldChanged_heldStable_heldCtx_heldHyp | 0.625 | 0.533 | 0.517 | 0.758 | 0.692 | 0.56 | 1.11 |
| heldChanged_heldStable_heldCtx_trainHyp | 0.598 | 0.533 | 0.467 | 0.692 | 0.700 | -0.83 | 2.94 |
| heldChanged_heldStable_laterFirst_heldHyp | 0.719 | 0.458 | 0.467 | 0.992 | 0.958 | -0.64 | 9.02 |
| heldChanged_heldStable_laterFirst_trainHyp | 0.752 | 0.733 | 0.275 | 1.000 | 1.000 | -5.42 | 17.57 |
| heldChanged_heldStable_nonDirCtx_trainHyp | 0.350 | 0.500 | 0.492 | 0.208 | 0.200 | -0.01 | -8.67 |
| heldChanged_heldStable_secondaryFirst_heldHyp | 0.756 | 0.392 | 0.633 | 1.000 | 1.000 | 2.05 | 10.68 |
| heldChanged_heldStable_secondaryFirst_trainHyp | 0.750 | 0.383 | 0.617 | 1.000 | 1.000 | 1.77 | 18.06 |
| heldChanged_heldStable_trainCtx_heldHyp | 0.725 | 0.508 | 0.450 | 0.975 | 0.967 | 0.04 | 9.33 |
| heldChanged_heldStable_trainCtx_trainHyp | 0.746 | 0.383 | 0.600 | 1.000 | 1.000 | 3.60 | 18.17 |
| heldChanged_trainStable_dirCtx_heldHyp | 0.608 | 0.492 | 0.525 | 0.717 | 0.700 | -0.09 | 1.84 |
| heldChanged_trainStable_dirCtx_trainHyp | 0.660 | 0.508 | 0.492 | 0.842 | 0.800 | -0.89 | 8.96 |
| heldChanged_trainStable_heldCtx_heldHyp | 0.581 | 0.558 | 0.492 | 0.667 | 0.608 | 0.40 | 0.68 |
| heldChanged_trainStable_heldCtx_trainHyp | 0.569 | 0.508 | 0.483 | 0.633 | 0.650 | 0.11 | 2.63 |
| heldChanged_trainStable_laterFirst_heldHyp | 0.742 | 0.567 | 0.433 | 0.992 | 0.975 | -1.69 | 9.80 |
| heldChanged_trainStable_laterFirst_trainHyp | 0.748 | 0.683 | 0.308 | 1.000 | 1.000 | -4.03 | 18.06 |
| heldChanged_trainStable_nonDirCtx_trainHyp | 0.346 | 0.575 | 0.425 | 0.192 | 0.192 | -1.78 | -9.53 |
| heldChanged_trainStable_secondaryFirst_heldHyp | 0.758 | 0.450 | 0.600 | 0.983 | 1.000 | 1.54 | 9.45 |
| heldChanged_trainStable_secondaryFirst_trainHyp | 0.748 | 0.400 | 0.592 | 1.000 | 1.000 | 2.29 | 18.13 |
| heldChanged_trainStable_trainCtx_heldHyp | 0.748 | 0.408 | 0.592 | 1.000 | 0.992 | 1.01 | 10.54 |
| heldChanged_trainStable_trainCtx_trainHyp | 0.748 | 0.267 | 0.725 | 1.000 | 1.000 | 4.77 | 17.63 |
| heldStable_heldStable_dirCtx_heldHyp | 0.690 | 0.642 | 0.625 | 0.742 | 0.750 | -0.28 | 1.99 |
| heldStable_heldStable_dirCtx_trainHyp | 0.783 | 0.833 | 0.825 | 0.725 | 0.750 | 10.33 | 8.60 |
| heldStable_heldStable_heldCtx_heldHyp | 0.681 | 0.683 | 0.700 | 0.708 | 0.633 | 3.06 | 2.15 |
| heldStable_heldStable_heldCtx_trainHyp | 0.637 | 0.642 | 0.650 | 0.617 | 0.642 | 4.50 | 2.26 |
| heldStable_heldStable_laterFirst_heldHyp | 0.944 | 0.975 | 0.967 | 0.917 | 0.917 | 10.77 | 7.65 |
| heldStable_heldStable_laterFirst_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 18.16 | 18.13 |
| heldStable_heldStable_nonDirCtx_trainHyp | 0.223 | 0.208 | 0.208 | 0.233 | 0.242 | -9.03 | -7.07 |
| heldStable_heldStable_secondaryFirst_heldHyp | 0.990 | 1.000 | 1.000 | 0.983 | 0.975 | 9.11 | 10.90 |
| heldStable_heldStable_secondaryFirst_trainHyp | 0.983 | 1.000 | 1.000 | 0.967 | 0.967 | 18.10 | 16.70 |
| heldStable_heldStable_trainCtx_heldHyp | 0.975 | 0.950 | 0.958 | 0.992 | 1.000 | 7.98 | 8.75 |
| heldStable_heldStable_trainCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 18.08 | 17.56 |
| trainChanged_heldStable_dirCtx_heldHyp | 0.592 | 0.533 | 0.425 | 0.733 | 0.675 | -1.78 | 1.66 |
| trainChanged_heldStable_dirCtx_trainHyp | 0.658 | 0.550 | 0.458 | 0.808 | 0.817 | -0.66 | 8.39 |
| trainChanged_heldStable_heldCtx_heldHyp | 0.590 | 0.533 | 0.508 | 0.692 | 0.625 | 0.70 | 0.18 |
| trainChanged_heldStable_heldCtx_trainHyp | 0.585 | 0.467 | 0.533 | 0.675 | 0.667 | 0.54 | 1.77 |
| trainChanged_heldStable_laterFirst_heldHyp | 0.708 | 0.525 | 0.317 | 1.000 | 0.992 | -2.49 | 10.77 |
| trainChanged_heldStable_laterFirst_trainHyp | 0.752 | 0.692 | 0.317 | 1.000 | 1.000 | -3.70 | 18.13 |
| trainChanged_heldStable_nonDirCtx_trainHyp | 0.371 | 0.500 | 0.500 | 0.242 | 0.242 | -1.10 | -7.47 |
| trainChanged_heldStable_secondaryFirst_heldHyp | 0.748 | 0.500 | 0.542 | 0.975 | 0.975 | 0.93 | 10.33 |
| trainChanged_heldStable_secondaryFirst_trainHyp | 0.744 | 0.375 | 0.633 | 0.983 | 0.983 | 2.81 | 16.87 |
| trainChanged_heldStable_trainCtx_heldHyp | 0.744 | 0.475 | 0.550 | 0.975 | 0.975 | 0.57 | 10.33 |
| trainChanged_heldStable_trainCtx_trainHyp | 0.750 | 0.350 | 0.650 | 1.000 | 1.000 | 2.91 | 18.14 |

