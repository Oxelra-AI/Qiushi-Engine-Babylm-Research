# earlier analysis temporal bridge central analysis

Source: `experiments/archive/representation_and_objectives/data/temporal_pilot1/temporal_change_bridge_summary.json`

## Construction snapshot

Base rows 1920 labels {'0': 960, '1': 960} queries {'focal_after': 480, 'focal_before': 480, 'secondary_after': 480, 'secondary_before': 480}

- exposure: rows 384 train queries {'neutral_mention': 384} labels {'0': 192, '1': 192} changed_focal {'False': 192, 'True': 192}
- stable_only: rows 384 train queries {'focal_after': 96, 'focal_before': 96, 'neutral_mention': 192} labels {'0': 192, '1': 192} changed_focal {'False': 192, 'True': 192}
- changed_only: rows 384 train queries {'focal_after': 96, 'focal_before': 96, 'neutral_mention': 192} labels {'0': 192, '1': 192} changed_focal {'False': 192, 'True': 192}
- balanced_temporal: rows 384 train queries {'focal_after': 192, 'focal_before': 192} labels {'0': 192, '1': 192} changed_focal {'False': 192, 'True': 192}

## Central held changed-focal / stable-secondary readouts

### balanced_temporal — train_acc 0.901

| eval_set | con_acc | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldChanged_heldStable_trainCtx_trainHyp | 0.746 | 0.433 | 0.567 | 0.992 | 0.992 | 0.09 | 2.12 |
| heldChanged_heldStable_trainCtx_heldHyp | 0.748 | 0.467 | 0.525 | 1.000 | 1.000 | 0.08 | 1.41 |
| heldChanged_heldStable_heldCtx_trainHyp | 0.606 | 0.508 | 0.492 | 0.717 | 0.708 | -0.01 | 1.03 |
| heldStable_heldStable_trainCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.64 | 1.97 |
| heldStable_heldStable_trainCtx_heldHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.40 | 1.31 |

On `heldChanged_heldStable_trainCtx_trainHyp`: focal_after=0.567 (margin 0.09), secondary_after=0.992 (margin 2.12).

### changed_only — train_acc 0.839

| eval_set | con_acc | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldChanged_heldStable_trainCtx_trainHyp | 0.544 | 0.542 | 0.442 | 0.600 | 0.592 | -0.10 | 0.16 |
| heldChanged_heldStable_trainCtx_heldHyp | 0.348 | 0.425 | 0.492 | 0.217 | 0.258 | 0.05 | -0.04 |
| heldChanged_heldStable_heldCtx_trainHyp | 0.492 | 0.525 | 0.458 | 0.492 | 0.492 | 0.10 | 0.01 |
| heldStable_heldStable_trainCtx_trainHyp | 0.554 | 0.583 | 0.592 | 0.517 | 0.525 | 0.15 | 0.24 |
| heldStable_heldStable_trainCtx_heldHyp | 0.352 | 0.317 | 0.325 | 0.367 | 0.400 | -0.01 | 0.06 |

On `heldChanged_heldStable_trainCtx_trainHyp`: focal_after=0.442 (margin -0.10), secondary_after=0.592 (margin 0.16).

### exposure — train_acc 0.535

| eval_set | con_acc | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldChanged_heldStable_trainCtx_trainHyp | 0.442 | 0.525 | 0.467 | 0.392 | 0.383 | 0.02 | -0.06 |
| heldChanged_heldStable_trainCtx_heldHyp | 0.485 | 0.492 | 0.500 | 0.458 | 0.492 | -0.07 | -0.06 |
| heldChanged_heldStable_heldCtx_trainHyp | 0.533 | 0.442 | 0.533 | 0.583 | 0.575 | -0.01 | 0.00 |
| heldStable_heldStable_trainCtx_trainHyp | 0.502 | 0.567 | 0.550 | 0.433 | 0.458 | -0.01 | -0.08 |
| heldStable_heldStable_trainCtx_heldHyp | 0.540 | 0.592 | 0.667 | 0.442 | 0.458 | -0.01 | -0.06 |

On `heldChanged_heldStable_trainCtx_trainHyp`: focal_after=0.467 (margin 0.02), secondary_after=0.383 (margin -0.06).

### stable_only — train_acc 0.990

| eval_set | con_acc | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldChanged_heldStable_trainCtx_trainHyp | 0.746 | 0.383 | 0.600 | 1.000 | 1.000 | 3.60 | 18.17 |
| heldChanged_heldStable_trainCtx_heldHyp | 0.725 | 0.508 | 0.450 | 0.975 | 0.967 | 0.04 | 9.33 |
| heldChanged_heldStable_heldCtx_trainHyp | 0.598 | 0.533 | 0.467 | 0.692 | 0.700 | -0.83 | 2.94 |
| heldStable_heldStable_trainCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 18.08 | 17.56 |
| heldStable_heldStable_trainCtx_heldHyp | 0.975 | 0.950 | 0.958 | 0.992 | 1.000 | 7.98 | 8.75 |

On `heldChanged_heldStable_trainCtx_trainHyp`: focal_after=0.600 (margin 3.60), secondary_after=1.000 (margin 18.17).

## Train histories

- exposure seed 26500: best 0.535; e1:0.500, e2:0.500, e3:0.500, e4:0.535, e5:0.500, e6:0.500
- stable_only seed 26500: best 0.990; e1:0.500, e2:0.500, e3:0.500, e4:0.500, e5:0.982, e6:0.990
- changed_only seed 26500: best 0.839; e1:0.500, e2:0.500, e3:0.500, e4:0.500, e5:0.500, e6:0.839
- balanced_temporal seed 26500: best 0.901; e1:0.500, e2:0.500, e3:0.500, e4:0.500, e5:0.500, e6:0.901
