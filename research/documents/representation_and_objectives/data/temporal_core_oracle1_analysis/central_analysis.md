# earlier analysis temporal bridge central analysis

Source: `experiments/archive/representation_and_objectives/data/temporal_core_oracle1/temporal_change_bridge_summary.json`

## Construction snapshot

Base rows 5120 labels {'0': 2560, '1': 2560} queries {'focal_after': 1280, 'focal_before': 1280, 'secondary_after': 1280, 'secondary_before': 1280}

- stable_only: rows 512 train queries {'focal_after': 128, 'focal_before': 128, 'neutral_mention': 256} labels {'0': 256, '1': 256} changed_focal {'False': 256, 'True': 256}
- balanced_temporal: rows 512 train queries {'focal_after': 256, 'focal_before': 256} labels {'0': 256, '1': 256} changed_focal {'False': 256, 'True': 256}
- oracle_secondary: rows 1024 train queries {'focal_after': 256, 'focal_before': 256, 'secondary_after': 256, 'secondary_before': 256} labels {'0': 512, '1': 512} changed_focal {'False': 512, 'True': 512}

## Central held changed-focal / stable-secondary readouts

### balanced_temporal — train_acc 0.984

| eval_set | con_acc | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldChanged_heldStable_trainCtx_trainHyp | 0.784 | 0.938 | 0.200 | 1.000 | 1.000 | -2.16 | 8.49 |
| heldChanged_heldStable_trainCtx_heldHyp | 0.750 | 0.787 | 0.219 | 1.000 | 0.994 | -2.62 | 4.40 |
| heldChanged_heldStable_heldCtx_trainHyp | 0.602 | 0.444 | 0.456 | 0.756 | 0.750 | -0.54 | 5.22 |
| heldStable_heldStable_trainCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 17.44 | 17.53 |
| heldStable_heldStable_trainCtx_heldHyp | 0.997 | 1.000 | 1.000 | 0.994 | 0.994 | 14.15 | 12.42 |

On `heldChanged_heldStable_trainCtx_trainHyp`: focal_after=0.200 (margin -2.16), secondary_after=1.000 (margin 8.49).

### oracle_secondary — train_acc 0.979

| eval_set | con_acc | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldChanged_heldStable_trainCtx_trainHyp | 0.750 | 0.375 | 0.625 | 1.000 | 1.000 | 0.03 | 15.73 |
| heldChanged_heldStable_trainCtx_heldHyp | 0.756 | 0.487 | 0.537 | 1.000 | 1.000 | 0.03 | 11.82 |
| heldChanged_heldStable_heldCtx_trainHyp | 0.616 | 0.537 | 0.475 | 0.725 | 0.725 | -0.19 | 3.94 |
| heldStable_heldStable_trainCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 16.03 | 17.43 |
| heldStable_heldStable_trainCtx_heldHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 13.92 | 14.48 |

On `heldChanged_heldStable_trainCtx_trainHyp`: focal_after=0.625 (margin 0.03), secondary_after=1.000 (margin 15.73).

### stable_only — train_acc 1.000

| eval_set | con_acc | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldChanged_heldStable_trainCtx_trainHyp | 0.750 | 0.319 | 0.681 | 1.000 | 1.000 | 5.32 | 21.33 |
| heldChanged_heldStable_trainCtx_heldHyp | 0.725 | 0.375 | 0.606 | 0.956 | 0.963 | 0.32 | 3.59 |
| heldChanged_heldStable_heldCtx_trainHyp | 0.561 | 0.544 | 0.456 | 0.625 | 0.619 | -1.31 | 9.39 |
| heldStable_heldStable_trainCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 26.79 | 26.81 |
| heldStable_heldStable_trainCtx_heldHyp | 0.998 | 1.000 | 0.994 | 1.000 | 1.000 | 5.21 | 3.69 |

On `heldChanged_heldStable_trainCtx_trainHyp`: focal_after=0.681 (margin 5.32), secondary_after=1.000 (margin 21.33).

## Train histories

- stable_only seed 26500: best 1.000; e1:0.986, e2:0.989, e3:0.992, e4:1.000, e5:0.997, e6:1.000, e7:0.999, e8:0.997, e9:0.997, e10:0.999
- balanced_temporal seed 26500: best 0.984; e1:0.977, e2:0.977, e3:0.977, e4:0.977, e5:0.977, e6:0.975, e7:0.971, e8:0.979, e9:0.979, e10:0.984
- oracle_secondary seed 26500: best 0.979; e1:0.979, e2:0.978, e3:0.979, e4:0.976, e5:0.978, e6:0.977, e7:0.966, e8:0.979, e9:0.979, e10:0.974
