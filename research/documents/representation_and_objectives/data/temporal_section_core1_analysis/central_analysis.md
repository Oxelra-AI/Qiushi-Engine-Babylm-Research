# earlier analysis temporal bridge central analysis

Source: `experiments/archive/representation_and_objectives/data/temporal_section_core1/temporal_change_bridge_summary.json`

## Construction snapshot

Base rows 5120 labels {'0': 2560, '1': 2560} queries {'focal_after': 1280, 'focal_before': 1280, 'secondary_after': 1280, 'secondary_before': 1280}

- balanced_temporal: rows 512 train queries {'focal_after': 256, 'focal_before': 256} labels {'0': 256, '1': 256} changed_focal {'False': 256, 'True': 256}
- oracle_secondary: rows 1024 train queries {'focal_after': 256, 'focal_before': 256, 'secondary_after': 256, 'secondary_before': 256} labels {'0': 512, '1': 512} changed_focal {'False': 512, 'True': 512}

## Central held changed-focal / stable-secondary readouts

### balanced_temporal — train_acc 1.000

| eval_set | con_acc | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldChanged_heldStable_trainCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 23.10 | 30.10 |
| heldChanged_heldStable_trainCtx_heldHyp | 0.934 | 0.950 | 0.787 | 1.000 | 1.000 | 6.67 | 19.18 |
| heldChanged_heldStable_heldCtx_trainHyp | 0.775 | 0.550 | 0.769 | 0.894 | 0.887 | 8.37 | 15.31 |
| heldStable_heldStable_trainCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 30.84 | 30.71 |
| heldStable_heldStable_trainCtx_heldHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 22.00 | 20.05 |

On `heldChanged_heldStable_trainCtx_trainHyp`: focal_after=1.000 (margin 23.10), secondary_after=1.000 (margin 30.10).

### oracle_secondary — train_acc 1.000

| eval_set | con_acc | fb | fa | sb | sa | fa_m | sa_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldChanged_heldStable_trainCtx_trainHyp | 0.998 | 1.000 | 0.994 | 1.000 | 1.000 | 15.52 | 33.18 |
| heldChanged_heldStable_trainCtx_heldHyp | 0.992 | 0.969 | 1.000 | 1.000 | 1.000 | 15.83 | 33.52 |
| heldChanged_heldStable_heldCtx_trainHyp | 0.911 | 0.925 | 0.762 | 0.969 | 0.988 | 2.16 | 4.73 |
| heldStable_heldStable_trainCtx_trainHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 34.81 | 32.78 |
| heldStable_heldStable_trainCtx_heldHyp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 35.41 | 33.44 |

On `heldChanged_heldStable_trainCtx_trainHyp`: focal_after=0.994 (margin 15.52), secondary_after=1.000 (margin 33.18).

## Train histories

- balanced_temporal seed 26500: best 1.000; e1:0.500, e2:0.974, e3:0.977, e4:0.976, e5:0.985, e6:0.997, e7:1.000, e8:1.000, e9:0.999, e10:0.997
- oracle_secondary seed 26500: best 1.000; e1:0.500, e2:0.882, e3:0.979, e4:0.996, e5:1.000, e6:1.000, e7:1.000, e8:1.000, e9:1.000, e10:0.999
