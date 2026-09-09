# fastpath closure and ordinary84 endpoint within-trajectory fast-path readout

**Decision: STOP_FASTPATH_INSTANTIATION**

This is a same-seed scale-1.75 trajectory test: chck_80M is only two million words before the original chck_82M anchor. A positive result would not prove trajectory-general stability-plasticity; it would justify a truly independent seed or trajectory test.

## Cheap7

| Arm | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | Cheap7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor80 | 68.491 | 62.938 | 50.055 | 28.314 | 52.191 | 37.578 | 8.149 | 43.9594 |
| coherent80 | 68.520 | 63.650 | 49.910 | 28.440 | 51.990 | 38.065 | 8.170 | 44.1064 |
| shuffled80 | 68.180 | 62.360 | 49.370 | 24.240 | 52.040 | 37.565 | 8.095 | 43.1214 |
| ordinary84 | 68.470 | 62.680 | 50.140 | 28.580 | 52.290 | 36.135 | 8.100 | 43.7707 |

Cheap7 beats anchor, ordinary, and shuffled: **PASS**

## R3: task-balanced retention and discovery

| Column | Coh ret | Ord ret | Shuf ret | Ret ≥ both | Coh disc | Ord disc | Shuf disc | Disc ≥ both | Coh net | Ord net | Shuf net |
|---|---:|---:|---:|:---:|---:|---:|---:|:---:|---:|---:|---:|
| BLiMP | 0.9821 | 0.9711 | 0.9523 | ✓ | 0.0402 | 0.0619 | 0.0935 | ✗ | +30 | -12 | -182 |
| Supplement | 0.9868 | 0.9835 | 0.9628 | ✓ | 0.0274 | 0.0527 | 0.0749 | ✗ | -14 | +7 | -43 |
| EWoK | 0.9608 | 0.9343 | 0.8846 | ✓ | 0.0385 | 0.0721 | 0.1120 | ✗ | -2 | +25 | -12 |
| Entity | 0.9651 | 0.9344 | 0.6642 | ✓ | 0.0149 | 0.0297 | 0.0994 | ✗ | +7 | +21 | -149 |
| COMPS | 0.9534 | 0.9245 | 0.8763 | ✓ | 0.0486 | 0.0857 | 0.1349 | ✗ | -137 | +80 | -105 |
| GlobalPIQA | 0.9737 | 0.8816 | 0.9342 | ✓ | 0.0236 | 0.0472 | 0.0394 | ✗ | +1 | -3 | +0 |

Retention wins vs both controls: 6/6 (need ≥4)
Discovery wins vs both controls: 0/6 (need ≥3)
R3 vs both controls: **FAIL**

## R4: prediction-change-zone signature

The change zone is selected by prediction changes relative to the frozen anchor, before looking at correctness. Helpful residuals should have positive benefit and exceed ordinary and shuffled controls.

| Arm | change-zone size | cand correct | anchor correct | cand acc | anchor acc | benefit | total net vs anchor |
|---|---:|---:|---:|---:|---:|---:|---:|
| coherent80 | 6521 | 3116 | 3231 | 0.4778 | 0.4955 | -0.0176 | -115 |
| ordinary84 | 10921 | 5366 | 5248 | 0.4913 | 0.4805 | +0.0108 | +118 |
| shuffled80 | 18821 | 8608 | 9099 | 0.4574 | 0.4834 | -0.0261 | -491 |

R4 positive and above ordinary/shuffled: **FAIL**

## Multi-arm sharing

- coherent gains unique against both controls: 670/3116
- coherent gains also correct under shuffled: 1651/3116
- coherent gains also correct under ordinary: 1791/3116
- coherent losses unique against both controls: 699/3231
- unique gain minus unique loss: -29

## Scientific interpretation

- Fast-path instantiation should stop under the fastpath closure and ordinary84 endpoint readout because task-balanced retention/discovery does not beat both ordinary and shuffled controls; prediction-change-zone benefit is not positive and above both controls.
- The anchor is chck_80M from the same seed43022 scale-1.75 trajectory, so any positive result would indicate within-trajectory checkpoint robustness rather than an independent-anchor or trajectory-general learning principle.
- Pooled discrete item nets vs anchor: coherent -115, ordinary +118, shuffled -491.
- Pooled prediction-change benefits: coherent -0.0176, ordinary +0.0108, shuffled -0.0261.
- Coherent gains unique against both controls: 670/3116; coherent losses unique against both controls: 699; gain-overlap with shuffled: 1651/3116.

JSON: `experiments/archive/representation_and_objectives/data/readout_dryrun_82to86/task_balanced_readout_d3717bb0.json`
