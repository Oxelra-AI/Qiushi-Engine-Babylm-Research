# fastpath closure and ordinary84 endpoint within-trajectory fast-path readout

**Decision: STOP_FASTPATH_INSTANTIATION**

This is a same-seed scale-1.75 trajectory test: chck_80M is only two million words before the original chck_82M anchor. A positive result would not prove trajectory-general stability-plasticity; it would justify a truly independent seed or trajectory test.

## Cheap7

| Arm | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | Cheap7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor80 | 68.120 | 62.620 | 49.240 | 28.200 | 52.110 | 38.105 | 8.300 | 43.8136 |
| coherent80 | 68.230 | 62.750 | 48.920 | 28.240 | 52.100 | 37.105 | 8.345 | 43.6700 |
| shuffled80 | 67.970 | 61.580 | 47.980 | 24.190 | 52.330 | 37.090 | 8.350 | 42.7843 |
| ordinary84 | 68.240 | 63.480 | 50.070 | 28.580 | 52.210 | 38.120 | 8.155 | 44.1221 |

Cheap7 beats anchor, ordinary, and shuffled: **FAIL**

## R3: task-balanced retention and discovery

| Column | Coh ret | Ord ret | Shuf ret | Ret ≥ both | Coh disc | Ord disc | Shuf disc | Disc ≥ both | Coh net | Ord net | Shuf net |
|---|---:|---:|---:|:---:|---:|---:|---:|:---:|---:|---:|---:|
| BLiMP | 0.9837 | 0.9664 | 0.9613 | ✓ | 0.0385 | 0.0758 | 0.0780 | ✗ | +72 | +87 | -80 |
| Supplement | 0.9875 | 0.9758 | 0.9649 | ✓ | 0.0380 | 0.0753 | 0.0826 | ✗ | +4 | +10 | -22 |
| EWoK | 0.9544 | 0.9230 | 0.8745 | ✓ | 0.0458 | 0.0870 | 0.1113 | ✗ | +6 | +47 | -41 |
| Entity | 0.9473 | 0.9170 | 0.6798 | ✓ | 0.0204 | 0.0363 | 0.0922 | ✗ | +1 | +22 | -150 |
| COMPS | 0.9569 | 0.9110 | 0.8952 | ✓ | 0.0473 | 0.1019 | 0.1200 | ✗ | -18 | +140 | +168 |
| GlobalPIQA | 0.9351 | 0.9221 | 0.8961 | ✓ | 0.0238 | 0.0476 | 0.0476 | ✗ | -2 | +0 | -2 |

Retention wins vs both controls: 6/6 (need ≥4)
Discovery wins vs both controls: 0/6 (need ≥3)
R3 vs both controls: **FAIL**

## R4: prediction-change-zone signature

The change zone is selected by prediction changes relative to the frozen anchor, before looking at correctness. Helpful residuals should have positive benefit and exceed ordinary and shuffled controls.

| Arm | change-zone size | cand correct | anchor correct | cand acc | anchor acc | benefit | total net vs anchor |
|---|---:|---:|---:|---:|---:|---:|---:|
| coherent80 | 6410 | 3111 | 3048 | 0.4853 | 0.4755 | +0.0098 | +63 |
| ordinary84 | 13038 | 6476 | 6170 | 0.4967 | 0.4732 | +0.0235 | +306 |
| shuffled80 | 16614 | 7677 | 7804 | 0.4621 | 0.4697 | -0.0076 | -127 |

R4 positive and above ordinary/shuffled: **FAIL**

## Multi-arm sharing

- coherent gains unique against both controls: 682/3111
- coherent gains also correct under shuffled: 1717/3111
- coherent gains also correct under ordinary: 1743/3111
- coherent losses unique against both controls: 679/3048
- unique gain minus unique loss: +3

## Scientific interpretation

- Fast-path instantiation should stop under the fastpath closure and ordinary84 endpoint readout because cheap7 does not beat all matched alternatives; task-balanced retention/discovery does not beat both ordinary and shuffled controls; prediction-change-zone benefit is not positive and above both controls.
- The anchor is chck_80M from the same seed43022 scale-1.75 trajectory, so any positive result would indicate within-trajectory checkpoint robustness rather than an independent-anchor or trajectory-general learning principle.
- Pooled discrete item nets vs anchor: coherent +63, ordinary +306, shuffled -127.
- Pooled prediction-change benefits: coherent +0.0098, ordinary +0.0235, shuffled -0.0076.
- Coherent gains unique against both controls: 682/3111; coherent losses unique against both controls: 679; gain-overlap with shuffled: 1717/3111.

JSON: `experiments/archive/representation_and_objectives/data/fastpath80_fourway_readout/task_balanced_readout.json`
