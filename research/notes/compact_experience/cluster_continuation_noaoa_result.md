# cluster continuation noaoa result cluster continuation no-AoA result

Created UTC: 2026-08-27T14:21:08Z

Clean-Qwen reference: equal7 = 43.112857; target recovery columns are EWoK, COMPS, GlobalPIQA.

| arm | best | equal7 | Δequal7 vs clean | Δ(EWoK+COMPS+GPIQA) vs clean | Δpreserve(BLiMP+Supp+Entity+Reading) vs clean | EWoK | COMPS | GPIQA | Supp | Entity | Read | BLiMP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E4_untouched_tail | chck_90M | 43.3886 | +0.2757 | -0.7100 | +2.6400 | 50.79 | 51.51 | 35.58 | 62.22 | 28.66 | 8.030 | 66.93 |
| E3_anchor_repeat | chck_99M | 43.3836 | +0.2707 | -0.5550 | +2.4500 | 50.42 | 51.51 | 36.11 | 62.36 | 28.31 | 7.830 | 67.15 |
| E1_true_cluster | chck_90M | 43.2336 | +0.1207 | +0.6750 | +0.1700 | 50.95 | 51.71 | 36.61 | 61.30 | 27.67 | 7.750 | 66.65 |
| E2_anchor_shuffle | chck_90M | 42.9814 | -0.1314 | -1.6850 | +0.7650 | 50.06 | 51.74 | 35.11 | 60.96 | 28.03 | 7.895 | 67.08 |

## E1 best-checkpoint contrasts

- `E1_best_minus_E2_anchor_shuffle_best`: equal7 +0.2521; target-recovery-sum +2.3600; preserve-sum -0.5950; per-column {"BLiMP": -0.4299999999999926, "Supplement": 0.3399999999999963, "EWoK": 0.8900000000000006, "Entity": -0.35999999999999943, "COMPS": -0.030000000000001137, "GlobalPIQA": 1.5, "Reading": -0.14500000000000046}
- `E1_best_minus_E3_anchor_repeat_best`: equal7 -0.1500; target-recovery-sum +1.2300; preserve-sum -2.2800; per-column {"BLiMP": -0.5, "Supplement": -1.0600000000000023, "EWoK": 0.5300000000000011, "Entity": -0.639999999999997, "COMPS": 0.20000000000000284, "GlobalPIQA": 0.5, "Reading": -0.08000000000000007}
- `E1_best_minus_E4_untouched_tail_best`: equal7 -0.1550; target-recovery-sum +1.3850; preserve-sum -2.4700; per-column {"BLiMP": -0.28000000000000114, "Supplement": -0.9200000000000017, "EWoK": 0.1600000000000037, "Entity": -0.9899999999999984, "COMPS": 0.20000000000000284, "GlobalPIQA": 1.0250000000000057, "Reading": -0.28000000000000114}

## Same-checkpoint E1 contrasts

### chck_85M
- `E1_minus_E2_anchor_shuffle`: equal7 +0.4486; target-recovery-sum +3.0400; preserve-sum +0.1000; per-column {"BLiMP": -0.01999999999999602, "Supplement": 0.5399999999999991, "EWoK": -0.09999999999999432, "Entity": -0.38000000000000256, "COMPS": 0.1699999999999946, "GlobalPIQA": 2.969999999999999, "Reading": -0.03999999999999915}
- `E1_minus_E3_anchor_repeat`: equal7 -0.0471; target-recovery-sum +0.4000; preserve-sum -0.7300; per-column {"BLiMP": 0.0800000000000125, "Supplement": 0.21999999999999886, "EWoK": 0.5700000000000003, "Entity": -0.8500000000000014, "COMPS": -0.1700000000000017, "GlobalPIQA": 0.0, "Reading": -0.17999999999999883}
- `E1_minus_E4_untouched_tail`: equal7 +0.5514; target-recovery-sum +4.6150; preserve-sum -0.7550; per-column {"BLiMP": 0.10000000000000853, "Supplement": 0.3200000000000003, "EWoK": -0.39000000000000057, "Entity": -0.9600000000000009, "COMPS": 0.04999999999999716, "GlobalPIQA": 4.954999999999998, "Reading": -0.21499999999999986}
### chck_90M
- `E1_minus_E2_anchor_shuffle`: equal7 +0.2521; target-recovery-sum +2.3600; preserve-sum -0.5950; per-column {"BLiMP": -0.4299999999999926, "Supplement": 0.3399999999999963, "EWoK": 0.8900000000000006, "Entity": -0.35999999999999943, "COMPS": -0.030000000000001137, "GlobalPIQA": 1.5, "Reading": -0.14500000000000046}
- `E1_minus_E3_anchor_repeat`: equal7 +0.0479; target-recovery-sum +1.8550; preserve-sum -1.5200; per-column {"BLiMP": -0.3299999999999983, "Supplement": -0.4100000000000037, "EWoK": 0.8000000000000043, "Entity": -0.6999999999999993, "COMPS": 0.07000000000000028, "GlobalPIQA": 0.9850000000000065, "Reading": -0.08000000000000007}
- `E1_minus_E4_untouched_tail`: equal7 -0.1550; target-recovery-sum +1.3850; preserve-sum -2.4700; per-column {"BLiMP": -0.28000000000000114, "Supplement": -0.9200000000000017, "EWoK": 0.1600000000000037, "Entity": -0.9899999999999984, "COMPS": 0.20000000000000284, "GlobalPIQA": 1.0250000000000057, "Reading": -0.28000000000000114}
### chck_95M
- `E1_minus_E2_anchor_shuffle`: equal7 +0.6371; target-recovery-sum +2.2950; preserve-sum +2.1650; per-column {"BLiMP": -0.03999999999999204, "Supplement": 1.8200000000000003, "EWoK": 0.7000000000000028, "Entity": 0.4299999999999997, "COMPS": 0.11000000000000654, "GlobalPIQA": 1.4850000000000065, "Reading": -0.04500000000000082}
- `E1_minus_E3_anchor_repeat`: equal7 -0.0029; target-recovery-sum +0.5100; preserve-sum -0.5300; per-column {"BLiMP": -0.11999999999999034, "Supplement": -0.28999999999999915, "EWoK": 0.240000000000002, "Entity": -0.07000000000000028, "COMPS": 0.2700000000000031, "GlobalPIQA": 0.0, "Reading": -0.04999999999999982}
- `E1_minus_E4_untouched_tail`: equal7 +0.3521; target-recovery-sum +1.9050; preserve-sum +0.5600; per-column {"BLiMP": 0.060000000000002274, "Supplement": 1.25, "EWoK": -0.35999999999999943, "Entity": -0.4200000000000017, "COMPS": 0.28000000000000114, "GlobalPIQA": 1.9850000000000065, "Reading": -0.33000000000000096}
### chck_99M
- `E1_minus_E2_anchor_shuffle`: equal7 +0.5936; target-recovery-sum +2.1850; preserve-sum +1.9700; per-column {"BLiMP": -0.1600000000000108, "Supplement": 1.4600000000000009, "EWoK": 1.0499999999999972, "Entity": 0.75, "COMPS": 0.14999999999999858, "GlobalPIQA": 0.9850000000000065, "Reading": -0.08000000000000007}
- `E1_minus_E3_anchor_repeat`: equal7 -0.1557; target-recovery-sum +0.3400; preserve-sum -1.4300; per-column {"BLiMP": -0.19000000000001194, "Supplement": -0.6499999999999986, "EWoK": 0.6199999999999974, "Entity": -0.5299999999999976, "COMPS": 0.21999999999999886, "GlobalPIQA": -0.5, "Reading": -0.0600000000000005}
- `E1_minus_E4_untouched_tail`: equal7 +0.2450; target-recovery-sum +2.0100; preserve-sum -0.2950; per-column {"BLiMP": -0.010000000000005116, "Supplement": 0.5700000000000003, "EWoK": 0.21999999999999886, "Entity": -0.5700000000000003, "COMPS": 0.28999999999999915, "GlobalPIQA": 1.5, "Reading": -0.28500000000000014}

Scientific signal: `do_not_promote_cluster_E1_as_frontier_route`.

E1 does not dominate the matched controls on the no-AoA profile needed for a frontier route; the natural shared-anchor cluster family should not receive a 100M run unless a distinct construction changes the active ingredient substantially.
