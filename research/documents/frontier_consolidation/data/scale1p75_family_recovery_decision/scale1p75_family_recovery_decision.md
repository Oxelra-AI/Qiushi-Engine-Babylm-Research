# earlier analysis scale1.75 family-recovery decision

Route signal: `close_fixed_scale1p75_as_localized_redistribution`

## Rule

Only launch exact 100M if losing families (BLiMP island/NPI and EWoK material/quantitative/social-interaction groups) recover while winning superlative/wh/high-operation Entity families persist; otherwise close fixed scale1.75 even if cheap7 remains modestly positive.

## Exposure readouts

| exposure | cheap7 Δ | EWoK Δ | GP Δ | Reading Δ | win net | win net % | loss net | loss net % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50M | +0.4114 | -0.730 | -1.475 | -0.235 | 833 | +16.21 | -620 | -7.78 |
| 70M | +0.0614 | -1.920 | -0.870 | -0.700 | 269 | +5.23 | 84 | +1.05 |
| 80M | +0.8636 | -1.770 | +2.525 | +0.010 | 252 | +4.90 | 43 | +0.54 |

## Interpretation

- 50M: cheap7 Δ +0.4114; watch-column deltas {'EWoK': -0.730000000000004, 'GlobalPIQA': -1.4749999999999943, 'Reading': -0.23499999999999943}
- 50M: named win net 833 (+16.21%), named loss net -620 (-7.78%).
- 70M: cheap7 Δ +0.0614; watch-column deltas {'EWoK': -1.9200000000000017, 'GlobalPIQA': -0.8699999999999974, 'Reading': -0.6999999999999993}
- 70M: named win net 269 (+5.23%), named loss net 84 (+1.05%).
- 80M: cheap7 Δ +0.8636; watch-column deltas {'EWoK': -1.769999999999996, 'GlobalPIQA': 2.5250000000000057, 'Reading': 0.010000000000001563}
- 80M: named win net 252 (+4.90%), named loss net 43 (+0.54%).
- 80M does not pass the family-recovery pattern (cheap7 Δ=+0.8636, win_net=252, loss_net=43, watch_bad={'EWoK': -1.769999999999996}). A positive aggregate alone is not enough if the same families stay traded.

## Comparator status

- 50M: {'status': 'ran', 'returncode': 0, 'stdout_tail': '{\n  "status": "PAIRWISE_ITEM_FLIP_ANALYSIS",\n  "label": "scale1p75_vs_step35_50M",\n  "out_json": "experiments/archive/frontier_consolidation/data/scale1p75_family_recovery_decision/scale1p75_vs_50M.json",\n  "out_md": "research/documents/frontier_consolidation/data/scale1p75_family_recovery_decision/scale1p75_vs_50M.md",\n  "cheap7_delta": 0.4114285714285728,\n  "discrete_mean_delta": 0.5183138281651581,\n  "total_net_items": 1951\n}\n', 'stderr_tail': '', 'path': 'experiments/archive/frontier_consolidation/data/scale1p75_family_recovery_decision/scale1p75_vs_50M.json'}
- 70M: {'status': 'ran', 'returncode': 0, 'stdout_tail': '{\n  "status": "PAIRWISE_ITEM_FLIP_ANALYSIS",\n  "label": "scale1p75_vs_step35_70M",\n  "out_json": "experiments/archive/frontier_consolidation/data/scale1p75_family_recovery_decision/scale1p75_vs_70M.json",\n  "out_md": "research/documents/frontier_consolidation/data/scale1p75_family_recovery_decision/scale1p75_vs_70M.md",\n  "cheap7_delta": 0.06142857142857139,\n  "discrete_mean_delta": 0.18892690934558173,\n  "total_net_items": 1445\n}\n', 'stderr_tail': '', 'path': 'experiments/archive/frontier_consolidation/data/scale1p75_family_recovery_decision/scale1p75_vs_70M.json'}
- 80M: {'status': 'ran', 'returncode': 0, 'stdout_tail': '{\n  "status": "PAIRWISE_ITEM_FLIP_ANALYSIS",\n  "label": "scale1p75_vs_step35_80M",\n  "out_json": "experiments/archive/frontier_consolidation/data/scale1p75_family_recovery_decision/scale1p75_vs_80M.json",\n  "out_md": "research/documents/frontier_consolidation/data/scale1p75_family_recovery_decision/scale1p75_vs_80M.md",\n  "cheap7_delta": 0.8635714285714329,\n  "discrete_mean_delta": 1.0089167906118301,\n  "total_net_items": 1087\n}\n', 'stderr_tail': '', 'path': 'experiments/archive/frontier_consolidation/data/scale1p75_family_recovery_decision/scale1p75_vs_80M.json'}
