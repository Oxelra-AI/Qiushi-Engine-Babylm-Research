# aoa overall update sparse temporal probe

Existing compact_view_reinvest checkpoint ladders only; selected task slices, not full official evaluation.

## blimp_control
- label: no_large_final_gap
- seed43122-minus-seed43022 by exposure: {1: 7.9375, 10: 1.6875, 40: 1.0625, 100: 10.0625}
- early mean 1/10M: 4.8125 ; late mean 40/100M: 5.5625 ; final: 10.0625

## blimp_worst
- label: early_seed_gap
- seed43122-minus-seed43022 by exposure: {1: -7.0, 10: 2.1499999999999986, 40: -1.3000000000000043, 100: -14.049999999999997}
- early mean 1/10M: -2.4250000000000007 ; late mean 40/100M: -7.675000000000001 ; final: -14.049999999999997

## entity_full
- label: late_divergence
- seed43122-minus-seed43022 by exposure: {1: 0.6202771047875082, 10: -1.333708144538452, 40: -2.902960414488472, 100: -1.4604021386814097}
- early mean 1/10M: -0.3567155198754719 ; late mean 40/100M: -2.181681276584941 ; final: -1.4604021386814097

## ewok_all
- label: late_divergence
- seed43122-minus-seed43022 by exposure: {1: 0.45454545454545325, 10: 0.0, 40: -2.4545454545454533, 100: -3.7272727272727266}
- early mean 1/10M: 0.22727272727272663 ; late mean 40/100M: -3.09090909090909 ; final: -3.7272727272727266

## supplement_all
- label: early_seed_gap
- seed43122-minus-seed43022 by exposure: {1: -6.0, 10: -3.200000000000003, 40: 2.4000000000000057, 100: -3.200000000000003}
- early mean 1/10M: -4.600000000000001 ; late mean 40/100M: -0.3999999999999986 ; final: -3.200000000000003

Machine-readable output: `experiments/archive/frontier_consolidation/training/runs/sparse_temporal_probe_main_direct/sparse_temporal_probe_summary.json`
