# clean full trajectory seed control sparse difference-in-differences: exact_reinvest_vs_clean

This compares seed43122-minus-seed43022 under compact_view_reinvest with the same seed gap under clean-Qwen.

## Task-group excess gaps
### 1M
- blimp_control: reinvest_gap=7.938, clean_gap=8.750, excess=-0.812
- blimp_worst: reinvest_gap=-7.000, clean_gap=-9.400, excess=2.400
- entity_full: reinvest_gap=0.620, clean_gap=0.485, excess=0.136
- ewok_all: reinvest_gap=0.455, clean_gap=0.091, excess=0.364
- supplement_all: reinvest_gap=-6.000, clean_gap=-5.200, excess=-0.800
### 10M
- blimp_control: reinvest_gap=1.688, clean_gap=-3.500, excess=5.188
- blimp_worst: reinvest_gap=2.150, clean_gap=-7.700, excess=9.850
- entity_full: reinvest_gap=-1.334, clean_gap=1.172, excess=-2.506
- ewok_all: reinvest_gap=0.000, clean_gap=3.364, excess=-3.364
- supplement_all: reinvest_gap=-3.200, clean_gap=-0.800, excess=-2.400
### 40M
- blimp_control: reinvest_gap=1.062, clean_gap=3.188, excess=-2.125
- blimp_worst: reinvest_gap=-1.300, clean_gap=0.400, excess=-1.700
- entity_full: reinvest_gap=-2.903, clean_gap=-4.401, excess=1.498
- ewok_all: reinvest_gap=-2.455, clean_gap=1.364, excess=-3.818
- supplement_all: reinvest_gap=2.400, clean_gap=-4.800, excess=7.200
### 100M
- blimp_control: reinvest_gap=10.062, clean_gap=2.938, excess=7.125
- blimp_worst: reinvest_gap=-14.050, clean_gap=-4.650, excess=-9.400
- entity_full: reinvest_gap=-1.460, clean_gap=-0.502, excess=-0.958
- ewok_all: reinvest_gap=-3.727, clean_gap=1.455, excess=-5.182
- supplement_all: reinvest_gap=-3.200, clean_gap=0.800, excess=-4.000

## Summary by task group
- blimp_control: excess_by_exposure={'1': -0.8125, '10': 5.1875, '40': -2.125, '100': 7.125}, mean=2.344, final=7.125, max_abs=7.125
- blimp_worst: excess_by_exposure={'1': 2.3999999999999986, '10': 9.850000000000001, '40': -1.7000000000000028, '100': -9.399999999999999}, mean=0.287, final=-9.400, max_abs=9.850
- entity_full: excess_by_exposure={'1': 0.1357205904810037, '10': -2.505772550823787, '40': 1.4979622190831208, '100': -0.9581975887792709}, mean=-0.458, final=-0.958, max_abs=2.506
- ewok_all: excess_by_exposure={'1': 0.36363636363636687, '10': -3.363636363636367, '40': -3.818181818181813, '100': -5.18181818181818}, mean=-3.000, final=-5.182, max_abs=5.182
- supplement_all: excess_by_exposure={'1': -0.7999999999999972, '10': -2.3999999999999986, '40': 7.200000000000003, '100': -4.0}, mean=0.000, final=-4.000, max_abs=7.200

Interpretation: excess near zero means the seed spread is inherited; negative excess means the compact-view reinvestment route enlarges seed43122's shortfall on that slice; positive excess means the seed gap is smaller under reinvestment.

Machine-readable output: `experiments/archive/frontier_consolidation/data/sparse_temporal_did_exact/exact_reinvest_vs_clean_sparse_did.json`
