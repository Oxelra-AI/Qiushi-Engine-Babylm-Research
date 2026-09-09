# entity consistency 20m training validation — Entity Mention Consistency available-coordinate comparison

Evidence JSON: `experiments/archive/initial_model_studies/data/entity_consistency_available_coordinate_comparison.json`

Higher is better for all columns including Reading.

## 10M: consistency vs shuffled_pair

| column | consistency | shuffled_pair | delta |
|---|---:|---:|---:|
| BLiMP | 55.45 | 55.18 | +0.27 |
| Supplement | 54.20 | 53.73 | +0.47 |
| Entity | 17.49 | 17.43 | +0.06 |
| COMPS | 50.48 | 50.25 | +0.23 |
| GlobalPIQA parallel | 22.33 | 20.39 | +1.94 |
| GlobalPIQA nonparallel | 42.00 | 40.00 | +2.00 |
| GlobalPIQA mean | 32.16 | 30.20 | +1.97 |
| Reading eye | 12.21 | 12.24 | -0.03 |
| Reading self-paced | 4.88 | 5.12 | -0.24 |
| Reading mean | 8.54 | 8.68 | -0.13 |

## 20M: consistency vs shuffled_pair

| column | consistency | shuffled_pair | delta |
|---|---:|---:|---:|
| BLiMP | 61.04 | 59.00 | +2.04 |
| Supplement | 57.61 | 58.59 | -0.98 |
| Entity | 17.87 | 17.48 | +0.39 |
| COMPS | 50.59 | 50.62 | -0.03 |
| GlobalPIQA parallel | 23.30 | 18.45 | +4.85 |
| GlobalPIQA nonparallel | 47.00 | 44.00 | +3.00 |
| GlobalPIQA mean | 35.15 | 31.23 | +3.92 |
| Reading eye | 10.76 | 10.89 | -0.13 |
| Reading self-paced | 4.60 | 4.89 | -0.29 |
| Reading mean | 7.68 | 7.89 | -0.21 |

## Summary

- known_six_sum_delta_10M: 2.8650000000000038
- known_six_sum_delta_20M: 5.134999999999998
- target_entity_globalpiqa_sum_delta_10M: 2.0299999999999976
- target_entity_globalpiqa_sum_delta_20M: 4.314999999999998
- guard_supp_blimp_reading_sum_delta_10M: 0.6050000000000093
- guard_supp_blimp_reading_sum_delta_20M: 0.8499999999999943
- entity_delta_10M: 0.05999999999999872
- entity_delta_20M: 0.39000000000000057
- globalpiqa_mean_delta_10M: 1.9699999999999989
- globalpiqa_mean_delta_20M: 3.924999999999997
- reading_mean_delta_10M: -0.1349999999999998
- reading_mean_delta_20M: -0.21000000000000085
