# seed2 entity consistency 20m training validation — seed2 Entity Mention Consistency available-coordinate comparison

Evidence JSON: `experiments/archive/initial_model_studies/data/seed2_entity_consistency_available_coordinate_comparison.json`

Higher is better for all columns including Reading.

## 10M: consistency vs shuffled_pair

| column | consistency | shuffled_pair | delta |
|---|---:|---:|---:|
| BLiMP | 54.51 | 54.74 | -0.23 |
| Supplement | 53.40 | 51.47 | +1.93 |
| Entity | 17.01 | 17.13 | -0.12 |
| COMPS | 50.02 | 50.22 | -0.20 |
| GlobalPIQA parallel | 21.36 | 20.39 | +0.97 |
| GlobalPIQA nonparallel | 47.00 | 45.00 | +2.00 |
| GlobalPIQA mean | 34.18 | 32.70 | +1.48 |
| Reading eye | 10.67 | 10.63 | +0.04 |
| Reading self-paced | 4.26 | 4.36 | -0.10 |
| Reading mean | 7.46 | 7.50 | -0.03 |

## 20M: consistency vs shuffled_pair

| column | consistency | shuffled_pair | delta |
|---|---:|---:|---:|
| BLiMP | 59.33 | 59.84 | -0.51 |
| Supplement | 56.27 | 56.44 | -0.17 |
| Entity | 17.68 | 17.47 | +0.21 |
| COMPS | 50.87 | 50.44 | +0.43 |
| GlobalPIQA parallel | 20.39 | 18.45 | +1.94 |
| GlobalPIQA nonparallel | 41.00 | 39.00 | +2.00 |
| GlobalPIQA mean | 30.70 | 28.73 | +1.97 |
| Reading eye | 11.00 | 11.96 | -0.96 |
| Reading self-paced | 3.85 | 4.94 | -1.09 |
| Reading mean | 7.42 | 8.45 | -1.03 |

## Summary

- known_six_sum_delta_10M: 2.835000000000001
- known_six_sum_delta_20M: 0.9049999999999985
- target_entity_globalpiqa_sum_delta_10M: 1.365000000000002
- target_entity_globalpiqa_sum_delta_20M: 2.1799999999999997
- guard_supp_blimp_reading_sum_delta_10M: 1.6699999999999946
- guard_supp_blimp_reading_sum_delta_20M: -1.705000000000001
- entity_delta_10M: -0.11999999999999744
- entity_delta_20M: 0.21000000000000085
- globalpiqa_mean_delta_10M: 1.4849999999999994
- globalpiqa_mean_delta_20M: 1.9699999999999989
- reading_mean_delta_10M: -0.030000000000001137
- reading_mean_delta_20M: -1.0250000000000012
