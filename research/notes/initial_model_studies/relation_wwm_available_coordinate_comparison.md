# relation wwm available coordinate comparison — relation-WWM available-coordinate comparison

Evidence JSON: `experiments/archive/initial_model_studies/data/relation_wwm_available_coordinate_comparison.json`

Higher is better for all columns shown, including Reading.

## 10M: relation_wwm vs shuffled

| column | relation | shuffled | delta |
|---|---:|---:|---:|
| BLiMP | 58.68 | 59.11 | -0.43 |
| Supplement | 56.11 | 60.01 | -3.90 |
| Entity | 19.11 | 17.66 | +1.45 |
| COMPS | 51.21 | 50.18 | +1.03 |
| GlobalPIQA parallel | 19.42 | 23.30 | -3.88 |
| GlobalPIQA nonparallel | 45.00 | 46.00 | -1.00 |
| GlobalPIQA mean | 32.21 | 34.65 | -2.44 |
| Reading eye | 11.48 | 11.44 | +0.04 |
| Reading self-paced | 4.70 | 5.41 | -0.71 |
| Reading mean | 8.09 | 8.43 | -0.34 |

## 20M: relation_wwm vs shuffled

| column | relation | shuffled | delta |
|---|---:|---:|---:|
| BLiMP | 58.68 | 59.12 | -0.44 |
| Supplement | 56.11 | 60.01 | -3.90 |
| Entity | 19.11 | 17.66 | +1.45 |
| COMPS | 51.21 | 50.18 | +1.03 |
| GlobalPIQA parallel | 19.42 | 23.30 | -3.88 |
| GlobalPIQA nonparallel | 45.00 | 46.00 | -1.00 |
| GlobalPIQA mean | 32.21 | 34.65 | -2.44 |
| Reading eye | 11.48 | 11.44 | +0.04 |
| Reading self-paced | 4.70 | 5.41 | -0.71 |
| Reading mean | 8.09 | 8.43 | -0.34 |

## Summary

- known_six_sum_delta_relation_minus_shuffled_10M: -4.6249999999999964
- known_six_sum_delta_relation_minus_shuffled_20M: -4.6349999999999945
- target_entity_globalpiqa_sum_delta_10M: -0.9899999999999984
- target_entity_globalpiqa_sum_delta_20M: -0.9899999999999984
- reading_mean_delta_10M: -0.33500000000000085
- reading_mean_delta_20M: -0.33500000000000085
