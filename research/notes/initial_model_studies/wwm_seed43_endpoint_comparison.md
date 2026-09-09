# wwm seed43 endpoint comparison wwm_seed43 endpoint comparison

Evidence JSON: `data/wwm_seed43_endpoint_comparison.json`

## Endpoint scores

| endpoint | blimp | supplement | entity_tracking | ewok | comps | GlobalPIQA_mean | Reading_mean | superglue | endpoint_8col_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wwm_seed43_chck80M | 66.540 | 61.000 | 22.200 | 50.440 | 53.000 | 37.590 | 7.300 | 68.255 | 45.791 |
| wwm_seed43_chck100M | 67.020 | 61.390 | 21.780 | 51.140 | 53.110 | 36.120 | 7.330 | 67.968 | 45.732 |
| protected_seed42_100M | 66.760 | 59.880 | 22.620 | 52.190 | 52.190 | 35.635 | 7.620 | 68.022 | 45.615 |

## Delta: 80M - 100M

| column | delta |
|---|---:|
| blimp | -0.480 |
| supplement | -0.390 |
| entity_tracking | +0.420 |
| ewok | -0.700 |
| comps | -0.110 |
| GlobalPIQA_mean | +1.470 |
| Reading_mean | -0.030 |
| superglue | +0.287 |
| endpoint_8col_mean | +0.058 |

## SuperGLUE prediction concentration

### wwm_seed43_chck80M

| task | accuracy | majority_acc | pred_counts | all_one_label |
|---|---:|---:|---|---|
| boolq | 69.480 | 64.037 | {1: 1230, 0: 405} | False |
| multirc | 66.914 | 57.550 | {0: 1637, 1: 787} | False |
| rte | 60.432 | 53.957 | {1: 53, 0: 86} | False |
| wsc | 67.308 | 61.538 | {0: 35, 1: 17} | False |
| mrpc | 80.882 | 68.137 | {1: 164, 0: 40} | False |
| qqp | 76.616 | 62.775 | {1: 8580, 0: 11635} | False |
| mnli | 56.153 | 35.697 | {2: 1533, 1: 1696, 0: 1679} | False |

### wwm_seed43_chck100M

| task | accuracy | majority_acc | pred_counts | all_one_label |
|---|---:|---:|---|---|
| boolq | 68.746 | 64.037 | {1: 1240, 0: 395} | False |
| multirc | 67.120 | 57.550 | {0: 1488, 1: 936} | False |
| rte | 61.151 | 53.957 | {1: 56, 0: 83} | False |
| wsc | 63.462 | 61.538 | {0: 45, 1: 7} | False |
| mrpc | 82.353 | 68.137 | {1: 145, 0: 59} | False |
| qqp | 77.037 | 62.775 | {1: 8237, 0: 11978} | False |
| mnli | 55.909 | 35.697 | {2: 1528, 1: 2066, 0: 1314} | False |

