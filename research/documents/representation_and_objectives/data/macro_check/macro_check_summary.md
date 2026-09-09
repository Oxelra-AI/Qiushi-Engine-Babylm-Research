# Macro Quantity Recalculation

This CPU-only file read recomputes the macro quantities from the CSV results.

## Entity late means

| contrast | n | official | zero-op | nonzero | balanced zero/nonzero | spread |
|---|---:|---:|---:|---:|---:|---:|
| deberta_basin1 | 3 | +4.018 | -9.661 | +6.753 | -1.454 | +16.414 |
| deberta_basin2 | 3 | +2.539 | -9.115 | +4.870 | -2.123 | +13.985 |
| deberta_breadth_minus_repeat | 3 | +1.023 | -13.037 | +3.836 | -4.601 | +16.873 |
| deberta_view_minus_breadth | 3 | +2.994 | +3.376 | +2.918 | +3.147 | -0.459 |
| roberta_max | 3 | +0.259 | -0.046 | +0.320 | +0.137 | +0.366 |

## Visible late V-B rows by family

| family | rows | deltas |
|---|---:|---|
| BLiMP | 2 | chck_80M -0.870, chck_90M -0.610 |
| COMPS | 1 | chck_80M -1.180 |
| EWoK | 2 | chck_80M -0.150, chck_90M +1.260 |
| EWoK_plus_Entity_sum | 2 | chck_80M +1.475, chck_90M +2.170 |
| Entity | 3 | chck_80M +3.100, chck_90M +3.080, chck_100M +2.800 |
| Reading | 1 | chck_80M +0.070 |
| Supplement | 2 | chck_80M +1.820, chck_90M +1.190 |
| cheap5_no_GlobalPIQA_Reading | 1 | chck_80M +0.544 |
| cheap6_no_GlobalPIQA | 1 | chck_80M +0.465 |
| exEntity4_noReading | 1 | chck_80M -0.095 |
| exEntity5 | 1 | chck_80M -0.062 |

## Missing late V-B rows

| family | missing rows |
|---|---:|
| BLiMP | 1 |
| COMPS | 2 |
| EWoK | 1 |
| EWoK_plus_Entity_sum | 1 |
| Reading | 2 |
| Supplement | 1 |
| cheap5_no_GlobalPIQA_Reading | 2 |
| cheap6_no_GlobalPIQA | 2 |
| exEntity4_noReading | 2 |
| exEntity5 | 2 |

## EWoK selected all-domain margin rows

| dose | window | n | margin mean | frac positive |
|---|---|---:|---:|---:|
| dose1 | common10_80 | 440 | +0.0599 | 0.505 |
| dose1 | endpoint80 | 55 | +0.3385 | 0.600 |
| dose1p82 | common10_80 | 440 | -0.2117 | 0.491 |
| dose1p82 | endpoint80 | 55 | +0.0111 | 0.455 |
| dose2p64 | common10_80 | 440 | +0.0090 | 0.511 |
| dose2p64 | endpoint80 | 55 | -0.2960 | 0.527 |

## Scientific reading

The recomputation matches the peer summaries: Entity V-B is balanced-positive, B-R and V-R are operation-skewed, and broad ex-Entity V-B is still too sparse and mixed to support a uniform compact companion law.
