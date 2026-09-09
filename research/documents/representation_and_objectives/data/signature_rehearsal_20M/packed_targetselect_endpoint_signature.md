# earlier analysis packed target-selective endpoint signature

JSON: `experiments/archive/representation_and_objectives/data/signature_rehearsal_20M/packed_targetselect_endpoint_signature.json`

## Cheap7-style scores

| arm | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | equal7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 60.190 | 57.900 | 48.020 | 17.940 | 50.340 | 33.225 | 8.040 | 39.379 |
| drop_abs | 60.700 | 58.500 | 50.810 | 18.000 | 50.510 | 34.240 | 8.015 | 40.111 |
| drop_copied_word | 60.100 | 57.200 | 49.990 | 18.830 | 50.560 | 30.765 | 7.750 | 39.314 |

## Deltas

| contrast | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | equal7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| drop_abs_minus_full | +0.510 | +0.600 | +2.790 | +0.060 | +0.170 | +1.015 | -0.025 | +0.731 |
| drop_copied_word_minus_full | -0.090 | -0.700 | +1.970 | +0.890 | +0.220 | -2.460 | -0.290 | -0.066 |
| drop_abs_minus_drop_copied_word | +0.600 | +1.300 | +0.820 | -0.830 | -0.050 | +3.475 | +0.265 | +0.797 |

## EWoK predeclared groups

### relational_domains
| arm | n | acc |
|---|---:|---:|
| full | 1756 | 45.900 |
| drop_abs | 1756 | 48.462 |
| drop_copied_word | 1756 | 47.836 |
Deltas: {"drop_abs_minus_full": 2.562643, "drop_copied_word_minus_full": 1.936219, "drop_abs_minus_drop_copied_word": 0.626424}

### adjacency_independent_domains
| arm | n | acc |
|---|---:|---:|
| full | 464 | 51.940 |
| drop_abs | 464 | 52.155 |
| drop_copied_word | 464 | 50.862 |
Deltas: {"drop_abs_minus_full": 0.215517, "drop_copied_word_minus_full": -1.077586, "drop_abs_minus_drop_copied_word": 1.293103}

