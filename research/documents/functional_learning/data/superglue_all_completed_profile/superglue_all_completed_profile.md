# clean replication and ms direct pending all completed SuperGLUE profile

This artifact uses only completed repaired-AutoModel SuperGLUE payloads and recomputes primary metrics from predictions and valid labels. It includes coherent86, dense `(M,M)` seeds, exact acquisition-only `(M,S)` seed62064, and clean preservation seeds 62064/62065.

## SuperGLUE means

- coherent86: 68.945712
- dense_seed62064_MM: 68.531820
- dense_seed62065_MM: 68.805804
- ms_acquisition_seed62064_MS: 68.887842
- clean_pres_seed62064_MSplusKL: 69.047711
- clean_pres_seed62065_MSplusKL: 69.020580
- clean64 minus coherent86: 0.101999
- clean65 minus coherent86: 0.074868
- clean65 minus clean64: -0.027131
- exact `(M,S)` minus coherent86: -0.057870
- clean64 / clean65 minus exact `(M,S)`: 0.159869 / 0.132739

## Payload metric reproduction

| model | task | metric | n | payload primary | computed primary | payload-computed | accuracy | f1 | pred counts |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| coherent86 | boolq | accuracy | 1635 | 67.278287 | 67.278287 | 0.00000000 | 67.2783 | 77.3975 | {1: 1320, 0: 315} |
| coherent86 | multirc | accuracy | 2424 | 68.275578 | 68.275578 | -0.00000000 | 68.2756 | 58.8550 | {0: 1584, 1: 840} |
| coherent86 | rte | accuracy | 139 | 64.028777 | 64.028777 | 0.00000000 | 64.0288 | 61.5385 | {1: 66, 0: 73} |
| coherent86 | wsc | accuracy | 52 | 63.461538 | 63.461538 | 0.00000000 | 63.4615 | 45.7143 | {0: 37, 1: 15} |
| coherent86 | mrpc | f1 | 204 | 87.586207 | 87.586207 | -0.00000000 | 82.3529 | 87.5862 | {1: 151, 0: 53} |
| coherent86 | qqp | f1 | 20215 | 71.557648 | 71.557648 | 0.00000000 | 77.8877 | 71.5576 | {1: 8191, 0: 12024} |
| coherent86 | mnli | accuracy | 4908 | 60.431948 | 60.431948 | 0.00000000 | 60.4319 | None | {2: 1739, 1: 1677, 0: 1492} |
| dense_seed62064_MM | boolq | accuracy | 1635 | 67.951070 | 67.951070 | 0.00000000 | 67.9511 | 76.1601 | {1: 1151, 0: 484} |
| dense_seed62064_MM | multirc | accuracy | 2424 | 67.574257 | 67.574257 | 0.00000000 | 67.5743 | 58.3245 | {1: 857, 0: 1567} |
| dense_seed62064_MM | rte | accuracy | 139 | 63.309353 | 63.309353 | 0.00000000 | 63.3094 | 59.2000 | {1: 61, 0: 78} |
| dense_seed62064_MM | wsc | accuracy | 52 | 61.538462 | 61.538462 | 0.00000000 | 61.5385 | 0.0000 | {0: 52} |
| dense_seed62064_MM | mrpc | f1 | 204 | 87.586207 | 87.586207 | -0.00000000 | 82.3529 | 87.5862 | {1: 151, 0: 53} |
| dense_seed62064_MM | qqp | f1 | 20215 | 71.392569 | 71.392569 | 0.00000000 | 78.1004 | 71.3926 | {1: 7950, 0: 12265} |
| dense_seed62064_MM | mnli | accuracy | 4908 | 60.370823 | 60.370823 | -0.00000000 | 60.3708 | None | {2: 1591, 1: 1937, 0: 1380} |
| dense_seed62065_MM | boolq | accuracy | 1635 | 67.889908 | 67.889908 | 0.00000000 | 67.8899 | 76.1038 | {1: 1150, 0: 485} |
| dense_seed62065_MM | multirc | accuracy | 2424 | 67.739274 | 67.739274 | 0.00000000 | 67.7393 | 56.9383 | {0: 1637, 1: 787} |
| dense_seed62065_MM | rte | accuracy | 139 | 63.309353 | 63.309353 | 0.00000000 | 63.3094 | 60.4651 | {1: 65, 0: 74} |
| dense_seed62065_MM | wsc | accuracy | 52 | 63.461538 | 63.461538 | 0.00000000 | 63.4615 | 38.7097 | {0: 41, 1: 11} |
| dense_seed62065_MM | mrpc | f1 | 204 | 87.586207 | 87.586207 | -0.00000000 | 82.3529 | 87.5862 | {1: 151, 0: 53} |
| dense_seed62065_MM | qqp | f1 | 20215 | 71.344652 | 71.344652 | 0.00000000 | 78.0411 | 71.3447 | {1: 7966, 0: 12249} |
| dense_seed62065_MM | mnli | accuracy | 4908 | 60.309698 | 60.309698 | 0.00000000 | 60.3097 | None | {2: 1589, 1: 1938, 0: 1381} |
| ms_acquisition_seed62064_MS | boolq | accuracy | 1635 | 67.951070 | 67.951070 | 0.00000000 | 67.9511 | 76.1167 | {1: 1147, 0: 488} |
| ms_acquisition_seed62064_MS | multirc | accuracy | 2424 | 67.904290 | 67.904290 | 0.00000000 | 67.9043 | 59.7308 | {0: 1521, 1: 903} |
| ms_acquisition_seed62064_MS | rte | accuracy | 139 | 63.309353 | 63.309353 | 0.00000000 | 63.3094 | 59.2000 | {1: 61, 0: 78} |
| ms_acquisition_seed62064_MS | wsc | accuracy | 52 | 63.461538 | 63.461538 | 0.00000000 | 63.4615 | 42.4242 | {0: 39, 1: 13} |
| ms_acquisition_seed62064_MS | mrpc | f1 | 204 | 87.889273 | 87.889273 | 0.00000000 | 82.8431 | 87.8893 | {1: 150, 0: 54} |
| ms_acquisition_seed62064_MS | qqp | f1 | 20215 | 71.430417 | 71.430417 | 0.00000000 | 78.1252 | 71.4304 | {1: 7953, 0: 12262} |
| ms_acquisition_seed62064_MS | mnli | accuracy | 4908 | 60.268949 | 60.268949 | 0.00000000 | 60.2689 | None | {2: 1587, 1: 1939, 0: 1382} |
| clean_pres_seed62064_MSplusKL | boolq | accuracy | 1635 | 67.951070 | 67.951070 | 0.00000000 | 67.9511 | 76.1167 | {1: 1147, 0: 488} |
| clean_pres_seed62064_MSplusKL | multirc | accuracy | 2424 | 68.151815 | 68.151815 | 0.00000000 | 68.1518 | 59.2827 | {1: 867, 0: 1557} |
| clean_pres_seed62064_MSplusKL | rte | accuracy | 139 | 63.309353 | 63.309353 | 0.00000000 | 63.3094 | 59.2000 | {1: 61, 0: 78} |
| clean_pres_seed62064_MSplusKL | wsc | accuracy | 52 | 63.461538 | 63.461538 | 0.00000000 | 63.4615 | 42.4242 | {0: 39, 1: 13} |
| clean_pres_seed62064_MSplusKL | mrpc | f1 | 204 | 88.275862 | 88.275862 | 0.00000000 | 83.3333 | 88.2759 | {1: 151, 0: 53} |
| clean_pres_seed62064_MSplusKL | qqp | f1 | 20215 | 71.589391 | 71.589391 | 0.00000000 | 77.9025 | 71.5894 | {1: 8198, 0: 12017} |
| clean_pres_seed62064_MSplusKL | mnli | accuracy | 4908 | 60.594947 | 60.594947 | 0.00000000 | 60.5949 | None | {2: 1766, 1: 1665, 0: 1477} |
| clean_pres_seed62065_MSplusKL | boolq | accuracy | 1635 | 67.767584 | 67.767584 | 0.00000000 | 67.7676 | 76.0127 | {1: 1150, 0: 485} |
| clean_pres_seed62065_MSplusKL | multirc | accuracy | 2424 | 68.193069 | 68.193069 | 0.00000000 | 68.1931 | 59.2279 | {1: 862, 0: 1562} |
| clean_pres_seed62065_MSplusKL | rte | accuracy | 139 | 63.309353 | 63.309353 | 0.00000000 | 63.3094 | 60.4651 | {1: 65, 0: 74} |
| clean_pres_seed62065_MSplusKL | wsc | accuracy | 52 | 63.461538 | 63.461538 | 0.00000000 | 63.4615 | 42.4242 | {0: 39, 1: 13} |
| clean_pres_seed62065_MSplusKL | mrpc | f1 | 204 | 88.275862 | 88.275862 | 0.00000000 | 83.3333 | 88.2759 | {1: 151, 0: 53} |
| clean_pres_seed62065_MSplusKL | qqp | f1 | 20215 | 71.562083 | 71.562083 | 0.00000000 | 77.8729 | 71.5621 | {1: 8204, 0: 12011} |
| clean_pres_seed62065_MSplusKL | mnli | accuracy | 4908 | 60.574572 | 60.574572 | 0.00000000 | 60.5746 | None | {2: 1761, 1: 1670, 0: 1477} |

## Task primary-score deltas

| task | metric | clean64-parent | clean65-parent | clean65-clean64 | exact `(M,S)`-parent | clean64-`(M,S)` | clean65-`(M,S)` |
|---|---|---:|---:|---:|---:|---:|---:|
| boolq | accuracy | 0.6728 | 0.4893 | -0.1835 | 0.6728 | 0.0000 | -0.1835 |
| multirc | accuracy | -0.1238 | -0.0825 | 0.0413 | -0.3713 | 0.2475 | 0.2888 |
| rte | accuracy | -0.7194 | -0.7194 | -0.0000 | -0.7194 | 0.0000 | 0.0000 |
| wsc | accuracy | 0.0000 | 0.0000 | -0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mrpc | f1 | 0.6897 | 0.6897 | -0.0000 | 0.3031 | 0.3866 | 0.3866 |
| qqp | f1 | 0.0317 | 0.0044 | -0.0273 | -0.1272 | 0.1590 | 0.1317 |
| mnli | accuracy | 0.1630 | 0.1426 | -0.0204 | -0.1630 | 0.3260 | 0.3056 |

## Clean seed stability and clean-vs-`(M,S)` item comparisons

| task | clean seed pred agree | clean seed correctness agree | clean65 correct/clean64 wrong | clean64 correct/clean65 wrong | clean64-`(M,S)` net pp | clean65-`(M,S)` net pp |
|---|---:|---:|---:|---:|---:|---:|
| boolq | 0.9982 | 0.9982 | 0 | 3 | 0.0000 | -0.1835 |
| multirc | 0.9402 | 0.9402 | 73 | 72 | 0.2475 | 0.2888 |
| rte | 0.9712 | 0.9712 | 2 | 2 | 0.0000 | 0.0000 |
| wsc | 1.0000 | 1.0000 | 0 | 0 | 0.0000 | 0.0000 |
| mrpc | 1.0000 | 1.0000 | 0 | 0 | 0.4902 | 0.4902 |
| qqp | 0.9983 | 0.9983 | 14 | 20 | -0.2226 | -0.2523 |
| mnli | 0.9980 | 0.9982 | 4 | 5 | 0.3260 | 0.3056 |

## Scientific reading

The clean SuperGLUE result replicates tightly across seeds: clean seed62065 is only 0.02713 points below clean seed62064 and remains above coherent86 by 0.07487. Exact `(M,S)` acquisition-only is below coherent86 on SuperGLUE by 0.05787, so clean preservation adds a repeatable supervised-transfer recovery beyond `(M,S)`: +0.15987 for seed62064 and +0.13274 for seed62065. The remaining direct method comparison is not yet complete because `(M,S)` official zero-shot/Reading is still running; the direct Overall difference will combine these SuperGLUE increments with any zero-shot/Reading component differences.
