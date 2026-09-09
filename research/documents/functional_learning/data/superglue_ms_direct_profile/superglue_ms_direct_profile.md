# clean replication and ms direct pending direct SuperGLUE profile: clean preservation vs exact `(M,S)` acquisition

This parses actual repaired-AutoModel SuperGLUE prediction files and valid labels. It uses the exact acquisition-only dense-mask/sparse-label `(M,S)` seed62064 payload now completed in zero reading and superglue item evidence, plus coherent86, `(M,M)` dense controls, and clean preservation seed62064. It does not use pending clean seed62065 SuperGLUE or pending `(M,S)` zero-shot/Reading.

## SuperGLUE means

- coherent86: 68.945712
- dense_seed62064_MM: 68.531820
- dense_seed62065_MM: 68.805804
- ms_acquisition_seed62064_MS: 68.887842
- clean_pres_seed62064_MSplusKL: 69.047711
- exact `(M,S)` minus coherent86: -0.057870
- clean preservation minus exact `(M,S)`: 0.159869
- clean preservation minus coherent86: 0.101999
- exact `(M,S)` minus dense `(M,M)` seed62064/62065: 0.356022 / 0.082037

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

## Task primary-score deltas

| task | metric | exact `(M,S)` - parent | clean - parent | clean - exact `(M,S)` | exact `(M,S)` - dense64 `(M,M)` | exact `(M,S)` - dense65 `(M,M)` |
|---|---|---:|---:|---:|---:|---:|
| boolq | accuracy | 0.6728 | 0.6728 | 0.0000 | 0.0000 | 0.0612 |
| multirc | accuracy | -0.3713 | -0.1238 | 0.2475 | 0.3300 | 0.1650 |
| rte | accuracy | -0.7194 | -0.7194 | 0.0000 | 0.0000 | 0.0000 |
| wsc | accuracy | 0.0000 | 0.0000 | 0.0000 | 1.9231 | 0.0000 |
| mrpc | f1 | 0.3031 | 0.6897 | 0.3866 | 0.3031 | 0.3031 |
| qqp | f1 | -0.1272 | 0.0317 | 0.1590 | 0.0378 | 0.0858 |
| mnli | accuracy | -0.1630 | 0.1630 | 0.3260 | -0.1019 | -0.0407 |

## Clean preservation versus exact `(M,S)` item comparisons

| task | pred agreement | correctness agreement | clean correct / `(M,S)` wrong | `(M,S)` correct / clean wrong | net clean-`(M,S)` accuracy pp |
|---|---:|---:|---:|---:|---:|
| boolq | 0.9768 | 0.9768 | 19 | 19 | 0.0000 |
| multirc | 0.9340 | 0.9340 | 83 | 77 | 0.2475 |
| rte | 1.0000 | 1.0000 | 0 | 0 | 0.0000 |
| wsc | 1.0000 | 1.0000 | 0 | 0 | 0.0000 |
| mrpc | 0.9951 | 0.9951 | 1 | 0 | 0.4902 |
| qqp | 0.9535 | 0.9535 | 447 | 492 | -0.2226 |
| mnli | 0.8144 | 0.8549 | 364 | 348 | 0.3260 |

## Parent / exact `(M,S)` / clean decomposition

| task | clean recovers `(M,S)` parent loss | clean loses parent item `(M,S)` kept | clean keeps `(M,S)` gain | clean drops `(M,S)` gain | clean new gain beyond `(M,S)` | shared loss vs parent | net clean-`(M,S)` acc pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| boolq | 14 | 7 | 183 | 12 | 5 | 170 | 0.0000 |
| multirc | 48 | 38 | 96 | 39 | 35 | 96 | 0.2475 |
| rte | 0 | 0 | 2 | 0 | 0 | 3 | 0.0000 |
| wsc | 0 | 0 | 1 | 0 | 0 | 1 | 0.0000 |
| mrpc | 1 | 0 | 2 | 0 | 0 | 0 | 0.4902 |
| qqp | 398 | 50 | 90 | 442 | 49 | 86 | -0.2226 |
| mnli | 338 | 29 | 37 | 319 | 26 | 26 | 0.3260 |

## Scientific reading

Exact `(M,S)` acquisition-only SuperGLUE is stronger than the fully evaluated dense `(M,M)` controls, so dense `(M,M)` understated the supervised-transfer coordinate of the acquisition policy used by clean preservation. Clean preservation still improves over exact `(M,S)` by a modest +0.159869 SuperGLUE points. The task table determines whether this increment is distributed and whether it comes from real item changes or only metric bookkeeping. The direct Overall comparison remains unresolved until the running `(M,S)` official zero-shot/Reading job completes and the guarded same-coordinate table admits all components.
