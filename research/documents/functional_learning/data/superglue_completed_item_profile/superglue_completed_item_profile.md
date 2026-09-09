# zero reading and superglue item evidence completed SuperGLUE item/task profile

This uses only completed repaired AutoModel SuperGLUE payloads: coherent86, dense seed62064, dense seed62065, and clean preservation seed62064. It does not use the running clean seed62065 or exact `(M,S)` jobs.

## SuperGLUE means

- coherent86: 68.945712
- dense_seed62064: 68.531820
- dense_seed62065: 68.805804
- clean_seed62064: 69.047711
- clean64 minus coherent86: 0.101999
- clean64 minus dense64: 0.515891
- clean64 minus dense65: 0.241907

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
| dense_seed62064 | boolq | accuracy | 1635 | 67.951070 | 67.951070 | 0.00000000 | 67.9511 | 76.1601 | {1: 1151, 0: 484} |
| dense_seed62064 | multirc | accuracy | 2424 | 67.574257 | 67.574257 | 0.00000000 | 67.5743 | 58.3245 | {1: 857, 0: 1567} |
| dense_seed62064 | rte | accuracy | 139 | 63.309353 | 63.309353 | 0.00000000 | 63.3094 | 59.2000 | {1: 61, 0: 78} |
| dense_seed62064 | wsc | accuracy | 52 | 61.538462 | 61.538462 | 0.00000000 | 61.5385 | 0.0000 | {0: 52} |
| dense_seed62064 | mrpc | f1 | 204 | 87.586207 | 87.586207 | -0.00000000 | 82.3529 | 87.5862 | {1: 151, 0: 53} |
| dense_seed62064 | qqp | f1 | 20215 | 71.392569 | 71.392569 | 0.00000000 | 78.1004 | 71.3926 | {1: 7950, 0: 12265} |
| dense_seed62064 | mnli | accuracy | 4908 | 60.370823 | 60.370823 | -0.00000000 | 60.3708 | None | {2: 1591, 1: 1937, 0: 1380} |
| dense_seed62065 | boolq | accuracy | 1635 | 67.889908 | 67.889908 | 0.00000000 | 67.8899 | 76.1038 | {1: 1150, 0: 485} |
| dense_seed62065 | multirc | accuracy | 2424 | 67.739274 | 67.739274 | 0.00000000 | 67.7393 | 56.9383 | {0: 1637, 1: 787} |
| dense_seed62065 | rte | accuracy | 139 | 63.309353 | 63.309353 | 0.00000000 | 63.3094 | 60.4651 | {1: 65, 0: 74} |
| dense_seed62065 | wsc | accuracy | 52 | 63.461538 | 63.461538 | 0.00000000 | 63.4615 | 38.7097 | {0: 41, 1: 11} |
| dense_seed62065 | mrpc | f1 | 204 | 87.586207 | 87.586207 | -0.00000000 | 82.3529 | 87.5862 | {1: 151, 0: 53} |
| dense_seed62065 | qqp | f1 | 20215 | 71.344652 | 71.344652 | 0.00000000 | 78.0411 | 71.3447 | {1: 7966, 0: 12249} |
| dense_seed62065 | mnli | accuracy | 4908 | 60.309698 | 60.309698 | 0.00000000 | 60.3097 | None | {2: 1589, 1: 1938, 0: 1381} |
| clean_seed62064 | boolq | accuracy | 1635 | 67.951070 | 67.951070 | 0.00000000 | 67.9511 | 76.1167 | {1: 1147, 0: 488} |
| clean_seed62064 | multirc | accuracy | 2424 | 68.151815 | 68.151815 | 0.00000000 | 68.1518 | 59.2827 | {1: 867, 0: 1557} |
| clean_seed62064 | rte | accuracy | 139 | 63.309353 | 63.309353 | 0.00000000 | 63.3094 | 59.2000 | {1: 61, 0: 78} |
| clean_seed62064 | wsc | accuracy | 52 | 63.461538 | 63.461538 | 0.00000000 | 63.4615 | 42.4242 | {0: 39, 1: 13} |
| clean_seed62064 | mrpc | f1 | 204 | 88.275862 | 88.275862 | 0.00000000 | 83.3333 | 88.2759 | {1: 151, 0: 53} |
| clean_seed62064 | qqp | f1 | 20215 | 71.589391 | 71.589391 | 0.00000000 | 77.9025 | 71.5894 | {1: 8198, 0: 12017} |
| clean_seed62064 | mnli | accuracy | 4908 | 60.594947 | 60.594947 | 0.00000000 | 60.5949 | None | {2: 1766, 1: 1665, 0: 1477} |

## Task primary-score deltas

| task | metric | dense64-parent | dense65-parent | clean64-parent | clean64-dense64 | clean64-dense65 |
|---|---|---:|---:|---:|---:|---:|
| boolq | accuracy | 0.6728 | 0.6116 | 0.6728 | 0.0000 | 0.0612 |
| multirc | accuracy | -0.7013 | -0.5363 | -0.1238 | 0.5776 | 0.4125 |
| rte | accuracy | -0.7194 | -0.7194 | -0.7194 | 0.0000 | 0.0000 |
| wsc | accuracy | -1.9231 | 0.0000 | 0.0000 | 1.9231 | 0.0000 |
| mrpc | f1 | 0.0000 | 0.0000 | 0.6897 | 0.6897 | 0.6897 |
| qqp | f1 | -0.1651 | -0.2130 | 0.0317 | 0.1968 | 0.2447 |
| mnli | accuracy | -0.0611 | -0.1222 | 0.1630 | 0.2241 | 0.2852 |

## Clean64 item comparisons

| task | clean-vs-parent pred agree | clean-vs-parent correctness agree | clean correct parent wrong | parent correct clean wrong | clean-vs-dense64 pred agree | clean correct dense64 wrong | dense64 correct clean wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| boolq | 0.7768 | 0.7768 | 188 | 177 | 0.9780 | 18 | 18 |
| multirc | 0.8907 | 0.8907 | 131 | 134 | 0.9340 | 87 | 73 |
| rte | 0.9640 | 0.9640 | 2 | 3 | 1.0000 | 0 | 0 |
| wsc | 0.9615 | 0.9615 | 1 | 1 | 0.7500 | 7 | 6 |
| mrpc | 0.9902 | 0.9902 | 2 | 0 | 0.9902 | 2 | 0 |
| qqp | 0.9864 | 0.9864 | 139 | 136 | 0.9533 | 452 | 492 |
| mnli | 0.9711 | 0.9760 | 63 | 55 | 0.8148 | 361 | 350 |

## Dense-to-clean decompositions against parent

| task | dense seed | recover dense parent loss | clean loses dense-kept parent item | keep dense gain | drop dense gain | new clean gain | shared loss | net clean-dense acc pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| boolq | 62064 | 12 | 7 | 182 | 11 | 6 | 170 | 0.0000 |
| boolq | 62065 | 12 | 7 | 182 | 10 | 6 | 170 | 0.0612 |
| multirc | 62064 | 54 | 37 | 98 | 36 | 33 | 97 | 0.5776 |
| multirc | 62065 | 47 | 101 | 37 | 30 | 94 | 33 | 0.4125 |
| rte | 62064 | 0 | 0 | 2 | 0 | 0 | 3 | 0.0000 |
| rte | 62065 | 2 | 2 | 1 | 1 | 1 | 1 | 0.0000 |
| wsc | 62064 | 7 | 0 | 1 | 6 | 0 | 1 | 1.9231 |
| wsc | 62065 | 1 | 0 | 1 | 1 | 0 | 1 | 0.0000 |
| mrpc | 62064 | 2 | 0 | 2 | 0 | 0 | 0 | 0.9804 |
| mrpc | 62065 | 2 | 0 | 2 | 0 | 0 | 0 | 0.9804 |
| qqp | 62064 | 404 | 50 | 91 | 442 | 48 | 86 | -0.1979 |
| qqp | 62065 | 135 | 26 | 119 | 157 | 20 | 110 | -0.1385 |
| mnli | 62064 | 334 | 29 | 36 | 321 | 27 | 26 | 0.2241 |
| mnli | 62065 | 338 | 29 | 37 | 321 | 26 | 26 | 0.2852 |

## Scientific reading

Clean seed62064's SuperGLUE advantage over dense controls is not a single-task artifact: it improves over dense on MultiRC, WSC, MRPC F1, QQP F1, and MNLI, while RTE stays at the dense value and BoolQ is equal or slightly above dense. Relative to coherent86 the picture is smaller and mixed: BoolQ/MRPC/MNLI/QQP contribute gains, MultiRC/RTE lose, and WSC is equal. This is consistent with preservation bounding a dense-induced transfer cost rather than installing an entirely new supervised-transfer mechanism. The pending clean seed62065 and exact `(M,S)` payloads must be evaluated in the same way before this interpretation is used for v5 promotion.
