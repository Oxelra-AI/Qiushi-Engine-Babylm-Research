# compact mixture model and predictions component-level copy and target-class checks

These are existing-data checks prompted by independent_review. They prevent two overstatements: treating raw copy gain as source-present copy ability, and treating the mixture model as established without the clean overlap/nonoverlap target interaction.

## Natural-copy components across DeBERTa seeds

Copy gain = no-source/control NLL minus source-present repeated NLL. Source-present NLL and control NLL are reported separately.

| seed | role | raw gain | normalized gain | source-present NLL | no-source/control NLL |
|---:|---|---:|---:|---:|---:|
| 43022 | C | +3.6869 | +0.9040 | +0.3913 | +4.0782 |
| 43022 | R | +4.1864 | +0.9537 | +0.2032 | +4.3896 |
| 43022 | RS | +3.4863 | +0.8734 | +0.5054 | +3.9918 |
| 43022 | V | +4.0140 | +0.9131 | +0.3820 | +4.3960 |
| 43122 | C | +3.6208 | +0.8943 | +0.4282 | +4.0489 |
| 43122 | R | +4.2886 | +0.9474 | +0.2383 | +4.5270 |
| 43122 | V | +3.9311 | +0.9062 | +0.4071 | +4.3382 |
| 43222 | C | +3.6063 | +0.8909 | +0.4417 | +4.0480 |
| 43222 | R | +4.1657 | +0.9512 | +0.2138 | +4.3795 |
| 43222 | V | +3.9210 | +0.9060 | +0.4068 | +4.3278 |

## Copy component contrasts

| seed | contrast | raw gain Δ | norm gain Δ | source-present NLL Δ | control NLL Δ |
|---:|---|---:|---:|---:|---:|
| 43022 | RSminusC | -0.2005 | -0.0307 | +0.1141 | -0.0864 |
| 43022 | RSminusR | -0.7001 | -0.0804 | +0.3022 | -0.3979 |
| 43022 | RminusC | +0.4996 | +0.0497 | -0.1881 | +0.3115 |
| 43022 | VminusC | +0.3271 | +0.0091 | -0.0093 | +0.3178 |
| 43022 | VminusR | -0.1725 | -0.0406 | +0.1788 | +0.0064 |
| 43122 | RminusC | +0.6678 | +0.0531 | -0.1898 | +0.4780 |
| 43122 | VminusC | +0.3103 | +0.0119 | -0.0211 | +0.2892 |
| 43122 | VminusR | -0.3575 | -0.0412 | +0.1687 | -0.1888 |
| 43222 | RminusC | +0.5594 | +0.0603 | -0.2278 | +0.3315 |
| 43222 | VminusC | +0.3147 | +0.0151 | -0.0349 | +0.2798 |
| 43222 | VminusR | -0.2447 | -0.0452 | +0.1930 | -0.0517 |

Reading: REPEAT's raw copy advantage over CLEAN is supported by source-present NLL improvement in all three DeBERTa seeds. VIEW's raw copy advantage over CLEAN is much smaller and is not a clean source-present copy improvement: its source-present NLL is only slightly better than CLEAN while the no-source control is also worse. Therefore VIEW's positive copy statement should be weak and component-qualified rather than treated as a resolved copy competence.

## Compact-rewrite target-class interaction

| seed | contrast | overlap gain Δ | nonoverlap gain Δ | overlap−nonoverlap gain Δ | overlap true Δ | nonoverlap true Δ |
|---:|---|---:|---:|---:|---:|---:|
| 43022 | RSminusC | -0.7817 | -0.0470 | -0.7347 | +0.0463 | -0.5285 |
| 43022 | RSminusR | -0.5175 | +0.7381 | -1.2556 | +0.1449 | -1.0123 |
| 43022 | RminusC | -0.2642 | -0.7852 | +0.5209 | -0.0986 | +0.4837 |
| 43022 | VminusC | +0.5703 | +0.6711 | -0.1007 | -0.9521 | -1.1550 |
| 43022 | VminusR | +0.8346 | +1.4562 | -0.6216 | -0.8534 | -1.6388 |
| 43122 | RminusC | -0.2863 | -1.0550 | +0.7688 | -0.0717 | +0.7088 |
| 43122 | VminusC | +0.6643 | +0.8891 | -0.2248 | -0.9900 | -1.3517 |
| 43122 | VminusR | +0.9505 | +1.9441 | -0.9936 | -0.9183 | -2.0605 |
| 43222 | RminusC | -0.7105 | -0.8493 | +0.1387 | +0.2039 | +0.4226 |
| 43222 | VminusC | +0.4199 | +0.8485 | -0.4287 | -0.8737 | -1.3771 |
| 43222 | VminusR | +1.1304 | +1.6978 | -0.5674 | -1.0776 | -1.7997 |

Reading: The clean within-probe interaction is strong for original R−C in DeBERTa seeds: REPEAT is much less harmful on targets whose token appears in the source than on nonoverlap targets, and in the first two seeds its true-source NLL is slightly better than CLEAN on overlap targets while becoming worse on nonoverlap targets. This directly supports the identity-readout interpretation. V−C is positive on both token classes and is larger on nonoverlap targets in all three seeds, so VIEW is not merely improving copied overlap tokens; it improves content-conditioned use where source tokens do not contain the answer. RS−C lacks the original nonoverlap sign reversal; on nonoverlap it improves both T and U, while on overlap it is near CLEAN on T but better on U, consistent with cross-row content exposure without the local identity readout installed by original REPEAT.

Data outputs: `experiments/archive/relation_learning/data/component_targetclass_checks`.
