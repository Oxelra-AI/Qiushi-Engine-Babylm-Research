# compact mixture model and predictions REPEAT_SPLIT pair-level and strict-word robustness

This CPU-only check repeats the pair level relation robustness robustness logic on the relation practice principle REPEAT_SPLIT scored rows. It compares RS with CLEAN and original REPEAT in seed43022 over 80M/90M/100M.

## Pair-level tokenizer-nonoverlap distributions

| contrast | n pairs | mean gain Δ | median gain Δ | mean true Δ | mean unrel Δ | mean excess true cost | median excess | frac excess>0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RSminusC | 1465 | -0.0470 | -0.0165 | -0.5285 | -0.5756 | +0.0470 | +0.0165 | 0.505 |
| RSminusR | 1465 | +0.7381 | +0.5940 | -1.0123 | -0.2741 | -0.7381 | -0.5940 | 0.346 |
| RminusC | 1465 | -0.7852 | -0.6067 | +0.4837 | -0.3014 | +0.7852 | +0.6067 | 0.661 |
| VminusC | 1465 | +0.6711 | +0.6679 | -1.1550 | -0.4840 | -0.6711 | -0.6679 | 0.346 |
| VminusR | 1465 | +1.4562 | +1.1990 | -1.6388 | -0.1826 | -1.4562 | -1.1990 | 0.227 |

## Strict word-level nonoverlap

| contrast | n tokens min | gain Δ | true Δ | unrel Δ | excess true cost | frac tokens excess>0 |
|---|---:|---:|---:|---:|---:|---:|
| RSminusC | 2184 | -0.0964 | -0.5575 | -0.6539 | +0.0964 | 0.516 |
| RSminusR | 2184 | +0.4919 | -0.8606 | -0.3687 | -0.4919 | 0.425 |
| RminusC | 2184 | -0.5883 | +0.3031 | -0.2852 | +0.5883 | 0.601 |
| VminusC | 2184 | +0.5400 | -1.0679 | -0.5279 | -0.5400 | 0.404 |
| VminusR | 2184 | +1.1283 | -1.3710 | -0.2427 | -1.1283 | 0.333 |

Reading: RS−C remains near zero on tokenizer-nonoverlap gain at pair level and its excess true-source cost is near zero rather than original-REPEAT-like. RS−R is strongly positive in gain because RS removes the true-source cost while retaining lower unrelated-source NLL. The strict word filter preserves this reading: removing in-window exact recurrence eliminates the excess true-source cost rather than merely moving it into lexical-overlap artifacts.

Data outputs: `experiments/archive/relation_learning/data/repeat_split_robustness`.
