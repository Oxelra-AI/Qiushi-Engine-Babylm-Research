# anchor corrected price interpretation CHILDES variation-set probe scoring

The primary surface-held-out CHILDES adjacent-utterance probe was scored on the original DeBERTa VIEW/CLEAN/REPEAT arms across seeds 43022, 43122, and 43222 at checkpoints 80M/90M/100M. Delta is NLL(replaced context) minus NLL(intact adjacent context), so positive values mean the natural neighbor helps the masked target.

Probe records: 4222; paired targets: 2111; pairs: 732; pair bins low/partial/high = {'low': 498, 'partial': 217, 'high': 17}. Exact surface exposure hits in the original three training pools were zero in the validator.

## Across-seed role means

| bin | target class | role | mean delta | seed SD | mean pairs |
|---|---|---|---:|---:|---:|
| ALL | ALL | C | +1.3467 | +0.0369 | 732.0 |
| ALL | ALL | R | +1.3634 | +0.0524 | 732.0 |
| ALL | ALL | V | +1.3575 | +0.0328 | 732.0 |
| ALL | nonoverlap | C | -0.0947 | +0.0163 | 727.0 |
| ALL | nonoverlap | R | -0.1149 | +0.0545 | 727.0 |
| ALL | nonoverlap | V | -0.0822 | +0.0396 | 727.0 |
| ALL | overlap | C | +3.0696 | +0.0685 | 663.0 |
| ALL | overlap | R | +3.1262 | +0.1528 | 663.0 |
| ALL | overlap | V | +3.0687 | +0.0976 | 663.0 |
| low | ALL | C | +1.2505 | +0.0326 | 498.0 |
| low | ALL | R | +1.2373 | +0.0457 | 498.0 |
| low | ALL | V | +1.2540 | +0.0418 | 498.0 |
| low | nonoverlap | C | -0.0162 | +0.0047 | 495.0 |
| low | nonoverlap | R | -0.0527 | +0.0545 | 495.0 |
| low | nonoverlap | V | -0.0092 | +0.0216 | 495.0 |
| low | overlap | C | +2.8567 | +0.0733 | 440.0 |
| low | overlap | R | +2.8671 | +0.1530 | 440.0 |
| low | overlap | V | +2.8418 | +0.1183 | 440.0 |
| partial | ALL | C | +1.5309 | +0.0693 | 217.0 |
| partial | ALL | R | +1.6143 | +0.0781 | 217.0 |
| partial | ALL | V | +1.5640 | +0.0689 | 217.0 |
| partial | nonoverlap | C | -0.2630 | +0.0552 | 216.0 |
| partial | nonoverlap | R | -0.2456 | +0.0614 | 216.0 |
| partial | nonoverlap | V | -0.2589 | +0.0945 | 216.0 |
| partial | overlap | C | +3.4801 | +0.0875 | 206.0 |
| partial | overlap | R | +3.6314 | +0.1720 | 206.0 |
| partial | overlap | V | +3.5425 | +0.1061 | 206.0 |
| high | ALL | C | +1.8125 | +0.0479 | 17.0 |
| high | ALL | R | +1.8557 | +0.0478 | 17.0 |
| high | ALL | V | +1.7547 | +0.0804 | 17.0 |
| high | nonoverlap | C | -0.2500 | +0.1252 | 16.0 |
| high | nonoverlap | R | -0.2757 | +0.1830 | 16.0 |
| high | nonoverlap | V | +0.0454 | +0.1342 | 16.0 |
| high | overlap | C | +3.6074 | +0.0497 | 17.0 |
| high | overlap | R | +3.7117 | +0.2899 | 17.0 |
| high | overlap | V | +3.2002 | +0.1747 | 17.0 |

## Across-seed contrasts

| bin | target class | contrast | mean delta difference | seed SD | mean pairs |
|---|---|---|---:|---:|---:|
| ALL | ALL | VminusC | +0.0108 | +0.0110 | 732.0 |
| ALL | ALL | RminusC | +0.0167 | +0.0354 | 732.0 |
| ALL | ALL | VminusR | -0.0059 | +0.0455 | 732.0 |
| ALL | nonoverlap | VminusC | +0.0124 | +0.0375 | 727.0 |
| ALL | nonoverlap | RminusC | -0.0203 | +0.0675 | 727.0 |
| ALL | nonoverlap | VminusR | +0.0327 | +0.0535 | 727.0 |
| ALL | overlap | VminusC | -0.0009 | +0.0501 | 663.0 |
| ALL | overlap | RminusC | +0.0566 | +0.1043 | 663.0 |
| ALL | overlap | VminusR | -0.0575 | +0.1406 | 663.0 |
| low | ALL | VminusC | +0.0035 | +0.0226 | 498.0 |
| low | ALL | RminusC | -0.0132 | +0.0366 | 498.0 |
| low | ALL | VminusR | +0.0167 | +0.0592 | 498.0 |
| low | nonoverlap | VminusC | +0.0070 | +0.0252 | 495.0 |
| low | nonoverlap | RminusC | -0.0365 | +0.0592 | 495.0 |
| low | nonoverlap | VminusR | +0.0435 | +0.0397 | 495.0 |
| low | overlap | VminusC | -0.0149 | +0.0622 | 440.0 |
| low | overlap | RminusC | +0.0104 | +0.1070 | 440.0 |
| low | overlap | VminusR | -0.0254 | +0.1506 | 440.0 |
| partial | ALL | VminusC | +0.0331 | +0.0102 | 217.0 |
| partial | ALL | RminusC | +0.0834 | +0.0424 | 217.0 |
| partial | ALL | VminusR | -0.0503 | +0.0322 | 217.0 |
| partial | nonoverlap | VminusC | +0.0040 | +0.0731 | 216.0 |
| partial | nonoverlap | RminusC | +0.0174 | +0.1000 | 216.0 |
| partial | nonoverlap | VminusR | -0.0134 | +0.0909 | 216.0 |
| partial | overlap | VminusC | +0.0625 | +0.0515 | 206.0 |
| partial | overlap | RminusC | +0.1513 | +0.0950 | 206.0 |
| partial | overlap | VminusR | -0.0888 | +0.1304 | 206.0 |
| high | ALL | VminusC | -0.0579 | +0.1264 | 17.0 |
| high | ALL | RminusC | +0.0431 | +0.0915 | 17.0 |
| high | ALL | VminusR | -0.1010 | +0.0358 | 17.0 |
| high | nonoverlap | VminusC | +0.2953 | +0.2027 | 16.0 |
| high | nonoverlap | RminusC | -0.0257 | +0.0871 | 16.0 |
| high | nonoverlap | VminusR | +0.3211 | +0.2027 | 16.0 |
| high | overlap | VminusC | -0.4072 | +0.2175 | 17.0 |
| high | overlap | RminusC | +0.1043 | +0.3096 | 17.0 |
| high | overlap | VminusR | -0.5115 | +0.1824 | 17.0 |

## Reading

This natural probe does not reproduce the compact-rewrite arm ordering. Its strongest signal is simpler: the adjacent utterance helps targets that reoccur from the first utterance, but on new nonoverlap content words the mean context benefit is negative in all three arms. The all-bin nonoverlap arm contrasts are small relative to seed-to-seed movement (V−C +0.012, R−C −0.020, V−R +0.033 across seeds), and the partial-bin nonoverlap contrast is also near zero with V−R slightly negative. The high-bin nonoverlap V−R value is positive but has only 16 pairs and cannot carry the natural-domain result. Therefore this probe is a useful bridge for the lexical-reuse side of relation practice and a warning that adjacent child-directed utterances are often topic continuation rather than compact restatement; it is not a natural-domain replication of the compact source/rewrite mechanism.

Data outputs: `experiments/archive/relation_learning/data/childes_variation_probe_score`.
