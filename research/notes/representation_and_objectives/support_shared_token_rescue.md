# discriminating span support — Support-shared rare-token representation measurement

CPU-only. No training, no model evaluation, and no managed-task query. This measures whether the legal40k low-support-token problem can be attacked by sharing representation with legal16k components learned from the same allowed 10M corpus.

## Vocabulary-level decomposition

| direct legal40k count band | tokens | all legal16 components >=50 | all components >=100 | all components >=200 |
|---|---:|---:|---:|---:|
| lt20 | 11452 | 0.9251 | 0.7751 | 0.5632 |
| 20_49 | 14077 | 0.9657 | 0.7963 | 0.5599 |
| 50_99 | 5996 | 0.9922 | 0.5622 | 0.3574 |
| 100_199 | 3620 | 0.9959 | 0.9865 | 0.2655 |
| ge200 | 4850 | 0.9981 | 0.9895 | 0.9757 |

## Official-text surface exposure through legal40k low-count tokens

| family | legal40k frac<50 | frac<100 | among <50, legal16 component-min >=50 | among <100, component-min >=50 | mean component-min support for <100 | leader gap from legal40k 8x480 |
|---|---:|---:|---:|---:|---:|---:|
| EWoK | 0.0878 | 0.1495 | 0.9895 | 0.9933 | 992.7 | 4.596 |
| GlobalPIQA | 0.0975 | 0.1609 | 0.8129 | 0.8866 | 2598.3 | 5.005 |
| Supplement | 0.0495 | 0.0822 | 0.8447 | 0.9064 | 366.1 | -4.956 |
| SuperGLUE | 0.0738 | 0.1240 | 0.9483 | 0.9672 | 715.0 | 1.280 |
| BLiMP | 0.0851 | 0.1574 | 0.9705 | 0.9830 | 430.0 | -0.749 |
| COMPS | 0.1292 | 0.1798 | 0.9304 | 0.9500 | 561.9 | 1.895 |
| Entity | 0.0031 | 0.1151 | 1.0000 | 1.0000 | 113.6 | 1.246 |
| Reading | 0.0126 | 0.0294 | 1.0000 | 1.0000 | 767.6 | -2.403 |

## Scientific reading

A support-shared 40k representation is mechanically plausible if many score-text low-support 40k tokens decompose into legal16 pieces with much higher corpus support. It would attack a different bottleneck than U256: preserve the shorter 40k segmentation and high Supplement/BLiMP behavior while reducing the undertrained-row problem visible in EWoK and GlobalPIQA. This is construction evidence only; it becomes a serious next route only if the pending depth and minfreq50 score vectors show complementary recovery consistent with representation support rather than merely dialogue-tail visibility.

JSON: `experiments/archive/representation_and_objectives/data/support_shared_token_rescue/support_shared_token_rescue.json`
CSV: `experiments/archive/representation_and_objectives/data/support_shared_token_rescue/eval_family_component_rescue_summary.csv`, `experiments/archive/representation_and_objectives/data/support_shared_token_rescue/rare40k_token_component_support.csv`
