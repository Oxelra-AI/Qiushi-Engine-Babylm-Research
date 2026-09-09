# Answer-only natural-row training pilot

This is a scorer/training preflight on the noisy 55-pair state-update pilot rows, not evidence that these rows are ready for BabyLM training.

Train pairs: 40; held pairs: 15; epochs: 90; lr: 5e-05.

| Eval | joint | update | retain | mean U | mean R | mean U+R |
|---|---:|---:|---:|---:|---:|---:|
| baseline train | 2/40 | 30/40 | 9/40 | +2.171 | -2.086 | +0.085 |
| baseline held | 1/15 | 12/15 | 4/15 | +1.307 | -1.603 | -0.296 |
| final train | 14/40 | 34/40 | 20/40 | +2.009 | +0.594 | +2.603 |
| final held | 0/15 | 11/15 | 3/15 | +2.273 | -2.040 | +0.233 |

## Interpretation

The training masks only the final answer span and leaves source/update evidence visible. Movement on train rows primarily validates that the multi-token answer-loss path can shape the private adapter. Movement on held rows, if any, is only suggestive because the pilot rows contain semantic noise and are few. The decisive next experiment still needs cleaner accepted state-update rows and an exposure-matched ALN-preserving continuation control.
