# causal gpt relation boundary COMPACT_EXPERIENCE relation-design ordinary held-out MLM loss

This readout scores deterministic-mask MLM loss on the same 6,992 ordinary held-out rows for the five COMPACT_EXPERIENCE seed43022 relation-design arms. It separates relation-specific source-use effects from broad held-out language fit and helps bound the row-count mismatch of `SEP`.

## Late mean loss

| role | mean loss | n rows | reading |
|---|---:|---:|---|
| ALN | 2.4968 | 6992 | local own Qwen restatement is slightly better than no-Qwen OFF on ordinary held-out text |
| OFF | 2.5093 | 6992 | official-only no selected Qwen-pair practice baseline |
| SHUF | 2.5223 | 6992 | wrong-rewrite adjacency is only mildly worse than OFF |
| DUP | 2.5874 | 6992 | local exact selected-original duplication carries a broader held-out cost |
| SEP | 2.6477 | 6992 | separated original/rewrite coexistence is substantially worse, so its row-count/packing mismatch remains an important residual |

## Row-paired contrasts

| contrast | late Δloss | row-paired Δloss | reading |
|---|---:|---:|---|
| ALNminusOFF | -0.0125 | -0.0125 ± 0.0022 | aligned local restatement does not degrade ordinary held-out MLM fit |
| ALNminusSEP | -0.1509 | -0.1509 ± 0.0033 | aligned beats separated far beyond ordinary noise |
| ALNminusSHUF | -0.0255 | -0.0255 ± 0.0024 | aligned is modestly better than wrong-rewrite adjacency in broad fit |
| SHUFminusOFF | +0.0130 | +0.0130 ± 0.0020 | shuffled has only small broad fit cost, much smaller than its source-recurring T/U/N effects |
| SEPminusOFF | +0.1384 | +0.1384 ± 0.0027 | separated is broadly degraded; do not use it as an exact row-matched locality analogue |
| DUPminusOFF | +0.0781 | +0.0781 ± 0.0023 | exact duplication has broad cost in addition to relation-specific effects |
| ALNminusDUP | -0.0906 | -0.0906 ± 0.0022 | aligned restatement is broadly better than exact duplication |
| SHUFminusSEP | -0.1254 | -0.1254 ± 0.0025 | shuffled is much less broadly damaged than separated |

Data: `experiments/archive/relation_learning/data/paired_context_ordinary_heldout_loss`.
