# relation filtered pair pools — no-update four-cell calibration on active relation pairs

Created: 2026-08-31T04:54:58Z  Runtime: 235.7 s

## What was scored
For each pair, the scorer masked the saved target token positions and computed `M = s(Ca,Ta)+s(Cb,Tb)-s(Ca,Tb)-s(Cb,Ta)` with frozen checkpoint weights. Two controls used same-stratum target and context permutations. No training or optimizer update was performed.

Selected pairs: **10377**; by family: `{'temporal': 3588, 'spatial': 2987, 'causal_connector': 2456, 'comparative': 1346}`
Donor assignment: `{'pairs': 10390, 'assigned': 10377, 'failed': 13, 'donor_unique': 10227, 'level_counts': {'exact': 8446, 'loose': 1931}}`
Context verification: `{'input_pairs': 10377, 'good_pairs': 10377, 'bad_contexts': 0, 'bad_examples': []}`

## Arm means for four-cell margin
| margin | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---|---:|
| true_m | 13.44625436555855 | 12.997020989369476 | 12.995300164102273 | compact > rowblock > interleaved | 0.45095420145627685 |
| tperm_m | 0.7318549225375364 | 0.715335199077503 | 0.7096753990435448 | compact > rowblock > interleaved | 0.02217952349399155 |
| cperm_m | 0.782189500407483 | 0.7711644949681552 | 0.7820558728088565 | compact > interleaved > rowblock | 0.011025005439327829 |

## Family means for true four-cell margin
| family | n | compact | rowblock | interleaved | true rank | true range | target-perm range | context-perm range |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| causal_connector | 2456 | 13.498786775394896 | 13.159314824915947 | 13.158191529489363 | compact > rowblock > interleaved | 0.3405952459055328 | 0.053141794901596096 | 0.03219677990925096 |
| comparative | 1346 | 13.882554488578489 | 13.329881624928413 | 13.288005590379683 | compact > rowblock > interleaved | 0.5945488981988056 | 0.07967265909015275 | 0.040431093469879276 |
| spatial | 2987 | 13.111815783372174 | 12.602138232950647 | 12.62273661712937 | compact > interleaved > rowblock | 0.509677550421527 | 0.023733423621244643 | 0.033419230921991616 |
| temporal | 3588 | 13.525041567578558 | 13.089799896241992 | 13.084153179054328 | compact > rowblock > interleaved | 0.44088838852423073 | 0.05745488893821604 | 0.02962478465756757 |

## Relation-oriented paired deltas
### rowblock_minus_compact
- true_m: mean=-0.4492333761890717 stderr=0.026362077955421576 success_gt0=0.4555266454659343 n=10377
- tperm_m: mean=-0.016519723460033366 stderr=0.021182871906327456 success_gt0=0.4958080370049147 n=10377
- cperm_m: mean=-0.011025005439327776 stderr=0.020863218862833335 success_gt0=0.5023609906524044 n=10377

### interleaved_minus_compact
- true_m: mean=-0.45095420145627585 stderr=0.026189618677187883 success_gt0=0.4561048472583598 n=10377
- tperm_m: mean=-0.02217952349399149 stderr=0.021013170991687185 success_gt0=0.49667533969355304 n=10377
- cperm_m: mean=-0.0001336275986265192 stderr=0.020941606087297712 success_gt0=0.49233882625036135 n=10377

### rowblock_minus_interleaved
- true_m: mean=0.001720825267204105 stderr=0.021615527494117142 success_gt0=0.4997590825864894 n=10377
- tperm_m: mean=0.005659800033958124 stderr=0.01956110972627472 success_gt0=0.49763900934759564 n=10377
- cperm_m: mean=-0.010891377840701258 stderr=0.019476263017469145 success_gt0=0.4933988628698082 n=10377

## Files
- JSON summary: `experiments/archive/representation_and_objectives/data/intrarow_twoarg_calibration_pilot20k/fourcell_calibration_summary.json`
- Pair scores: `experiments/archive/representation_and_objectives/data/intrarow_twoarg_calibration_pilot20k/fourcell_pair_scores.jsonl`
- CSV: `experiments/archive/representation_and_objectives/data/intrarow_twoarg_calibration_pilot20k/fourcell_pair_scores_summary.csv`
