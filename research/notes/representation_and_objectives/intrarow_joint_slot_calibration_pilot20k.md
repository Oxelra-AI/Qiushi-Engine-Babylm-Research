# intrarow calibration synthesis — intra-row joint slot calibration

Created: 2026-08-31T04:59:38Z  Runtime: 78.2 s

Both argument positions are masked simultaneously. The margin is original left/right assignment minus swapped left/right assignment from the same frozen logits. No training was performed.
Selected pairs: **9,800**; by family: `{'temporal': 3379, 'spatial': 2841, 'causal_connector': 2297, 'comparative': 1283}`
Context verification: `{'input_pairs': 9800, 'good_pairs': 9800, 'bad_contexts': 0, 'bad_examples': []}`

## Arm means
| metric | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---|---:|
| slot_margin | 6.820404555146744 | 6.638092926572106 | 6.615369065550336 | compact > rowblock > interleaved | 0.2050354895964075 |
| left_margin | 6.831701153279656 | 6.634579641666959 | 6.599813596414133 | compact > rowblock > interleaved | 0.23188755686552298 |
| right_margin | 6.8091079431322905 | 6.641606206761546 | 6.630924524396056 | compact > rowblock > interleaved | 0.17818341873623478 |
| orig_score | -3.6812834311383584 | -3.7322207283956113 | -3.7416183305451436 | compact > rowblock > interleaved | 0.06033489940678516 |
| swap_score | -10.501687986285102 | -10.370313654967717 | -10.35698739609548 | interleaved > rowblock > compact | 0.1447005901896219 |

## Family slot-margin means
| family | n | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---:|---|---:|
| causal_connector | 2297 | 6.877488002765714 | 6.728794002021198 | 6.741638416730105 | compact > interleaved > rowblock | 0.14869400074451633 |
| comparative | 1283 | 7.04006857595233 | 6.815845015275111 | 6.748544414207595 | compact > rowblock > interleaved | 0.29152416174473483 |
| spatial | 2841 | 6.614797917903565 | 6.4224834386551075 | 6.391758559980594 | compact > rowblock > interleaved | 0.22303935792297125 |
| temporal | 3379 | 6.871064169983539 | 6.690223809987219 | 6.666973911462279 | compact > rowblock > interleaved | 0.20409025852126028 |

## Relation-oriented paired deltas
### rowblock_minus_compact
- slot_margin: mean=-0.1823116285746385 stderr=0.013084151735045641 success_gt0=0.4605102040816327 n=9800
- left_margin: mean=-0.1971215116126969 stderr=0.01846113344262386 success_gt0=0.4736734693877551 n=9800
- right_margin: mean=-0.16750173637074448 stderr=0.018781078120738297 success_gt0=0.4736734693877551 n=9800

### interleaved_minus_compact
- slot_margin: mean=-0.2050354895964075 stderr=0.013163407944949345 success_gt0=0.4633673469387755 n=9800
- left_margin: mean=-0.23188755686552268 stderr=0.018515990169806793 success_gt0=0.47 n=9800
- right_margin: mean=-0.17818341873623447 stderr=0.01892025777556602 success_gt0=0.4754081632653061 n=9800

### rowblock_minus_interleaved
- slot_margin: mean=0.022723861021768987 stderr=0.0108399171393774 success_gt0=0.5031632653061224 n=9800
- left_margin: mean=0.0347660452528258 stderr=0.01563824296177282 success_gt0=0.5040816326530613 n=9800
- right_margin: mean=0.010681682365490016 stderr=0.015834215912860532 success_gt0=0.5009183673469387 n=9800

Files:
- JSON summary: `experiments/archive/representation_and_objectives/data/intrarow_joint_slot_calibration_pilot20k/joint_slot_calibration_summary.json`
- pair scores: `experiments/archive/representation_and_objectives/data/intrarow_joint_slot_calibration_pilot20k/joint_slot_pair_scores.jsonl`
- CSV: `experiments/archive/representation_and_objectives/data/intrarow_joint_slot_calibration_pilot20k/joint_slot_pair_scores_summary.csv`
