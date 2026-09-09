# token value private readout synthesis — edit token-value and private residual readout

This is a zero-training discriminator using the frozen legal spatial repair route status checkpoint. It does not change LM parameters or run official endpoint evaluation.

## Sample
- items: `128`; train/test: `89`/`39`; split keys: `126`.
- target kinds: `{'content': 54, 'connective': 12, 'pronoun': 14, 'number': 5, 'function': 43}`.
- legal pool SHA: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`; tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`.

## Frozen spatial repair route status raw token value
Positive true-vs-false numbers mean lower NLL with true structure. This locates structure already usable by the frozen model, not unsaturation by itself.

| comparison | mean NLL advantage | 5% boot | 95% boot | n |
|---|---:|---:|---:|---:|
| true vs decoy | 1.2432499827555148 | 0.7565129213617183 | 1.7441003906860715 | 128 |

| context | mean NLL | top1 | mean rank | NLL reduction vs base |
|---|---:|---:|---:|---:|
| base | 7.98530164710246 | 0.03125 | 1667.2734375 | 0.0 |
| true | 7.4808975074702175 | 0.1328125 | 1193.8125 | 0.5044041396322427 |
| decoy | 8.724147490225732 | 0.015625 | 2076.8125 | -0.7388458431232721 |

## Detached residual private readout
Each private readout has the same bottleneck size and uses the unchanged base-only frozen logits as an offset. Positive true-vs-false numbers mean the true-structure hidden state explains held-out residual token errors better than matched false structure.

| comparison | held-out NLL advantage | 5% boot | 95% boot | high-error advantage | high 5% | high 95% |
|---|---:|---:|---:|---:|---:|---:|
| true private vs decoy private | -0.02002692833924905 | -0.030533845608051006 | -0.010291808690780248 | -0.01601095199584961 | -0.026893424987792968 | -0.005913591384887696 |

| private context | held-out NLL reduction vs base offset | high-error reduction vs base offset | held-out top1 |
|---|---:|---:|---:|
| base | 0.042978995885604464 | 0.020347166061401366 | 0.0 |
| true | 0.024779906639685996 | 0.004736757278442383 | 0.0 |
| decoy | 0.04480683497893505 | 0.02074770927429199 | 0.0 |

## Route readout
- raw_true_structure_has_token_value: `True`; min raw true advantage `1.2432499827555148`.
- private_true_structure_explains_residual_errors: `False`; min held-out private advantage `-0.02002692833924905`; min high-error advantage `-0.01601095199584961`.
- Raw token-value gains alone mean spatial repair route status can use supplied structure. The private true-vs-false residual readout is the relevant cheap signal for a future protected/private train-time pathway.

Examples CSV: `experiments/archive/frontier_consolidation/data/pilot_edit_private_readout_v2/edit_token_value_examples.csv`
Group CSV: `experiments/archive/frontier_consolidation/data/pilot_edit_private_readout_v2/edit_token_value_group_summary.csv`
JSON: `experiments/archive/frontier_consolidation/data/pilot_edit_private_readout_v2/edit_token_value_private_readouts.json`
