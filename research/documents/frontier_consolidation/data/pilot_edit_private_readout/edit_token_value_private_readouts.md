# token value private readout synthesis — edit token-value and private residual readout

This is a zero-training discriminator using the frozen legal spatial repair route status checkpoint. It does not change LM parameters or run official endpoint evaluation.

## Sample
- items: `64`; train/test: `45`/`19`; split keys: `63`.
- target kinds: `{'content': 63, 'number': 1}`.
- legal pool SHA: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`; tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`.

## Frozen spatial repair route status raw token value
Positive true-vs-false numbers mean lower NLL with true structure. This locates structure already usable by the frozen model, not unsaturation by itself.

| comparison | mean NLL advantage | 5% boot | 95% boot | n |
|---|---:|---:|---:|---:|
| true vs decoy | 2.392829327232903 | 1.5036477710818872 | 3.219261860009283 | 64 |

| context | mean NLL | top1 | mean rank | NLL reduction vs base |
|---|---:|---:|---:|---:|
| base | 8.720177499577403 | 0.015625 | 829.96875 | 0.0 |
| true | 7.011581701837713 | 0.203125 | 1801.515625 | 1.7085957977396902 |
| decoy | 9.404411029070616 | 0.015625 | 2520.078125 | -0.6842335294932127 |

## Detached residual private readout
Each private readout has the same bottleneck size and uses the unchanged base-only frozen logits as an offset. Positive true-vs-false numbers mean the true-structure hidden state explains held-out residual token errors better than matched false structure.

| comparison | held-out NLL advantage | 5% boot | 95% boot | high-error advantage | high 5% | high 95% |
|---|---:|---:|---:|---:|---:|---:|
| true private vs decoy private | 0.003094371996427837 | -0.00450804359034488 | 0.010048289048044305 | 0.0070702552795410155 | -0.002082347869873047 | 0.015212535858154297 |

| private context | held-out NLL reduction vs base offset | high-error reduction vs base offset | held-out top1 |
|---|---:|---:|---:|
| base | 0.0029613344292891653 | 0.0020593643188476563 | 0.0 |
| true | 0.004302853032162315 | 0.005210399627685547 | 0.0 |
| decoy | 0.0012084810357344778 | -0.0018598556518554688 | 0.0 |

## Route readout
- raw_true_structure_has_token_value: `True`; min raw true advantage `2.392829327232903`.
- private_true_structure_explains_residual_errors: `False`; min held-out private advantage `0.003094371996427837`; min high-error advantage `0.0070702552795410155`.
- Raw token-value gains alone mean spatial repair route status can use supplied structure. The private true-vs-false residual readout is the relevant cheap signal for a future protected/private train-time pathway.

Examples CSV: `experiments/archive/frontier_consolidation/data/pilot_edit_private_readout/edit_token_value_examples.csv`
Group CSV: `experiments/archive/frontier_consolidation/data/pilot_edit_private_readout/edit_token_value_group_summary.csv`
JSON: `experiments/archive/frontier_consolidation/data/pilot_edit_private_readout/edit_token_value_private_readouts.json`
