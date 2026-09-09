# token value private readout synthesis — edit token-value and private residual readout

This is a zero-training discriminator using the frozen legal spatial repair route status checkpoint. It does not change LM parameters or run official endpoint evaluation.

## Sample
- items: `4096`; train/test: `2867`/`1229`; split keys: `2476`.
- target kinds: `{'content': 3652, 'connective': 64, 'number': 126, 'function': 156, 'pronoun': 98}`.
- legal pool SHA: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`; tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`.

## Frozen spatial repair route status raw token value
Positive true-vs-false numbers mean lower NLL with true structure. This locates structure already usable by the frozen model, not unsaturation by itself.

| comparison | mean NLL advantage | 5% boot | 95% boot | n |
|---|---:|---:|---:|---:|
| true vs decoy | 2.741850732602998 | 2.6289670648126986 | 2.8574200784137815 | 4096 |

| context | mean NLL | top1 | mean rank | NLL reduction vs base |
|---|---:|---:|---:|---:|
| base | 7.3362012475195115 | 0.06201171875 | 1088.331787109375 | 0.0 |
| true | 4.6738102907173555 | 0.4111328125 | 509.89208984375 | 2.662390956802156 |
| decoy | 7.415661023320354 | 0.060546875 | 1145.05322265625 | -0.07945977580084218 |

## Detached residual private readout
Each private readout has the same bottleneck size and uses the unchanged base-only frozen logits as an offset. Positive true-vs-false numbers mean the true-structure hidden state explains held-out residual token errors better than matched false structure.

| comparison | held-out NLL advantage | 5% boot | 95% boot | high-error advantage | high 5% | high 95% |
|---|---:|---:|---:|---:|---:|---:|
| true private vs decoy private | 2.831057869742323 | 2.560505882190186 | 3.090667047606946 | 3.5151178228365527 | 3.130263528881246 | 3.8799580977300687 |

| private context | held-out NLL reduction vs base offset | high-error reduction vs base offset | held-out top1 |
|---|---:|---:|---:|
| base | -3.7850193681571853 | -5.586902092610735 | 0.145646870136261 |
| true | -1.3001362351008454 | -2.4492468857449174 | 0.32790887355804443 |
| decoy | -4.131194104843169 | -5.96436470858147 | 0.1489015519618988 |

## Route readout
- raw_true_structure_has_token_value: `True`; min raw true advantage `2.741850732602998`.
- private_true_structure_explains_residual_errors: `True`; min held-out private advantage `2.831057869742323`; min high-error advantage `3.5151178228365527`.
- Raw token-value gains alone mean spatial repair route status can use supplied structure. The private true-vs-false residual readout is the relevant cheap signal for a future protected/private train-time pathway.

Examples CSV: `experiments/archive/frontier_consolidation/data/edit_state_token_value/edit_token_value_examples.csv`
Group CSV: `experiments/archive/frontier_consolidation/data/edit_state_token_value/edit_token_value_group_summary.csv`
JSON: `experiments/archive/frontier_consolidation/data/edit_state_token_value/edit_token_value_private_readouts.json`
