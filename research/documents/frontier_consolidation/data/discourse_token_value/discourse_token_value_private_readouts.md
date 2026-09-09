# token value private readout synthesis — discourse token-value and private residual readout

This is a zero-training discriminator using the frozen legal spatial repair route status checkpoint. It does not change LM parameters or run official endpoint evaluation.

## Sample
- items: `4096`; train/test: `2873`/`1223`; split keys: `3692`.
- target kinds: `{'content': 3670, 'number': 299, 'function': 59, 'pronoun': 52, 'connective': 16}`.
- legal pool SHA: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`; tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`.

## Frozen spatial repair route status raw token value
Positive true-vs-false numbers mean lower NLL with true structure. This locates structure already usable by the frozen model, not unsaturation by itself.

| comparison | mean NLL advantage | 5% boot | 95% boot | n |
|---|---:|---:|---:|---:|
| true vs reversed | 0.018266387062130462 | 0.007372943738740645 | 0.02849889656572202 | 4096 |
| true vs shuffled | 0.49339718942439603 | 0.4403684904568905 | 0.5445800530925133 | 4096 |

| context | mean NLL | top1 | mean rank | NLL reduction vs base |
|---|---:|---:|---:|---:|
| base | 4.7247096751821545 | 0.275390625 | 453.255859375 | 0.0 |
| true | 4.222108415553702 | 0.344482421875 | 360.400146484375 | 0.5026012596284524 |
| reversed | 4.2403748026158326 | 0.342041015625 | 361.548583984375 | 0.4843348725663219 |
| shuffled | 4.715505604978098 | 0.27294921875 | 440.664306640625 | 0.009204070204056336 |

## Detached residual private readout
Each private readout has the same bottleneck size and uses the unchanged base-only frozen logits as an offset. Positive true-vs-false numbers mean the true-structure hidden state explains held-out residual token errors better than matched false structure.

| comparison | held-out NLL advantage | 5% boot | 95% boot | high-error advantage | high 5% | high 95% |
|---|---:|---:|---:|---:|---:|---:|
| true private vs reversed private | 0.12252874615740718 | 0.029879368530865657 | 0.19724643445567863 | 0.15546050741644638 | 0.010840339214066226 | 0.3224723185344106 |
| true private vs shuffled private | 0.014503284964010316 | -0.10690400809677605 | 0.12402933252249052 | 0.13414308751455442 | -0.07266993263243314 | 0.36277760594727143 |

| private context | held-out NLL reduction vs base offset | high-error reduction vs base offset | held-out top1 |
|---|---:|---:|---:|
| base | -3.076941716825926 | -5.697572183264176 | 0.38266557455062866 |
| true | -3.0858982936769097 | -5.699246326455348 | 0.3875715434551239 |
| reversed | -3.2084270398343167 | -5.854706833871794 | 0.38103026151657104 |
| shuffled | -3.10040157864092 | -5.833389413969901 | 0.3883891999721527 |

## Route readout
- raw_true_structure_has_token_value: `False`; min raw true advantage `0.018266387062130462`.
- private_true_structure_explains_residual_errors: `False`; min held-out private advantage `0.014503284964010316`; min high-error advantage `0.13414308751455442`.
- Raw token-value gains alone mean spatial repair route status can use supplied structure. The private true-vs-false residual readout is the relevant cheap signal for a future protected/private train-time pathway.

Examples CSV: `experiments/archive/frontier_consolidation/data/discourse_token_value/discourse_token_value_examples.csv`
Group CSV: `experiments/archive/frontier_consolidation/data/discourse_token_value/discourse_token_value_group_summary.csv`
JSON: `experiments/archive/frontier_consolidation/data/discourse_token_value/discourse_token_value_private_readouts.json`
