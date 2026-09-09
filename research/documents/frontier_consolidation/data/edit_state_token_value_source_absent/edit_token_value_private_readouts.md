# token value private readout synthesis — edit token-value and private residual readout

This is a zero-training discriminator using the frozen legal spatial repair route status checkpoint. It does not change LM parameters or run official endpoint evaluation.

## Sample
- items: `3072`; train/test: `2153`/`919`; split keys: `2118`.
- target kinds: `{'content': 2651, 'connective': 77, 'function': 227, 'pronoun': 117}`.
- legal pool SHA: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`; tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`.

## Frozen spatial repair route status raw token value
Positive true-vs-false numbers mean lower NLL with true structure. This locates structure already usable by the frozen model, not unsaturation by itself.

| comparison | mean NLL advantage | 5% boot | 95% boot | n |
|---|---:|---:|---:|---:|
| true vs decoy | -0.3121858850837877 | -0.38324652059751924 | -0.23961844789452394 | 3072 |

| context | mean NLL | top1 | mean rank | NLL reduction vs base |
|---|---:|---:|---:|---:|
| base | 7.328919534304077 | 0.0419921875 | 1040.1712239583333 | 0.0 |
| true | 7.762004700954094 | 0.0263671875 | 876.2210286458334 | -0.4330851666500166 |
| decoy | 7.449818815870306 | 0.0436197929084301 | 1138.111328125 | -0.12089928156622894 |

## Detached residual private readout
Each private readout has the same bottleneck size and uses the unchanged base-only frozen logits as an offset. Positive true-vs-false numbers mean the true-structure hidden state explains held-out residual token errors better than matched false structure.

| comparison | held-out NLL advantage | 5% boot | 95% boot | high-error advantage | high 5% | high 95% |
|---|---:|---:|---:|---:|---:|---:|
| true private vs decoy private | 0.8364909677886369 | 0.6506157301614488 | 1.0051685398749681 | 1.2396961331691432 | 0.9663132199146988 | 1.5115360522237808 |

| private context | held-out NLL reduction vs base offset | high-error reduction vs base offset | held-out top1 |
|---|---:|---:|---:|
| base | -1.6951844573021695 | -3.0343797433271034 | 0.23177365958690643 |
| true | -1.1453247062944623 | -2.1500179965170507 | 0.26985853910446167 |
| decoy | -1.9818156740830992 | -3.3897141296861935 | 0.2241566926240921 |

## Route readout
- raw_true_structure_has_token_value: `False`; min raw true advantage `-0.3121858850837877`.
- private_true_structure_explains_residual_errors: `True`; min held-out private advantage `0.8364909677886369`; min high-error advantage `1.2396961331691432`.
- Raw token-value gains alone mean spatial repair route status can use supplied structure. The private true-vs-false residual readout is the relevant cheap signal for a future protected/private train-time pathway.

Examples CSV: `experiments/archive/frontier_consolidation/data/edit_state_token_value_source_absent/edit_token_value_examples.csv`
Group CSV: `experiments/archive/frontier_consolidation/data/edit_state_token_value_source_absent/edit_token_value_group_summary.csv`
JSON: `experiments/archive/frontier_consolidation/data/edit_state_token_value_source_absent/edit_token_value_private_readouts.json`
