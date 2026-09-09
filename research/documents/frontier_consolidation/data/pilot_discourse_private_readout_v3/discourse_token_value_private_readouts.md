# token value private readout synthesis — discourse token-value and private residual readout

This is a zero-training discriminator using the frozen legal spatial repair route status checkpoint. It does not change LM parameters or run official endpoint evaluation.

## Sample
- items: `128`; train/test: `90`/`38`; split keys: `126`.
- target kinds: `{'content': 112, 'number': 13, 'function': 2, 'connective': 1}`.
- legal pool SHA: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`; tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`.

## Frozen spatial repair route status raw token value
Positive true-vs-false numbers mean lower NLL with true structure. This locates structure already usable by the frozen model, not unsaturation by itself.

| comparison | mean NLL advantage | 5% boot | 95% boot | n |
|---|---:|---:|---:|---:|
| true vs reversed | 0.04328868941274777 | -0.020030400875384657 | 0.10707559793991095 | 128 |
| true vs shuffled | 0.4556609487094647 | 0.11799396594233258 | 0.7587115507672024 | 128 |

| context | mean NLL | top1 | mean rank | NLL reduction vs base |
|---|---:|---:|---:|---:|
| base | 4.530141889245897 | 0.2421875 | 381.71875 | 0.0 |
| true | 4.034671497594445 | 0.34375 | 328.96875 | 0.4954703916514518 |
| reversed | 4.077960187007193 | 0.3515625 | 351.109375 | 0.452181702238704 |
| shuffled | 4.49033244630391 | 0.25 | 397.09375 | 0.03980944294198707 |

## Detached residual private readout
Each private readout has the same bottleneck size and uses the unchanged base-only frozen logits as an offset. Positive true-vs-false numbers mean the true-structure hidden state explains held-out residual token errors better than matched false structure.

| comparison | held-out NLL advantage | 5% boot | 95% boot | high-error advantage | high 5% | high 95% |
|---|---:|---:|---:|---:|---:|---:|
| true private vs reversed private | -0.008262934408297664 | -0.013530541358417586 | -0.0036626296902173444 | -0.009311751315468237 | -0.018453096088610198 | -0.0019348546078330592 |
| true private vs shuffled private | -0.006501866728189941 | -0.011964777582570127 | -0.0011337352457064156 | -0.009647444674843237 | -0.0177503134074964 | -0.001736364866557874 |

| private context | held-out NLL reduction vs base offset | high-error reduction vs base offset | held-out top1 |
|---|---:|---:|---:|
| base | 0.03121255480647577 | 0.004187684310109992 | 0.2368421107530594 |
| true | 0.029232477113653562 | -0.004320370523553146 | 0.2368421107530594 |
| reversed | 0.037495411521951225 | 0.00499138079191509 | 0.2368421107530594 |
| shuffled | 0.035734343841843506 | 0.00532707415129009 | 0.2368421107530594 |

## Route readout
- raw_true_structure_has_token_value: `True`; min raw true advantage `0.04328868941274777`.
- private_true_structure_explains_residual_errors: `False`; min held-out private advantage `-0.008262934408297664`; min high-error advantage `-0.009647444674843237`.
- Raw token-value gains alone mean spatial repair route status can use supplied structure. The private true-vs-false residual readout is the relevant cheap signal for a future protected/private train-time pathway.

Examples CSV: `experiments/archive/frontier_consolidation/data/pilot_discourse_private_readout_v3/discourse_token_value_examples.csv`
Group CSV: `experiments/archive/frontier_consolidation/data/pilot_discourse_private_readout_v3/discourse_token_value_group_summary.csv`
JSON: `experiments/archive/frontier_consolidation/data/pilot_discourse_private_readout_v3/discourse_token_value_private_readouts.json`
