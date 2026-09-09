# edit state probe chck82 synthesis — source-absent edit-state probe: chck_82M_scale1p75

Checkpoint: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M`  
Model class: `AdapterDebertaV2ForMaskedLM` (35,463,008 params)  
Trust remote code: `True`  
Items: 3072 (train 2153, test 919)  
Item reproducibility vs spatial repair route status: `True`

## Raw frozen token value
Positive true-vs-decoy = lower NLL with true source → model uses edit structure.

| comparison | mean NLL advantage | 5% boot | 95% boot |
|---|---:|---:|---:|
| true vs decoy | -0.17559004255660207 | -0.25036324464538967 | -0.09996694037242075 |

| context | mean NLL | top1 | mean rank |
|---|---:|---:|---:|
| base | 7.618329157387204 | 0.0377604179084301 | 1216.4182942708333 |
| true | 7.921197098990281 | 0.0341796875 | 954.6490885416666 |
| decoy | 7.74560705643368 | 0.0387369804084301 | 1315.7682291666667 |

## Detached private readout
Positive true-vs-decoy = true-structure hidden state explains residual errors better.

| comparison | held-out NLL advantage | 5% boot | 95% boot | high-error advantage | high 5% | high 95% |
|---|---:|---:|---:|---:|---:|---:|
| true vs decoy | 0.7463929184499484 | 0.569878026501861 | 0.9120502970633481 | 0.9505718224115022 | 0.6541807152128414 | 1.2502119224012385 |

## Comparison vs spatial repair route status baseline

| metric | spatial repair route status | current | delta |
|---|---:|---:|---:|
| raw true-vs-decoy NLL | -0.3122 | -0.17559004255660207 | 0.13659584252718562 |
| private true-vs-decoy held-out | 0.8365 | 0.7463929184499484 | -0.09009804933868848 |
| private true-vs-decoy high-error | 1.2397 | 0.9505718224115022 | -0.289124310757641 |

## Route readout
- raw_true_structure_has_token_value: `False`
- private_true_structure_explains_residual_errors: `True`

## Interpretation
A positive private readout establishes ONLY decodability of edit-state information.
It does NOT establish that this information can survive source removal (source free transfer synthesis-style test),
that it relates to EWoK/Entity relation/state competence, or that it justifies a new trainer.

JSON: `experiments/archive/frontier_consolidation/data/edit_state_probe_chck82/edit_state_probe.json`
