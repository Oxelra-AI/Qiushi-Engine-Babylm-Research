# edit state probe chck82 synthesis — source-absent edit-state probe: verify

Checkpoint: `experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M`  
Model class: `DebertaV2ForMaskedLM` (34,467,424 params)  
Trust remote code: `False`  
Items: 20 (train 14, test 6)  
Item reproducibility vs spatial repair route status: `True`

## Raw frozen token value
Positive true-vs-decoy = lower NLL with true source → model uses edit structure.

| comparison | mean NLL advantage | 5% boot | 95% boot |
|---|---:|---:|---:|
| true vs decoy | -0.6361268613487482 | -1.6454877112060786 | 0.356586292013526 |

| context | mean NLL | top1 | mean rank |
|---|---:|---:|---:|
| base | 6.857267031818628 | 0.05000000074505806 | 770.5 |
| true | 7.695060642436147 | 0.10000000149011612 | 1155.35 |
| decoy | 7.058933781087399 | 0.05000000074505806 | 1030.0 |

## Detached private readout
Positive true-vs-decoy = true-structure hidden state explains residual errors better.

| comparison | held-out NLL advantage | 5% boot | 95% boot | high-error advantage | high 5% | high 95% |
|---|---:|---:|---:|---:|---:|---:|
| true vs decoy | 0.15842162646974126 | -0.02934913756325841 | 0.3633890946706136 | 0.35392268498738605 | 0.07724269231160481 | 0.6306026776631674 |

## Comparison vs spatial repair route status baseline

| metric | spatial repair route status | current | delta |
|---|---:|---:|---:|
| raw true-vs-decoy NLL | -0.3122 | -0.6361268613487482 | -0.3239409762649605 |
| private true-vs-decoy held-out | 0.8365 | 0.15842162646974126 | -0.6780693413188956 |
| private true-vs-decoy high-error | 1.2397 | 0.35392268498738605 | -0.8857734481817572 |

## Route readout
- raw_true_structure_has_token_value: `False`
- private_true_structure_explains_residual_errors: `True`

## Interpretation
A positive private readout establishes ONLY decodability of edit-state information.
It does NOT establish that this information can survive source removal (source free transfer synthesis-style test),
that it relates to EWoK/Entity relation/state competence, or that it justifies a new trainer.

JSON: `experiments/archive/frontier_consolidation/data/edit_state_probe_verify/edit_state_probe.json`
