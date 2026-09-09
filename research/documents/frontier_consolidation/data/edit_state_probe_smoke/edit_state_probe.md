# edit state probe chck82 synthesis — source-absent edit-state probe: smoke

Checkpoint: `experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M`  
Model class: `DebertaV2ForMaskedLM` (34,467,424 params)  
Trust remote code: `False`  
Items: 12 (train 8, test 4)  
Item reproducibility vs spatial repair route status: `True`

## Raw frozen token value
Positive true-vs-decoy = lower NLL with true source → model uses edit structure.

| comparison | mean NLL advantage | 5% boot | 95% boot |
|---|---:|---:|---:|
| true vs decoy | -0.6389953251928091 | -1.885440742596984 | 0.5181405190378428 |

| context | mean NLL | top1 | mean rank |
|---|---:|---:|---:|
| base | 6.675267235686381 | 0.0833333358168602 | 827.8333333333334 |
| true | 7.800272279108564 | 0.1666666716337204 | 1384.3333333333333 |
| decoy | 7.161276953915755 | 0.0833333358168602 | 1121.9166666666667 |

## Detached private readout
Positive true-vs-decoy = true-structure hidden state explains residual errors better.

| comparison | held-out NLL advantage | 5% boot | 95% boot | high-error advantage | high 5% | high 95% |
|---|---:|---:|---:|---:|---:|---:|
| true vs decoy | 0.3032548427581787 | -0.02399420738220215 | 0.5701873302459717 | 0.24293923377990723 | -0.20273399353027344 | 0.6886124610900879 |

## Comparison vs spatial repair route status baseline

| metric | spatial repair route status | current | delta |
|---|---:|---:|---:|
| raw true-vs-decoy NLL | -0.3122 | -0.6389953251928091 | -0.3268094401090214 |
| private true-vs-decoy held-out | 0.8365 | 0.3032548427581787 | -0.5332361250304581 |
| private true-vs-decoy high-error | 1.2397 | 0.24293923377990723 | -0.996756899389236 |

## Route readout
- raw_true_structure_has_token_value: `False`
- private_true_structure_explains_residual_errors: `True`

## Interpretation
A positive private readout establishes ONLY decodability of edit-state information.
It does NOT establish that this information can survive source removal (source free transfer synthesis-style test),
that it relates to EWoK/Entity relation/state competence, or that it justifies a new trainer.

JSON: `experiments/archive/frontier_consolidation/data/edit_state_probe_smoke/edit_state_probe.json`
