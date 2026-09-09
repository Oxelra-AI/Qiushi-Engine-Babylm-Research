# frozen anchor coherent replay item reading truthful coherent86 carrier materialization

Status: **TRUTHFUL_COHERENT86_CARRIER_MATERIALIZED**

Carrier: `experiments/archive/frontier_consolidation/data/truthful_coherent86_carrier/all_full_preds_truthful_coherent86_mlm.json`
Carrier SHA256: `4a0278a689ea48bdb88215090e34a94caa3fc8ba10df78d19a9195d3698b533e`
Model: `experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/final`
Model SHA256: `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`
Replay model SHA256: `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`
Replay bit-identical: `True`

## Truthful history policy
- AoA is represented only as scalar `{'aoa': 0.0}`.
- `fast_eval_results` is omitted; the protected 82M fast checkpoint history is not copied.
- Full-task predictions are merged from frozen anchor fastpath disruption design zero-shot/reading outputs and frozen anchor coherent replay item reading SuperGLUE outputs for the same coherent model.

## Candidate-native arithmetic
| column | score | delta vs chck82 |
|---|---:|---:|
| BLiMP | 68.52 | +0.0287159634801 |
| Supplement | 63.65 | +0.7121887438 |
| EWoK | 49.91 | -0.145453325533 |
| Entity | 28.44 | +0.125958069701 |
| COMPS | 51.99 | -0.201175094436 |
| GlobalPIQA | 38.065 | +0.487330097087 |
| Reading | 8.17 | +0.0212864107381 |
| SuperGLUE | 69.7779682643 | +0.011786892975 |
| AoA | 0 | +0 |

Cheap7: `44.10642857142857` (delta `+0.146978694977`)
Overall with AoA 0: `42.058107584920755` (delta `+0.115626417535`)

## Validator
`is_valid_predictions(..., strict-small)` -> `True` / `Upload successful.`

## Scientific reading
This is a truthful local prediction carrier for the coherent private-only replay endpoint candidate. It is stronger numerically than the submitted 82M anchor, but the item-transition and overlap analyses show family redistribution rather than a broad added-decision mechanism.

JSON: `experiments/archive/frontier_consolidation/data/truthful_coherent86_carrier/truthful_coherent86_carrier_manifest.json`
