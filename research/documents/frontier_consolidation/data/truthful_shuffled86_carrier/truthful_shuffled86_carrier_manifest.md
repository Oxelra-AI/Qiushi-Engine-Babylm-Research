# lead frozen anchor fastpath route truthful shuffled86 carrier materialization

Status: **TRUTHFUL_SHUFFLED86_CARRIER_MATERIALIZED**

Carrier: `experiments/archive/frontier_consolidation/data/truthful_shuffled86_carrier/all_full_preds_truthful_shuffled86_mlm.json`
Carrier SHA256: `8c9885efa84727c4997c4d6d492b8cc8ae78ec74775ac94698ea059bce636764`
Model: `experiments/archive/frontier_consolidation/training/runs/frozen82_tail4M_shuffled_seed43022/hf_model/final`
Model SHA256: `7a090773127763e764e5079a644e50463e2c4fd0788ab094d9be9fbea083bd6c`

## Truthful history policy
- AoA is represented only as scalar `{'aoa': 0.0}`.
- `fast_eval_results` is omitted; the protected 82M fast checkpoint history is not copied.
- Full-task predictions are merged from existing frozen82 short tail plan zero-shot/reading outputs and earlier analysis repeat SuperGLUE outputs for the same shuffled86 model.

## Candidate-native arithmetic
| column | score |
|---|---:|
| BLiMP | 69.23 |
| Supplement | 59.81 |
| EWoK | 51.44 |
| Entity | 26.77 |
| COMPS | 52.65 |
| GlobalPIQA | 39.565 |
| Reading | 8.625 |
| SuperGLUE | 69.8192223897 |
| AoA | 0 |

Cheap7: `44.01285714285714`
Overall with AoA 0: `41.98991359885548`
Delta vs protected chck82: `+0.047432431469`

## Current validator
`is_valid_predictions(..., strict-small)` -> `True` / `Upload successful.`

Scientific reading: this carrier is a truthful local prediction artifact for the generic shuffled private-tail endpoint hypothesis. It is not evidence for source correspondence and it should be treated as a baseline for stronger frozen-anchor fast-path consolidation, not as a mechanism by itself.

JSON: `experiments/archive/frontier_consolidation/data/truthful_shuffled86_carrier/truthful_shuffled86_carrier_manifest.json`
