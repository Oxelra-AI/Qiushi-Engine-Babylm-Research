# earlier analysis split paired-seed evaluation aggregation

Created: `2026-09-08T08:48:05Z`

This file reconciles independently executed component payloads. It does not alter scores or rerun evaluation. Official-entry payloads are preferred when independent evaluations produce the same component; all found records remain listed in JSON.

## o62065 (ordinary_inherited_wwm_seed62065)

Complete for Overall: `False`; Overall: `None`; missing: `['BLiMP', 'Entity', 'COMPS', 'SuperGLUE', 'AoA']`

| component | score | source | path |
|---|---:|---|---|
| BLiMP | - | - | `` |
| Supplement | 63.61985289165889 | A02_step127 | `experiments/archive/relation_learning/data/o62065_Supplement/o62065/with_special/Supplement_component_payload.json` |
| EWoK | 49.81310755092491 | A02_step127 | `experiments/archive/relation_learning/data/o62065_EWoK/o62065/with_special/EWoK_component_payload.json` |
| Entity | - | - | `` |
| COMPS | - | - | `` |
| GlobalPIQA_parallel | 31.067961165048544 | A02_step127 | `experiments/archive/relation_learning/data/o62065_GlobalPIQA_parallel/o62065/with_special/GlobalPIQA_parallel_component_payload.json` |
| GlobalPIQA_nonparallel | 48.0 | A02_step127 | `experiments/archive/relation_learning/data/o62065_GlobalPIQA_nonparallel/o62065/with_special/GlobalPIQA_nonparallel_component_payload.json` |
| GlobalPIQA mean | 39.53398058252427 | derived | - |
| Reading | 8.19655294965572 | A02_step127 | `experiments/archive/relation_learning/data/o62065_Reading/o62065/with_special/Reading_component_payload.json` |
| AoA | None | - | `` |
| SuperGLUE mean | None | derived from tasks | - |

SuperGLUE tasks: boolq=None, multirc=None, rte=None, wsc=None, mrpc=None, qqp=None, mnli=None

## ms62065 (ms_acquisition_seed62065)

Complete for Overall: `False`; Overall: `None`; missing: `['EWoK', 'Entity', 'COMPS', 'Reading', 'SuperGLUE']`

| component | score | source | path |
|---|---:|---|---|
| BLiMP | 68.09 | A01_step109_or_step110 | `experiments/archive/functional_learning/data/parallel_eval/ms62065_BLiMP/result_BLiMP.json` |
| Supplement | 63.09034033118927 | A02_step127 | `experiments/archive/relation_learning/data/ms62065_Supplement/ms62065/with_special/Supplement_component_payload.json` |
| EWoK | - | - | `` |
| Entity | - | - | `` |
| COMPS | - | - | `` |
| GlobalPIQA_parallel | 30.1 | A01_step109_or_step110 | `experiments/archive/functional_learning/data/parallel_zero_reading/ms62065_GlobalPIQA_parallel/result_GlobalPIQA_parallel.json` |
| GlobalPIQA_nonparallel | 50.0 | A01_step109_or_step110 | `experiments/archive/functional_learning/data/parallel_zero_reading/ms62065_GlobalPIQA_nonparallel/result_GlobalPIQA_nonparallel.json` |
| GlobalPIQA mean | 40.05 | derived | - |
| Reading | - | - | `` |
| AoA | 0.0 | A01_step110 | `experiments/archive/functional_learning/data/batched_aoa_measured/ms62065/full/aoa_manifest.json` |
| SuperGLUE mean | None | derived from tasks | - |

SuperGLUE tasks: boolq=None, multirc=None, rte=None, wsc=63.46153846153846, mrpc=None, qqp=None, mnli=None
