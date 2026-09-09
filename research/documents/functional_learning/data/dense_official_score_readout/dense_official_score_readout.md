# dense focus official and mechanism state dense official score readout

Dense payload: `experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json`
Reference zero/Reading: `experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json`
Reference SuperGLUE: `experiments/archive/frontier_consolidation/data/private_scale_superglue_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75_sg_retry.json`

| column | coherent86 | dense seen | dense-reference | present | complete |
|---|---:|---:|---:|---:|---:|
| BLiMP | 68.510000 | 68.060000 | -0.450000 | True | True |
| Supplement | 63.640000 | 63.040000 | -0.600000 | True | True |
| EWoK | 50.020000 | 49.920000 | -0.100000 | True | True |
| Entity | 28.320000 | 29.370000 | +1.050000 | True | True |
| COMPS | 52.050000 |  |  | False | False |
| SuperGLUE | 69.819222 |  |  | False | False |
| GlobalPIQA | 38.565000 |  |  | False | False |
| Reading | 8.165000 |  |  | False | False |
| AoA | 0.000000 |  |  | False | False |

## Arithmetic state

- Known non-AoA delta sum: -0.100000 over ['BLiMP', 'Supplement', 'EWoK', 'Entity'].
- Unknown non-AoA columns: ['COMPS', 'SuperGLUE', 'GlobalPIQA', 'Reading'].
- Remaining total delta needed to equal coherent86 Overall(AoA0): +0.100000, mean +0.025000 over each unknown non-AoA column.
- Dense projected Overall(AoA0) is not available until all non-AoA columns are present.

Unknown columns are not estimated here. Use this file to read the official payload as it actually exists.
