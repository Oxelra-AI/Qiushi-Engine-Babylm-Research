# chck84 carrier and scale125 grid synthesis chck_84M native-validated full-evaluation carrier

Status: **CHCK84_NATIVE_VALIDATED_FULL_CARRIER_READY**

Carrier: `experiments/archive/frontier_consolidation/data/chck84_native_validated_full_carrier/all_full_preds_truthful_chck84_mlm.json`
Carrier SHA256: `b55e1997c3fbc0255527f35c8225f788fd1dae6a1427b13b142a13ed9e64189d`
Native prediction validation: `True` / `Upload successful.`

## Score arithmetic from existing official-compatible records

| column | score |
|---|---:|
| BLiMP | 68.25 |
| Supplement | 63.48 |
| EWoK | 50.07 |
| Entity | 28.58 |
| COMPS | 52.21 |
| GlobalPIQA | 38.12 |
| Reading | 8.155 |
| SuperGLUE | 69.305216768 |
| AoA | 0 |

Cheap7: `44.12357142857143`; Overall(AoA0): `42.0189129742181`.
Delta vs protected chck_82M in Overall(AoA0): `+0.076431806832`.

## Contents

- Seven cheap-task prediction blocks come from the chck_84M selected evaluation payload.
- SuperGLUE prediction blocks come from the chck_84M SuperGLUE-only payload.
- `aoa` is scalar `0.0`.
- No `fast_eval_results` block is included.

## Scientific use

This submission package preserves the exact legal chck_84M endpoint predictions for reproducibility and submission preparation. It is not a new model result and it does not establish a transferable learning principle.

Manifest JSON: `experiments/archive/frontier_consolidation/data/chck84_native_validated_full_carrier/chck84_truthful_full_carrier_manifest.json`
