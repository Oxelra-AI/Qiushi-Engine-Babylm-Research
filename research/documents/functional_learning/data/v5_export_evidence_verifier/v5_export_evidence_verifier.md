# evidence hardening and pending ftseed44 source-level evidence verifier for the v5 candidate

Created: `2026-09-08T05:11:17Z`

All source-level validations passed: `true`

## Overall values recomputed from source payloads

| model | recomputed Overall | delta vs coherent86 | source payloads complete |
|---|---:|---:|---:|
| coherent86 | 42.023967991315 | 0.000000000000 | True |
| dense_seed62064_MM | 42.149091119367 | 0.125123128052 | True |
| dense_seed62065_MM | 42.168422702304 | 0.144454710988 | True |
| clean_pres_seed62064_MSplusKL | 42.246412332209 | 0.222444340894 | True |
| clean_pres_seed62065_MSplusKL | 42.231731132658 | 0.207763141343 | True |
| ms_acquisition_seed62064_MS | 42.202537954337 | 0.178569963021 | True |

## Direct clean seed62064 versus exact `(M,S)`

- clean64 minus exact `(M,S)` Overall: `0.043874377873`
- component deltas:
  - BLiMP: `0.140000000000`
  - Supplement: `0.200000000000`
  - EWoK: `-0.130000000000`
  - Entity: `0.010000000000`
  - COMPS: `0.010000000000`
  - SuperGLUE: `0.159869400856`
  - GlobalPIQA: `0.000000000000`
  - Reading: `0.005000000000`
  - AoA: `0.000000000000`

## SuperGLUE source rows

### coherent86: SuperGLUE `68.945711921836`
- boolq accuracy `67.278287`, returncode `0`, predictions exists `True`
- multirc accuracy `68.275578`, returncode `0`, predictions exists `True`
- rte accuracy `64.028777`, returncode `0`, predictions exists `True`
- wsc accuracy `63.461538`, returncode `0`, predictions exists `True`
- mrpc f1 `87.586207`, returncode `0`, predictions exists `True`
- qqp f1 `71.557648`, returncode `0`, predictions exists `True`
- mnli accuracy `60.431948`, returncode `0`, predictions exists `True`
### ms_acquisition_seed62064_MS: SuperGLUE `68.887841589029`
- boolq accuracy `67.951070`, returncode `0`, predictions exists `True`
- multirc accuracy `67.904290`, returncode `0`, predictions exists `True`
- rte accuracy `63.309353`, returncode `0`, predictions exists `True`
- wsc accuracy `63.461538`, returncode `0`, predictions exists `True`
- mrpc f1 `87.889273`, returncode `0`, predictions exists `True`
- qqp f1 `71.430417`, returncode `0`, predictions exists `True`
- mnli accuracy `60.268949`, returncode `0`, predictions exists `True`
### clean_pres_seed62064_MSplusKL: SuperGLUE `69.047710989885`
- boolq accuracy `67.951070`, returncode `0`, predictions exists `True`
- multirc accuracy `68.151815`, returncode `0`, predictions exists `True`
- rte accuracy `63.309353`, returncode `0`, predictions exists `True`
- wsc accuracy `63.461538`, returncode `0`, predictions exists `True`
- mrpc f1 `88.275862`, returncode `0`, predictions exists `True`
- qqp f1 `71.589391`, returncode `0`, predictions exists `True`
- mnli accuracy `60.594947`, returncode `0`, predictions exists `True`
### clean_pres_seed62065_MSplusKL: SuperGLUE `69.020580193922`
- boolq accuracy `67.767584`, returncode `0`, predictions exists `True`
- multirc accuracy `68.193069`, returncode `0`, predictions exists `True`
- rte accuracy `63.309353`, returncode `0`, predictions exists `True`
- wsc accuracy `63.461538`, returncode `0`, predictions exists `True`
- mrpc f1 `88.275862`, returncode `0`, predictions exists `True`
- qqp f1 `71.562083`, returncode `0`, predictions exists `True`
- mnli accuracy `60.574572`, returncode `0`, predictions exists `True`

## AutoModel records

{
  "evaluation repair synthesis": {
    "path": "experiments/archive/functional_learning/data/real_interface_validation/real_interface_validation.json",
    "interface_validated": true,
    "all_repaired_valid": true,
    "all_negatives_stock": true,
    "coherent86_repaired": true,
    "coherent86_hidden_exact": true,
    "dense64_repaired": true,
    "dense65_repaired": true
  },
  "earlier analysis": {
    "path": "experiments/archive/functional_learning/data/automodel_repair_clean_preservation/repair_validation.json",
    "all_valid": true,
    "labels_valid": {
      "clean_pres_lambda1_eval_seed62064_u0080": true
    },
    "hidden_exact": {
      "clean_pres_lambda1_eval_seed62064_u0080": true
    }
  },
  "earlier analysis": {
    "path": "experiments/archive/functional_learning/data/automodel_repair_candidates/repair_validation.json",
    "all_valid": true,
    "labels_valid": {
      "clean_pres_lambda1_eval_seed62065_u0080": true,
      "densemask_sparselabel_seed62064_u0080": true
    },
    "hidden_exact": {
      "clean_pres_lambda1_eval_seed62065_u0080": true,
      "densemask_sparselabel_seed62064_u0080": true
    }
  }
}

## Interpretation

The verifier recomputes the same all-six ordering from the actual official payloads: coherent86 < exact `(M,S)` < clean seed62065 < clean seed62064, with dense `(M,M)` endpoints in between parent and exact `(M,S)`. The direct preservation-specific score increment remains small (`+0.043874...` Overall), while the complete clean policy gain over the repaired parent remains larger (`+0.222444...` for seed62064 and `+0.207763...` for seed62065). These facts support preserving clean seed62064 as the candidate endpoint while keeping the mechanistic account narrow: acquisition supplies most movement, and deterministic anchoring adds partial retention/transfer repair.
