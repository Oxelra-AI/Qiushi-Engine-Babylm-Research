# earlier analysis strict split-evaluation admission

Created: `2026-09-08T09:47:05Z`

This pass admits split paired-seed components only after reconciling the score surface. For sentence zero-shot columns the admitted value is parsed from `best_temperature_report.txt`, which is the established two-decimal report surface. For Reading, the admitted value is the mean of the report-displayed eye and self-paced scores. Raw custom-scorer precision is retained in JSON/CSV but is not mixed into the same-coordinate Overall.

## o62065 — ordinary_inherited_wwm_seed62065

Complete for Overall: `False`; Overall: `None`; missing: `['BLiMP', 'COMPS', 'SuperGLUE']`; conflicts: `[]`

| component | admitted score | chosen source | chosen path | notes |
|---|---:|---|---|---|
| BLiMP | - | - | `` |  |
| Supplement | 63.620000000000 | A02_custom_single_component | `experiments/archive/relation_learning/data/o62065_Supplement/o62065/with_special/Supplement_component_payload.json` | no_predictions_for_item_analysis;score_coverage_uses_record_count |
| EWoK | 49.810000000000 | A02_custom_single_component | `experiments/archive/relation_learning/data/o62065_EWoK/o62065/with_special/EWoK_component_payload.json` | no_predictions_for_item_analysis;score_coverage_uses_record_count |
| Entity | 28.090000000000 | A02_custom_single_component | `experiments/archive/relation_learning/data/o62065_Entity/o62065/with_special/Entity_component_payload.json` | no_predictions_for_item_analysis;score_coverage_uses_record_count |
| COMPS | - | - | `` |  |
| GlobalPIQA_parallel | 31.070000000000 | A02_custom_single_component | `experiments/archive/relation_learning/data/o62065_GlobalPIQA_parallel/o62065/with_special/GlobalPIQA_parallel_component_payload.json` |  |
| GlobalPIQA_nonparallel | 48.000000000000 | A02_custom_single_component | `experiments/archive/relation_learning/data/o62065_GlobalPIQA_nonparallel/o62065/with_special/GlobalPIQA_nonparallel_component_payload.json` |  |
| Reading | 8.195000000000 | A02_custom_single_component | `experiments/archive/relation_learning/data/o62065_Reading/o62065/with_special/Reading_component_payload.json` |  |
| GlobalPIQA mean | 39.535000000000 | derived | - | mean of report-derived subcolumns |
| AoA | 0.000000000000 | A01_step113_from_A02_extract | `experiments/archive/functional_learning/data/o62065_aoa_measured_from_extract/o62065/full/aoa_manifest.json` | measured trajectory |
| SuperGLUE mean | None | derived | - | primary metrics: {"boolq": 67.4006116207951, "multirc": null, "rte": 64.02877697841727, "wsc": 65.38461538461539, "mrpc": 87.58620689655172, "qqp": 71.6198210773428, "mnli": 60.45232273838631} |

## ms62065 — ms_acquisition_seed62065

Complete for Overall: `True`; Overall: `42.17887384024569`; missing: `[]`; conflicts: `[]`

| component | admitted score | chosen source | chosen path | notes |
|---|---:|---|---|---|
| BLiMP | 68.090000000000 | A01_official_entry | `experiments/archive/functional_learning/data/parallel_eval/ms62065_BLiMP/result_BLiMP.json` |  |
| Supplement | 63.090000000000 | A02_custom_single_component | `experiments/archive/relation_learning/data/ms62065_Supplement/ms62065/with_special/Supplement_component_payload.json` | no_predictions_for_item_analysis;score_coverage_uses_record_count |
| EWoK | 49.820000000000 | A02_custom_single_component | `experiments/archive/relation_learning/data/ms62065_EWoK/ms62065/with_special/EWoK_component_payload.json` | no_predictions_for_item_analysis;score_coverage_uses_record_count |
| Entity | 29.290000000000 | A01_official_entry | `experiments/archive/functional_learning/data/parallel_zero_reading/ms62065_Entity/result_Entity.json` |  |
| COMPS | 52.140000000000 | A01_official_entry | `experiments/archive/functional_learning/data/parallel_zero_reading/ms62065_COMPS/result_COMPS.json` |  |
| GlobalPIQA_parallel | 30.100000000000 | A01_official_entry | `experiments/archive/functional_learning/data/parallel_zero_reading/ms62065_GlobalPIQA_parallel/result_GlobalPIQA_parallel.json` |  |
| GlobalPIQA_nonparallel | 50.000000000000 | A01_official_entry | `experiments/archive/functional_learning/data/parallel_zero_reading/ms62065_GlobalPIQA_nonparallel/result_GlobalPIQA_nonparallel.json` |  |
| Reading | 8.195000000000 | A01_official_entry | `experiments/archive/functional_learning/data/parallel_zero_reading/ms62065_Reading/result_Reading.json` |  |
| GlobalPIQA mean | 40.050000000000 | derived | - | mean of report-derived subcolumns |
| AoA | 0.000000000000 | A01_step110 | `experiments/archive/functional_learning/data/batched_aoa_measured/ms62065/full/aoa_manifest.json` | measured trajectory |
| SuperGLUE mean | 68.934864562211 | derived | - | primary metrics: {"boolq": 67.95107033639144, "multirc": 68.23432343234323, "rte": 63.30935251798561, "wsc": 63.46153846153846, "mrpc": 87.88927335640139, "qqp": 71.40917027743646, "mnli": 60.28932355338223} |

## Paired contrasts against seed62064 counterparts

```json
{
  "o62065_minus_ordinary_inherited_wwm_seed62064": {
    "available": false,
    "missing_seed62065": [
      "BLiMP",
      "COMPS",
      "SuperGLUE"
    ],
    "seed62064_available": true
  },
  "ms62065_minus_densemask_sparselabel_seed62064": {
    "available": true,
    "delta_overall": -0.023664114090834687,
    "component_deltas": {
      "BLiMP": -0.030000000000001137,
      "Supplement": 0.010000000000005116,
      "EWoK": -0.13000000000000256,
      "Entity": -0.10000000000000142,
      "COMPS": -0.00999999999999801,
      "SuperGLUE": 0.04702297318249293,
      "GlobalPIQA": 0.0,
      "Reading": 0.0,
      "AoA": 0.0
    },
    "seed62065_overall": 42.17887384024569,
    "seed62064_overall": 42.20253795433653
  },
  "anchoring_associated_residual_seed_comparison": {
    "seed62064_clean_minus_ms": 0.04387437787291759,
    "seed62065_clean_minus_ms": 0.05285729241231962,
    "difference_seed65_minus_seed64": 0.008982914539402032,
    "available": true
  }
}
```

## Current unresolved admission work

- o62065: missing ['BLiMP', 'COMPS', 'SuperGLUE'], conflicts []
