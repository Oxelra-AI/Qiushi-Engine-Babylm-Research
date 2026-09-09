# earlier analysis strict split-evaluation admission

Created: `2026-09-08T09:02:34Z`

This pass admits split paired-seed components only after reconciling the score surface. For sentence zero-shot columns the admitted value is parsed from `best_temperature_report.txt`, which is the established two-decimal report surface. For Reading, the admitted value is the mean of the report-displayed eye and self-paced scores. Raw custom-scorer precision is retained in JSON/CSV but is not mixed into the same-coordinate Overall.

## o62065 — ordinary_inherited_wwm_seed62065

Complete for Overall: `False`; Overall: `None`; missing: `['BLiMP', 'Entity', 'COMPS', 'SuperGLUE', 'AoA']`; conflicts: `[]`

| component | admitted score | chosen source | chosen path | notes |
|---|---:|---|---|---|
| BLiMP | - | - | `` |  |
| Supplement | 63.620000000000 | A02_custom_single_component | `experiments/archive/relation_learning/data/o62065_Supplement/o62065/with_special/Supplement_component_payload.json` | no_predictions_for_item_analysis;score_coverage_uses_record_count |
| EWoK | 49.810000000000 | A02_custom_single_component | `experiments/archive/relation_learning/data/o62065_EWoK/o62065/with_special/EWoK_component_payload.json` | no_predictions_for_item_analysis;score_coverage_uses_record_count |
| Entity | - | - | `` |  |
| COMPS | - | - | `` |  |
| GlobalPIQA_parallel | 31.070000000000 | A02_custom_single_component | `experiments/archive/relation_learning/data/o62065_GlobalPIQA_parallel/o62065/with_special/GlobalPIQA_parallel_component_payload.json` |  |
| GlobalPIQA_nonparallel | 48.000000000000 | A02_custom_single_component | `experiments/archive/relation_learning/data/o62065_GlobalPIQA_nonparallel/o62065/with_special/GlobalPIQA_nonparallel_component_payload.json` |  |
| Reading | 8.195000000000 | A02_custom_single_component | `experiments/archive/relation_learning/data/o62065_Reading/o62065/with_special/Reading_component_payload.json` |  |
| GlobalPIQA mean | 39.535000000000 | derived | - | mean of report-derived subcolumns |
| AoA | None | - | `` | measured trajectory |
| SuperGLUE mean | None | derived | - | primary metrics: {"boolq": null, "multirc": null, "rte": null, "wsc": null, "mrpc": null, "qqp": null, "mnli": null} |

## ms62065 — ms_acquisition_seed62065

Complete for Overall: `False`; Overall: `None`; missing: `['Entity', 'COMPS', 'SuperGLUE', 'AoA']`; conflicts: `[]`

| component | admitted score | chosen source | chosen path | notes |
|---|---:|---|---|---|
| BLiMP | 68.090000000000 | A01_official_entry | `experiments/archive/functional_learning/data/parallel_eval/ms62065_BLiMP/result_BLiMP.json` |  |
| Supplement | 63.090000000000 | A02_custom_single_component | `experiments/archive/relation_learning/data/ms62065_Supplement/ms62065/with_special/Supplement_component_payload.json` | no_predictions_for_item_analysis;score_coverage_uses_record_count |
| EWoK | 49.820000000000 | A02_custom_single_component | `experiments/archive/relation_learning/data/ms62065_EWoK/ms62065/with_special/EWoK_component_payload.json` | no_predictions_for_item_analysis;score_coverage_uses_record_count |
| Entity | - | - | `` |  |
| COMPS | - | - | `` |  |
| GlobalPIQA_parallel | 30.100000000000 | A01_official_entry | `experiments/archive/functional_learning/data/parallel_zero_reading/ms62065_GlobalPIQA_parallel/result_GlobalPIQA_parallel.json` |  |
| GlobalPIQA_nonparallel | 50.000000000000 | A01_official_entry | `experiments/archive/functional_learning/data/parallel_zero_reading/ms62065_GlobalPIQA_nonparallel/result_GlobalPIQA_nonparallel.json` |  |
| Reading | 8.195000000000 | A01_official_entry | `experiments/archive/functional_learning/data/parallel_zero_reading/ms62065_Reading/result_Reading.json` |  |
| GlobalPIQA mean | 40.050000000000 | derived | - | mean of report-derived subcolumns |
| AoA | None | - | `` | measured trajectory |
| SuperGLUE mean | None | derived | - | primary metrics: {"boolq": null, "multirc": null, "rte": 63.30935251798561, "wsc": 63.46153846153846, "mrpc": 87.88927335640139, "qqp": 71.40917027743646, "mnli": 60.28932355338223} |

## Paired contrasts against seed62064 counterparts

```json
{
  "o62065_minus_ordinary_inherited_wwm_seed62064": {
    "available": false,
    "missing_seed62065": [
      "BLiMP",
      "Entity",
      "COMPS",
      "SuperGLUE",
      "AoA"
    ],
    "seed62064_available": true
  },
  "ms62065_minus_densemask_sparselabel_seed62064": {
    "available": false,
    "missing_seed62065": [
      "Entity",
      "COMPS",
      "SuperGLUE",
      "AoA"
    ],
    "seed62064_available": true
  },
  "anchoring_associated_residual_seed_comparison": {
    "seed62064_clean_minus_ms": 0.04387437787291759,
    "seed62065_clean_minus_ms": null,
    "difference_seed65_minus_seed64": null,
    "available": false
  }
}
```

## Current unresolved admission work

- o62065: missing ['BLiMP', 'Entity', 'COMPS', 'SuperGLUE', 'AoA'], conflicts []
- ms62065: missing ['Entity', 'COMPS', 'SuperGLUE', 'AoA'], conflicts []
