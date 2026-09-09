# earlier analysis same-coordinate table with ordinary-control rung

Only complete compatible sources enter Overall arithmetic: full official-sized zero-shot/Reading, repaired AutoModel SuperGLUE primary metrics, and measured batched AoA.

## Model summary

### coherent86
- complete same-coordinate: `True`
- Overall: `42.023967991315104`
- delta vs coherent86: `0.0`
- present components: `{'present_components': ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'SuperGLUE', 'GlobalPIQA', 'Reading', 'AoA'], 'missing_components': [], 'sum_present': 378.215711921836, 'n_present': 9}`
- components: `{"BLiMP": 68.51, "Supplement": 63.64, "EWoK": 50.02, "Entity": 28.32, "COMPS": 52.05, "SuperGLUE": 68.94571192183594, "GlobalPIQA": 38.565, "Reading": 8.165, "AoA": 0.0}`

### dense_seed62064
- complete same-coordinate: `True`
- Overall: `42.14909111936738`
- delta vs coherent86: `0.1251231280522731`
- present components: `{'present_components': ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'SuperGLUE', 'GlobalPIQA', 'Reading', 'AoA'], 'missing_components': [], 'sum_present': 379.3418200743064, 'n_present': 9}`
- components: `{"BLiMP": 68.06, "Supplement": 63.04, "EWoK": 49.92, "Entity": 29.37, "COMPS": 52.15, "SuperGLUE": 68.5318200743064, "GlobalPIQA": 40.05, "Reading": 8.219999999999999, "AoA": 0.0}`

### dense_seed62065
- complete same-coordinate: `True`
- Overall: `42.168422702303516`
- delta vs coherent86: `0.14445471098841267`
- present components: `{'present_components': ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'SuperGLUE', 'GlobalPIQA', 'Reading', 'AoA'], 'missing_components': [], 'sum_present': 379.5158043207316, 'n_present': 9}`
- components: `{"BLiMP": 68.02, "Supplement": 63.09, "EWoK": 49.77, "Entity": 29.42, "COMPS": 52.15, "SuperGLUE": 68.80580432073164, "GlobalPIQA": 40.05, "Reading": 8.21, "AoA": 0.0}`

### clean_pres_lambda1_eval_seed62064
- complete same-coordinate: `True`
- Overall: `42.246412332209445`
- delta vs coherent86: `0.2224443408943415`
- present components: `{'present_components': ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'SuperGLUE', 'GlobalPIQA', 'Reading', 'AoA'], 'missing_components': [], 'sum_present': 380.217710989885, 'n_present': 9}`
- components: `{"BLiMP": 68.26, "Supplement": 63.28, "EWoK": 49.82, "Entity": 29.4, "COMPS": 52.16, "SuperGLUE": 69.047710989885, "GlobalPIQA": 40.05, "Reading": 8.2, "AoA": 0.0}`

### clean_pres_lambda1_eval_seed62065
- complete same-coordinate: `True`
- Overall: `42.23173113265801`
- delta vs coherent86: `0.20776314134290885`
- present components: `{'present_components': ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'SuperGLUE', 'GlobalPIQA', 'Reading', 'AoA'], 'missing_components': [], 'sum_present': 380.08558019392217, 'n_present': 9}`
- components: `{"BLiMP": 68.22, "Supplement": 63.29, "EWoK": 49.72, "Entity": 29.45, "COMPS": 52.14, "SuperGLUE": 69.02058019392214, "GlobalPIQA": 40.05, "Reading": 8.195, "AoA": 0.0}`

### densemask_sparselabel_seed62064
- complete same-coordinate: `True`
- Overall: `42.20253795433653`
- delta vs coherent86: `0.17856996302142392`
- present components: `{'present_components': ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'SuperGLUE', 'GlobalPIQA', 'Reading', 'AoA'], 'missing_components': [], 'sum_present': 379.8228415890287, 'n_present': 9}`
- components: `{"BLiMP": 68.12, "Supplement": 63.08, "EWoK": 49.95, "Entity": 29.39, "COMPS": 52.15, "SuperGLUE": 68.88784158902877, "GlobalPIQA": 40.05, "Reading": 8.195, "AoA": 0.0}`

### ordinary_inherited_wwm_seed62064
- complete same-coordinate: `False`
- Overall: `None`
- delta vs coherent86: `None`
- present components: `{'present_components': ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA', 'Reading', 'AoA'], 'missing_components': ['SuperGLUE'], 'sum_present': 309.845, 'n_present': 8}`
- components: `{"BLiMP": 68.47, "Supplement": 63.63, "EWoK": 49.83, "Entity": 28.16, "COMPS": 52.0, "SuperGLUE": null, "GlobalPIQA": 39.535, "Reading": 8.22, "AoA": 0.0}`
- blocking errors: `missing_superglue_subtask:multirc`, `missing_superglue_subtask:rte`, `missing_superglue_subtask:wsc`, `missing_superglue_subtask:mrpc`, `missing_superglue_subtask:qqp`, `missing_superglue_subtask:mnli`, `missing_primary_metric_detail:boolq`, `missing_primary_metric_detail:multirc`, `missing_primary_metric_detail:rte`, `missing_primary_metric_detail:wsc`, `missing_primary_metric_detail:mrpc`, `missing_primary_metric_detail:qqp`, `missing_primary_metric_detail:mnli`, `missing_component:SuperGLUE`

## Method comparisons

{
  "ordinary_seed62064_vs_coherent86": {
    "available": false,
    "overall_delta_ordinary_minus_coherent86": null,
    "component_deltas_ordinary_minus_coherent86": {
      "BLiMP": -0.04000000000000625,
      "Supplement": -0.00999999999999801,
      "EWoK": -0.19000000000000483,
      "Entity": -0.16000000000000014,
      "COMPS": -0.04999999999999716,
      "SuperGLUE": null,
      "GlobalPIQA": 0.9699999999999989,
      "Reading": 0.05500000000000149,
      "AoA": 0.0
    },
    "present_component_summary": {
      "present_components": [
        "BLiMP",
        "Supplement",
        "EWoK",
        "Entity",
        "COMPS",
        "GlobalPIQA",
        "Reading",
        "AoA"
      ],
      "missing_components": [
        "SuperGLUE"
      ],
      "sum_present": 309.845,
      "n_present": 8
    },
    "interpretation": "Parent -> ordinary measures additional ordinary WWM training on the same unchanged-Qwen rows. Until SuperGLUE and AoA are both present, this is only a partial component comparison."
  },
  "ordinary_seed62064_vs_exact_ms_seed62064": {
    "available": false,
    "overall_delta_ordinary_minus_ms": null,
    "component_deltas_ordinary_minus_ms": {
      "BLiMP": 0.3499999999999943,
      "Supplement": 0.5500000000000043,
      "EWoK": -0.12000000000000455,
      "Entity": -1.2300000000000004,
      "COMPS": -0.14999999999999858,
      "SuperGLUE": null,
      "GlobalPIQA": -0.5150000000000006,
      "Reading": 0.025000000000000355,
      "AoA": 0.0
    },
    "interpretation": "Ordinary -> (M,S) changes input corruption and target/credit allocation together, so it complements rather than replaces the sharper (S,S)->(M,S) masking contrast."
  },
  "clean_seed62064_vs_exact_ms_seed62064": {
    "available": true,
    "overall_delta_clean_minus_ms": 0.04387437787291759,
    "component_deltas_clean_minus_ms": {
      "BLiMP": 0.14000000000000057,
      "Supplement": 0.20000000000000284,
      "EWoK": -0.13000000000000256,
      "Entity": 0.00999999999999801,
      "COMPS": 0.00999999999999801,
      "SuperGLUE": 0.15986940085623758,
      "GlobalPIQA": 0.0,
      "Reading": 0.004999999999999005,
      "AoA": 0.0
    }
  },
  "clean_seed62065_replication_vs_clean_seed62064": {
    "available": true,
    "overall_delta_seed65_minus_seed64": -0.014681199551432655,
    "component_deltas_seed65_minus_seed64": {
      "BLiMP": -0.04000000000000625,
      "Supplement": 0.00999999999999801,
      "EWoK": -0.10000000000000142,
      "Entity": 0.05000000000000071,
      "COMPS": -0.01999999999999602,
      "SuperGLUE": -0.027130795962861498,
      "GlobalPIQA": 0.0,
      "Reading": -0.004999999999999005,
      "AoA": 0.0
    }
  }
}

## Contrast graph

- coherent86 -> ordinary_inherited_wwm_seed62064: extra ordinary WWM continuation on same unchanged-Qwen rows
- ordinary_inherited_wwm_seed62064 -> densemask_sparselabel_seed62064: joint change in Qwen input corruption and sparse focus target/credit allocation
- sparse (S,S) -> densemask_sparselabel (M,S): sharper effective-input masking contrast at fixed sparse labels/focus weighting
- densemask_sparselabel_seed62064 -> clean_pres_lambda1_eval_seed62064: ordinary full-row parent KL plus counted preservation presentations

## GlobalPIQA-zero sensitivity

{
  "coherent86": {
    "complete": true,
    "official_delta_vs_coherent86": 0.0,
    "delta_vs_coherent86_with_GlobalPIQA_difference_zeroed": 0.0
  },
  "dense_seed62064": {
    "complete": true,
    "official_delta_vs_coherent86": 0.1251231280522731,
    "delta_vs_coherent86_with_GlobalPIQA_difference_zeroed": -0.03987687194772723
  },
  "dense_seed62065": {
    "complete": true,
    "official_delta_vs_coherent86": 0.14445471098841267,
    "delta_vs_coherent86_with_GlobalPIQA_difference_zeroed": -0.020545289011588846
  },
  "clean_pres_lambda1_eval_seed62064": {
    "complete": true,
    "official_delta_vs_coherent86": 0.2224443408943415,
    "delta_vs_coherent86_with_GlobalPIQA_difference_zeroed": 0.05744434089434039
  },
  "clean_pres_lambda1_eval_seed62065": {
    "complete": true,
    "official_delta_vs_coherent86": 0.20776314134290885,
    "delta_vs_coherent86_with_GlobalPIQA_difference_zeroed": 0.04276314134291089
  },
  "densemask_sparselabel_seed62064": {
    "complete": true,
    "official_delta_vs_coherent86": 0.17856996302142392,
    "delta_vs_coherent86_with_GlobalPIQA_difference_zeroed": 0.013569963021425558
  },
  "ordinary_inherited_wwm_seed62064": {
    "complete": false,
    "official_delta_vs_coherent86": null,
    "delta_vs_coherent86_with_GlobalPIQA_difference_zeroed": null
  }
}
