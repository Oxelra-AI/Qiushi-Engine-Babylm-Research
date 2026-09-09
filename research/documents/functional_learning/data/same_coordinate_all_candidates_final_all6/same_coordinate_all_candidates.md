# earlier analysis same-coordinate table for replication and (M,S) comparison

Only complete compatible sources enter Overall arithmetic: full official-sized zero-shot/Reading, repaired AutoModel SuperGLUE primary metrics, and measured batched AoA.

## Model summary

### coherent86
- complete same-coordinate: `True`
- Overall: `42.023967991315104`
- delta vs coherent86: `0.0`
- components: `{"BLiMP": 68.51, "Supplement": 63.64, "EWoK": 50.02, "Entity": 28.32, "COMPS": 52.05, "SuperGLUE": 68.94571192183594, "GlobalPIQA": 38.565, "Reading": 8.165, "AoA": 0.0}`

### dense_seed62064
- complete same-coordinate: `True`
- Overall: `42.14909111936738`
- delta vs coherent86: `0.1251231280522731`
- components: `{"BLiMP": 68.06, "Supplement": 63.04, "EWoK": 49.92, "Entity": 29.37, "COMPS": 52.15, "SuperGLUE": 68.5318200743064, "GlobalPIQA": 40.05, "Reading": 8.219999999999999, "AoA": 0.0}`

### dense_seed62065
- complete same-coordinate: `True`
- Overall: `42.168422702303516`
- delta vs coherent86: `0.14445471098841267`
- components: `{"BLiMP": 68.02, "Supplement": 63.09, "EWoK": 49.77, "Entity": 29.42, "COMPS": 52.15, "SuperGLUE": 68.80580432073164, "GlobalPIQA": 40.05, "Reading": 8.21, "AoA": 0.0}`

### clean_pres_lambda1_eval_seed62064
- complete same-coordinate: `True`
- Overall: `42.246412332209445`
- delta vs coherent86: `0.2224443408943415`
- components: `{"BLiMP": 68.26, "Supplement": 63.28, "EWoK": 49.82, "Entity": 29.4, "COMPS": 52.16, "SuperGLUE": 69.047710989885, "GlobalPIQA": 40.05, "Reading": 8.2, "AoA": 0.0}`

### clean_pres_lambda1_eval_seed62065
- complete same-coordinate: `True`
- Overall: `42.23173113265801`
- delta vs coherent86: `0.20776314134290885`
- components: `{"BLiMP": 68.22, "Supplement": 63.29, "EWoK": 49.72, "Entity": 29.45, "COMPS": 52.14, "SuperGLUE": 69.02058019392214, "GlobalPIQA": 40.05, "Reading": 8.195, "AoA": 0.0}`

### densemask_sparselabel_seed62064
- complete same-coordinate: `True`
- Overall: `42.20253795433653`
- delta vs coherent86: `0.17856996302142392`
- components: `{"BLiMP": 68.12, "Supplement": 63.08, "EWoK": 49.95, "Entity": 29.39, "COMPS": 52.15, "SuperGLUE": 68.88784158902877, "GlobalPIQA": 40.05, "Reading": 8.195, "AoA": 0.0}`

## Method comparisons

{
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
    },
    "interpretation": "This is the direct preservation increment only after the exact (M,S) zero/Reading, SuperGLUE, and AoA sources are complete; before then it remains an awaiting comparison."
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
    },
    "interpretation": "This evaluates fixed-policy seed-level official replication; mechanism/fast replication is not enough without complete official components."
  }
}

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
  }
}
