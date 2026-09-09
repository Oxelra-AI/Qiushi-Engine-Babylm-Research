# earlier analysis same-coordinate comparison

This file combines only compatible full official-sized zero-shot/Reading, repaired AutoModel SuperGLUE, and measured AoA sources. Fast-screen scores and historical stripped-AutoModel values are not mixed into this coordinate.

## Historical coordinate kept separate

{
  "coherent86_historical_platform_style_overall": {
    "Overall": 42.1210247099666,
    "interpretation": "Preserved historical uploaded/platform-style coordinate; not used for repaired-loading scientific comparisons because historical SuperGLUE used stock AutoModel and omitted private adapters."
  }
}

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
- complete same-coordinate: `False`
- Overall: `None`
- delta vs coherent86: `None`
- components: `{"BLiMP": null, "Supplement": null, "EWoK": null, "Entity": null, "COMPS": null, "SuperGLUE": null, "GlobalPIQA": null, "Reading": null, "AoA": null}`
- blocking errors: `payload_not_loaded`, `payload_not_loaded`, `payload_not_loaded`, `missing_component:BLiMP`, `missing_component:Supplement`, `missing_component:EWoK`, `missing_component:Entity`, `missing_component:COMPS`, `missing_component:SuperGLUE`, `missing_component:GlobalPIQA`, `missing_component:Reading`, `missing_component:AoA`

## Delta table

{
  "coherent86": {
    "overall": 42.023967991315104,
    "delta_vs_coherent86": 0.0,
    "component_deltas_vs_coherent86": {
      "BLiMP": 0.0,
      "Supplement": 0.0,
      "EWoK": 0.0,
      "Entity": 0.0,
      "COMPS": 0.0,
      "SuperGLUE": 0.0,
      "GlobalPIQA": 0.0,
      "Reading": 0.0,
      "AoA": 0.0
    },
    "complete": true
  },
  "dense_seed62064": {
    "overall": 42.14909111936738,
    "delta_vs_coherent86": 0.1251231280522731,
    "component_deltas_vs_coherent86": {
      "BLiMP": -0.45000000000000284,
      "Supplement": -0.6000000000000014,
      "EWoK": -0.10000000000000142,
      "Entity": 1.0500000000000007,
      "COMPS": 0.10000000000000142,
      "SuperGLUE": -0.4138918475295412,
      "GlobalPIQA": 1.4849999999999994,
      "Reading": 0.054999999999999716,
      "AoA": 0.0
    },
    "complete": true
  },
  "dense_seed62065": {
    "overall": 42.168422702303516,
    "delta_vs_coherent86": 0.14445471098841267,
    "component_deltas_vs_coherent86": {
      "BLiMP": -0.4900000000000091,
      "Supplement": -0.5499999999999972,
      "EWoK": -0.25,
      "Entity": 1.1000000000000014,
      "COMPS": 0.10000000000000142,
      "SuperGLUE": -0.1399076011042979,
      "GlobalPIQA": 1.4849999999999994,
      "Reading": 0.045000000000001705,
      "AoA": 0.0
    },
    "complete": true
  },
  "clean_pres_lambda1_eval_seed62064": {
    "overall": 42.246412332209445,
    "delta_vs_coherent86": 0.2224443408943415,
    "component_deltas_vs_coherent86": {
      "BLiMP": -0.25,
      "Supplement": -0.35999999999999943,
      "EWoK": -0.20000000000000284,
      "Entity": 1.0799999999999983,
      "COMPS": 0.10999999999999943,
      "SuperGLUE": 0.10199906804906789,
      "GlobalPIQA": 1.4849999999999994,
      "Reading": 0.03500000000000014,
      "AoA": 0.0
    },
    "complete": true
  },
  "clean_pres_lambda1_eval_seed62065": {
    "overall": null,
    "delta_vs_coherent86": null,
    "component_deltas_vs_coherent86": {
      "BLiMP": null,
      "Supplement": null,
      "EWoK": null,
      "Entity": null,
      "COMPS": null,
      "SuperGLUE": null,
      "GlobalPIQA": null,
      "Reading": null,
      "AoA": null
    },
    "complete": false
  },
  "summary": {
    "completed_models": [
      "coherent86",
      "dense_seed62064",
      "dense_seed62065",
      "clean_pres_lambda1_eval_seed62064"
    ],
    "dense_completed": [
      "dense_seed62064",
      "dense_seed62065"
    ],
    "dense_mean_delta_vs_coherent86": 0.1347889195203429,
    "clean_seed62064_complete": true,
    "clean_seed62064_state": "complete"
  }
}
