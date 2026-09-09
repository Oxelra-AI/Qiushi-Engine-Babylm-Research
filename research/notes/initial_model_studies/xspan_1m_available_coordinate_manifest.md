# xspan 1m available coordinate manifest — XSpan 1M available-coordinate manifest

Evidence JSON: `experiments/archive/initial_model_studies/data/xspan_1m_available_coordinate_manifest.json`

Direct local checkpoint evaluation of earlier analysis XSpan 1M arms: model_path_or_name is hf_model/chck_1M; no local revision selection is used.

| arm | BLiMP | Supp | Entity | COMPS | GPIQA mean | Reading mean |
|---|---:|---:|---:|---:|---:|---:|
| xspan_true_1M | 55.97 | 45.75 | 17.63 | 50.17 | 31.30 | 5.98 |
| xspan_wrong_1M | 57.86 | 47.14 | 17.65 | 49.74 | 30.78 | 6.04 |
| wwm_only_rho0_1M | 55.27 | 50.09 | 17.44 | 49.73 | 33.25 | 6.29 |

## Deltas true-minus-controls
```json
{
  "blimp": {
    "true_minus_wrong": -1.8900000000000006,
    "true_minus_wwm": 0.6999999999999957,
    "wrong_minus_wwm": 2.5899999999999963
  },
  "supplement": {
    "true_minus_wrong": -1.3900000000000006,
    "true_minus_wwm": -4.340000000000003,
    "wrong_minus_wwm": -2.950000000000003
  },
  "entity_tracking": {
    "true_minus_wrong": -0.019999999999999574,
    "true_minus_wwm": 0.18999999999999773,
    "wrong_minus_wwm": 0.2099999999999973
  },
  "comps": {
    "true_minus_wrong": 0.4299999999999997,
    "true_minus_wwm": 0.44000000000000483,
    "wrong_minus_wwm": 0.010000000000005116
  },
  "global_piqa_mean": {
    "true_minus_wrong": 0.5150000000000006,
    "true_minus_wwm": -1.9549999999999983,
    "wrong_minus_wwm": -2.469999999999999
  },
  "reading_mean": {
    "true_minus_wrong": -0.0600000000000005,
    "true_minus_wwm": -0.3150000000000004,
    "wrong_minus_wwm": -0.2549999999999999
  }
}
```
