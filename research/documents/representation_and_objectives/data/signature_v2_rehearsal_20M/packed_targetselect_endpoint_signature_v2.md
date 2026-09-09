# earlier analysis packed target-selective endpoint signature v2

JSON: `experiments/archive/representation_and_objectives/data/signature_v2_rehearsal_20M/packed_targetselect_endpoint_signature_v2.json`

## Cheap7-style scores

| arm | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | equal7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 60.190 | 57.900 | 48.020 | 17.940 | 50.340 | 33.225 | 8.040 | 39.379 |
| drop_abs | 60.700 | 58.500 | 50.810 | 18.000 | 50.510 | 34.240 | 8.015 | 40.111 |
| drop_copied_word | 60.100 | 57.200 | 49.990 | 18.830 | 50.560 | 30.765 | 7.750 | 39.314 |

## Primary arm-to-arm deltas

| contrast | Supplement | EWoK | COMPS | GlobalPIQA | equal7 |
|---|---:|---:|---:|---:|---:|
| drop_abs_minus_full | 0.6 | 2.79 | 0.17 | 1.015 | 0.731428 |
| drop_copied_word_minus_full | -0.7 | 1.97 | 0.22 | -2.46 | -0.065715 |
| drop_abs_minus_drop_copied_word | 1.3 | 0.82 | -0.05 | 3.475 | 0.797143 |
| drop_copied_word_minus_drop_abs | -1.3 | -0.82 | 0.05 | -3.475 | -0.797143 |

## EWoK predeclared groups

### relational_domains
| arm | n | micro_acc |
|---|---:|---:|
| full | 1756 | 45.900 |
| drop_abs | 1756 | 48.462 |
| drop_copied_word | 1756 | 47.836 |

### adjacency_independent_domains
| arm | n | micro_acc |
|---|---:|---:|
| full | 464 | 51.940 |
| drop_abs | 464 | 52.155 |
| drop_copied_word | 464 | 50.862 |

## Vector similarity

- `drop_abs_minus_full`: {"view_vs_adjbreak": {"common_keys": ["Supplement", "social-properties", "physical-dynamics", "spatial-relations", "physical-relations", "material-properties", "social-interactions"], "sign_agreement_fraction": 1.0, "cosine": 0.619259, "pearson": 0.42108}, "view_vs_repeat": {"common_keys": ["Supplement", "social-properties", "physical-dynamics", "spatial-relations", "physical-relations", "material-properties", "social-interactions"], "sign_agreement_fraction": 0.571429, "cosine": 0.476062, "pearson": 0.150683}}
- `full_minus_drop_abs`: {"view_vs_adjbreak": {"common_keys": ["Supplement", "social-properties", "physical-dynamics", "spatial-relations", "physical-relations", "material-properties", "social-interactions"], "sign_agreement_fraction": 0.142857, "cosine": -0.619259, "pearson": -0.42108}, "view_vs_repeat": {"common_keys": ["Supplement", "social-properties", "physical-dynamics", "spatial-relations", "physical-relations", "material-properties", "social-interactions"], "sign_agreement_fraction": 0.285714, "cosine": -0.476062, "pearson": -0.150683}}
- `drop_copied_word_minus_full`: {"view_vs_adjbreak": {"common_keys": ["Supplement", "social-properties", "physical-dynamics", "spatial-relations", "physical-relations", "material-properties", "social-interactions"], "sign_agreement_fraction": 0.571429, "cosine": 0.487304, "pearson": 0.387581}, "view_vs_repeat": {"common_keys": ["Supplement", "social-properties", "physical-dynamics", "spatial-relations", "physical-relations", "material-properties", "social-interactions"], "sign_agreement_fraction": 0.428571, "cosine": 0.2162, "pearson": -0.091248}}
- `full_minus_drop_copied_word`: {"view_vs_adjbreak": {"common_keys": ["Supplement", "social-properties", "physical-dynamics", "spatial-relations", "physical-relations", "material-properties", "social-interactions"], "sign_agreement_fraction": 0.428571, "cosine": -0.487304, "pearson": -0.387581}, "view_vs_repeat": {"common_keys": ["Supplement", "social-properties", "physical-dynamics", "spatial-relations", "physical-relations", "material-properties", "social-interactions"], "sign_agreement_fraction": 0.571429, "cosine": -0.2162, "pearson": 0.091248}}
- `drop_abs_minus_drop_copied_word`: {"view_vs_adjbreak": {"common_keys": ["Supplement", "social-properties", "physical-dynamics", "spatial-relations", "physical-relations", "material-properties", "social-interactions"], "sign_agreement_fraction": 0.714286, "cosine": 0.636854, "pearson": 0.274621}, "view_vs_repeat": {"common_keys": ["Supplement", "social-properties", "physical-dynamics", "spatial-relations", "physical-relations", "material-properties", "social-interactions"], "sign_agreement_fraction": 0.857143, "cosine": 0.81798, "pearson": 0.674536}}
- `drop_copied_word_minus_drop_abs`: {"view_vs_adjbreak": {"common_keys": ["Supplement", "social-properties", "physical-dynamics", "spatial-relations", "physical-relations", "material-properties", "social-interactions"], "sign_agreement_fraction": 0.142857, "cosine": -0.636854, "pearson": -0.274621}, "view_vs_repeat": {"common_keys": ["Supplement", "social-properties", "physical-dynamics", "spatial-relations", "physical-relations", "material-properties", "social-interactions"], "sign_agreement_fraction": 0.285714, "cosine": -0.81798, "pearson": -0.674536}}
