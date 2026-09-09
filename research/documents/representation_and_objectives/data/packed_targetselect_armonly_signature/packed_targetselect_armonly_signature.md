# earlier analysis packed target-selective arm-to-arm endpoint signature

JSON: `experiments/archive/representation_and_objectives/data/packed_targetselect_armonly_signature/packed_targetselect_armonly_signature.json`

## Scores

| arm | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | equal7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| drop_abs | 65.640 | 63.730 | 51.510 | 26.720 | 51.600 | 36.665 | 8.195 | 43.437 |
| drop_copied_word | 67.320 | 59.840 | 51.930 | 25.670 | 51.580 | 35.090 | 8.115 | 42.792 |

## Main score deltas

| contrast | Supplement | EWoK | COMPS | GlobalPIQA | equal7 |
|---|---:|---:|---:|---:|---:|
| drop_abs_minus_drop_copied_word | 3.89 | -0.42 | 0.02 | 1.575 | 0.645 |
| drop_copied_word_minus_drop_abs | -3.89 | 0.42 | -0.02 | -1.575 | -0.645 |

## EWoK correctness transition analysis groups

### relational_domains
| arm | n | micro accuracy |
|---|---:|---:|
| drop_abs | 1756 | 49.032 |
| drop_copied_word | 1756 | 49.146 |

| contrast | delta pp | net items | bootstrap p025 | bootstrap p975 |
|---|---:|---:|---:|---:|
| drop_abs_minus_drop_copied_word | -0.113895 | -2 | -2.505695 | 2.277904 |
| drop_copied_word_minus_drop_abs | 0.113895 | 2 | -2.334852 | 2.448747 |

### adjacency_independent_domains
| arm | n | micro accuracy |
|---|---:|---:|
| drop_abs | 464 | 60.345 |
| drop_copied_word | 464 | 62.069 |

| contrast | delta pp | net items | bootstrap p025 | bootstrap p975 |
|---|---:|---:|---:|---:|
| drop_abs_minus_drop_copied_word | -1.724138 | -8 | -6.034483 | 3.017241 |
| drop_copied_word_minus_drop_abs | 1.724138 | 8 | -3.448276 | 6.034483 |

## Historical-vector comparison

- `drop_abs_minus_drop_copied_word`: {"view_vs_adjbreak": {"common_keys": ["Supplement", "social-properties", "physical-dynamics", "spatial-relations", "physical-relations", "material-properties", "social-interactions"], "sign_agreement_fraction": 0.428571, "cosine": -0.021789, "pearson": 0.123341}, "view_vs_repeat": {"common_keys": ["Supplement", "social-properties", "physical-dynamics", "spatial-relations", "physical-relations", "material-properties", "social-interactions"], "sign_agreement_fraction": 0.571429, "cosine": -0.21847, "pearson": -0.212824}}
- `drop_copied_word_minus_drop_abs`: {"view_vs_adjbreak": {"common_keys": ["Supplement", "social-properties", "physical-dynamics", "spatial-relations", "physical-relations", "material-properties", "social-interactions"], "sign_agreement_fraction": 0.571429, "cosine": 0.021789, "pearson": -0.123341}, "view_vs_repeat": {"common_keys": ["Supplement", "social-properties", "physical-dynamics", "spatial-relations", "physical-relations", "material-properties", "social-interactions"], "sign_agreement_fraction": 0.428571, "cosine": 0.21847, "pearson": 0.212824}}
