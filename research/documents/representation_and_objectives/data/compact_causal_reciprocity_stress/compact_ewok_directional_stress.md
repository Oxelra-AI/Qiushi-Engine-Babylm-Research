# earlier analysis compact EWoK directional stress test

Eval root: `experiments/archive/representation_and_objectives/data/compact_directional_cheap7_eval_r2`
Gold surface: `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered`

Pre-H100 survival: **False** — fails distributed relational-domain support: fewer than 3 of 4 correctness transition analysis relational domains have positive same-sign evidence.

## correctness transition analysis relational domains

| domain | n | FR-FF pp | RF-RR pp | I pp | f net | r net | same-sign + |
|---|---:|---:|---:|---:|---:|---:|---:|
| social-properties | 328 | 1.524 | 1.829 | 1.677 | 5 | 6 | True |
| physical-dynamics | 120 | 2.500 | -3.333 | -0.417 | 3 | -4 | False |
| spatial-relations | 490 | -2.245 | 0.204 | -1.020 | -11 | 1 | False |
| physical-relations | 818 | 1.589 | 0.367 | 0.978 | 13 | 3 | True |

## Groups

| group | n | FR-FF pp | RF-RR pp | I pp | bootstrap I 95% |
|---|---:|---:|---:|---:|---:|
| all_full_ewok | 7618 | -0.144 | -0.289 | -0.217 | [-0.643, 0.217] |
| relational | 1756 | 0.569 | 0.342 | 0.456 | [-0.456, 1.367] |
| adjacency_independent | 464 | -2.155 | -0.647 | -1.401 | [-3.017, 0.216] |
| positive_step211_relational_domains_only | 1146 | 1.571 | 0.785 | 1.178 | [0.044, 2.312] |
| nonpositive_step211_relational_domains_only | 610 | -1.311 | -0.492 | -0.902 | [-2.459, 0.656] |

The positive relational-group average is not domain-distributed: only social-properties and physical-relations have positive same-sign deltas; physical-dynamics and spatial-relations are nonpositive/mixed. The broad full-EWoK interaction remains negative.

JSON: `experiments/archive/representation_and_objectives/data/compact_causal_reciprocity_stress/compact_ewok_directional_stress.json`
