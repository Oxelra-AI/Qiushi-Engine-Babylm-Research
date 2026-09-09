# counterfactual binding credit gate counterfactual binding credit gate

Decision: **REJECT_CURRENT_COUNTERFACTUAL_BINDING_MECHANISM**

Primary split is disjoint in names, objects, event templates, update templates, and query templates; state tokens remain lexically trained while their bindings are new.

| arm | seed 42 pair acc | seed 43 pair acc | mean cross margin |
|---|---:|---:|---:|
| single_world_ce | 0.043 | 0.000 | +0.000 |
| paired_ce | 0.100 | 0.018 | +0.001 |
| random_cross | 0.055 | 0.025 | +0.002 |
| cbca | 0.043 | 0.007 | +0.002 |

CBCA minus best active control by seed: [-0.05666666666666667, -0.018333333333333333]

Paired CE minus single-world CE by seed: [0.05666666666666667, 0.018333333333333333]

Next action: Do not scale this objective; return to the causal gap and seek a non-template mechanism.

Evidence JSON: `experiments/archive/initial_model_studies/data/counterfactual_binding_credit_gate.json`
