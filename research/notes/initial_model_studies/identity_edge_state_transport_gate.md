# identity edge state transport gate identity-edge state transport gate

Decision: **REJECT_CURRENT_IDENTITY_EDGE_TRANSPORT_MECHANISM**

The primary split changes names, objects, event/update/query templates, and all bindings. No BabyLM evaluation data is used.

| arm | seed 42 pair acc | seed 43 pair acc |
|---|---:|---:|
| off | 0.238 | 0.183 |
| previous_token | 0.228 | 0.183 |
| wrong_identity | 0.225 | 0.177 |
| identity | 0.210 | 0.178 |

Identity minus best active control: [-0.01833333333333334, -0.004999999999999977]

Same trained model, identity minus path-off: [0.0, 0.0]

Identity latest-update pair accuracy: [0.2575, 0.175]

Next action: Do not scale this implementation; use the failed dimensions to revise the inductive bias rather than repairing thresholds.

Evidence JSON: `experiments/archive/initial_model_studies/data/identity_edge_state_transport_gate.json`
