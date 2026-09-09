# cpc profiles and prefix probe induction-seeded state-subspace gate

Decision: **REJECT_CURRENT_INDUCTION_SEEDED_STATE_SUBSPACE**

The primary split changes names, objects, templates, and bindings.
The prediction interface is the ordinary MLM vocabulary logits.

| arm | seed 42 pair acc | seed 43 pair acc |
|---|---:|---:|
| off | 0.365 | 0.340 |
| wrong_position | 0.805 | 0.847 |
| wrong_identity | 0.325 | 0.370 |
| identity_continuation | 0.945 | 0.933 |

Candidate minus best active control: [0.1399999999999999, 0.08666666666666667]

Same trained model, candidate minus path-off: [0.9416666666666667, 0.9333333333333333]

Candidate latest-update pair accuracy: [0.9125, 0.9025]

Next action: Do not scale this path. A repeated-cue continuation is insufficient; the next architecture must form reusable semantic variables rather than retrieve lexical episodes.

Evidence JSON: `experiments/archive/initial_model_studies/data/induction_seeded_state_subspace_gate.json`
