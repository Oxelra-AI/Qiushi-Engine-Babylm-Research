# state counterfactual ranking smoke results state-counterfactual ranking results

Inventory JSONL: `experiments/archive/initial_model_studies/data/state_counterfactual_inventory_smoke.jsonl`
Summary JSON: `experiments/archive/initial_model_studies/data/state_counterfactual_inventory_smoke_summary.json`
Results JSON: `experiments/archive/initial_model_studies/data/state_counterfactual_ranking_smoke_results.json`

Cases: 60; families: {'possession': 33, 'role_attribute': 6, 'location': 11, 'action_consequence': 7, 'container': 3}; mean early gap 28.1; mean unrelated gap 29.0

| model | n | score_orig | R_state | R_unrelated | R_surface | R_extra | SEM | frac R_extra>0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wwm43_80M | 56 | 9.1272 | +1.2260 | +0.0083 | -0.0148 | +1.2177 | 0.3712 | 0.679 |

## By family: R_extra mean

| model | family | n | R_extra | R_state | R_unrelated |
|---|---|---:|---:|---:|---:|
| wwm43_80M | action_consequence | 5 | +0.5096 | +0.4092 | -0.1004 |
| wwm43_80M | container | 2 | +0.2212 | +0.2219 | +0.0006 |
| wwm43_80M | location | 11 | +1.3855 | +1.4416 | +0.0561 |
| wwm43_80M | possession | 32 | +0.6343 | +0.6583 | +0.0239 |
| wwm43_80M | role_attribute | 6 | +4.9435 | +4.8738 | -0.0697 |

Interpretation: positive R_extra means the early state-changing fact changes y1/y2 ranking more than a matched unrelated-history edit. This must be positive and robust before a delayed state-contrastive pretraining objective is justified.
