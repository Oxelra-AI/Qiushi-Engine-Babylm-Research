# state counterfactual ranking smoke results v2 strict state-counterfactual ranking

Inventory JSONL: `experiments/archive/initial_model_studies/data/state_counterfactual_inventory_v2_smoke.jsonl`
Summary JSON: `experiments/archive/initial_model_studies/data/state_counterfactual_inventory_v2_smoke_summary.json`
Results JSON: `experiments/archive/initial_model_studies/data/state_counterfactual_ranking_v2_smoke_results.json`

Cases: 11; families: {'possession': 3, 'location': 6, 'action_consequence': 2}; unique entities 11; unique fillers 7; mean gap 26.545454545454547

| model | n | score_orig | R_state | R_unrelated | R_surface | R_extra | SEM | frac R_extra>0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wwm43_80M | 11 | 9.8528 | +5.4094 | +0.1722 | +0.1110 | +5.2372 | 1.2169 | 0.909 |

## By family

| model | family | n | R_extra | frac pos | R_state | R_unrelated |
|---|---|---:|---:|---:|---:|---:|
| wwm43_80M | action_consequence | 2 | +6.2411 | 1.000 | +6.4454 | +0.2043 |
| wwm43_80M | location | 6 | +4.4533 | 1.000 | +4.7103 | +0.2571 |
| wwm43_80M | possession | 3 | +6.1358 | 0.667 | +6.1169 | -0.0190 |
