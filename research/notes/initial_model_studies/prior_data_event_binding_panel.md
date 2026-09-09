# prior data event binding panel prior-data event-binding panel

Decision: **NO_PRIOR_DATA_ROUTE_MOVED_EVENT_BINDING_AT_10M**

| model | effect mean | 95% CI |
|---|---:|---:|
| official_s1 | -0.000037 | [-0.000146, +0.000071] |
| wikiauto_aligned | +0.000000 | [-0.000000, +0.000000] |
| wikiauto_shuffled | -0.000000 | [-0.000000, +0.000000] |
| high_entity_state | -0.000000 | [-0.000000, +0.000000] |
| matched_low_structure | -0.000000 | [-0.000000, +0.000000] |
| uniform_structure | +0.000000 | [-0.000000, +0.000000] |

| matched contrast | delta mean | 95% CI |
|---|---:|---:|
| wikiauto_aligned_minus_shuffled | +0.000000 | [-0.000000, +0.000000] |
| high_entity_state_minus_matched_low | +0.000000 | [-0.000000, +0.000000] |
| high_entity_state_minus_uniform | -0.000000 | [-0.000000, +0.000000] |

Next action: Do not reopen WikiAuto adjacency or static structure-density selection. Test a symmetric crossed-logit objective that directly requires ordinary MLM logits to reverse with bindings.

Evidence JSON: `experiments/archive/initial_model_studies/data/prior_data_event_binding_panel.json`
