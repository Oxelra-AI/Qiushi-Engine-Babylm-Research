# acs screen synthesis ACS paired-screen synthesis

The acs mechanism design ad hoc readout was not used for scientific interpretation. acs screen synthesis used the validated length-normalized all-option GlobalPIQA reader and the fixed cross-endpoint hard52 set from fw globalpiqa relevant substrate, plus the fw ewok interaction reader EWoK four-cell reader.

## Main treatment-minus-control deltas

- `globalpiqa_parallel_accuracy_pp`: -1.9417475728155367
- `globalpiqa_nonparallel_accuracy_pp`: -2.0
- `globalpiqa_fixed_hard52_accuracy_pp`: -1.9230769230769231
- `globalpiqa_fixed_hard52_mean_margin_nats`: -0.11161053173212543
- `globalpiqa_all_rows_mean_margin_nats`: -0.07227450478292963
- `ewok_accuracy`: -0.0009188763454974769
- `ewok_saved_wrong`: 7
- `ewok_stable_failure_count`: -20
- `ewok_stable_failure_frac_all`: -0.0026253609871357275
- `ewok_wrong_interaction_mean_nats`: 0.052132173797088566
- `training_ce_loss_mean`: 0.029413281188118923
- `training_ce_loss_last`: 0.028668000000000138
- `grad_clip_frac`: 0.039603960396039604
- `matched_batch_selected_mass_mean`: -0.05819028615951538
- `matched_batch_correct_prob_mean`: -0.015497535467147827
- `matched_batch_acs_ce_grad_cosine`: -0.30827363309562045

GlobalPIQA row transitions:
- parallel all rows: {'n': 103, 'control_correct': 25, 'treatment_correct': 23, 'net_correct_treat_minus_control': -2, 'transitions': {'W->W': 78, 'C->C': 23, 'C->W': 2}, 'rank_delta_stats': {'n': 103, 'min': -1.0, 'p10': 0.0, 'mean': 0.019417475728155338, 'median': 0.0, 'p90': 0.0, 'max': 1.0}, 'margin_delta_stats_treat_minus_control': {'n': 103, 'min': -0.842332124710083, 'p10': -0.2526842843918574, 'mean': -0.07227450478292956, 'median': -0.015913327534993638, 'p90': 0.038939740922715654, 'max': 0.15324179331461618}, 'changed_choice': 4, 'changed_choice_frac': 0.038834951456310676, 'categories': {'physical_object_interaction': 73, 'officialcat:object_properties_interactions': 36, 'officialcat:spatial': 22, 'officialcat:object_properties_interactions, affordances': 21, 'temporal_arithmetic_or_order': 13, 'direction_spatial': 12, 'tool_affordance': 8, 'officialcat:time, counting': 7, 'officialcat:counting': 7, 'officialcat:time': 4, 'officialcat:object_properties_interactions, spatial': 3, 'officialcat:object_properties': 2, 'officialcat:spatial, counting': 1}}
- fixed hard52: {'n': 52, 'control_correct': 2, 'treatment_correct': 1, 'net_correct_treat_minus_control': -1, 'transitions': {'W->W': 50, 'C->C': 1, 'C->W': 1}, 'rank_delta_stats': {'n': 52, 'min': 0.0, 'p10': 0.0, 'mean': 0.038461538461538464, 'median': 0.0, 'p90': 0.0, 'max': 1.0}, 'margin_delta_stats_treat_minus_control': {'n': 52, 'min': -0.842332124710083, 'p10': -0.2901556587219238, 'mean': -0.11161053173212528, 'median': -0.05738671620686853, 'p90': 0.04233926137288421, 'max': 0.13159569104512503}, 'changed_choice': 2, 'changed_choice_frac': 0.038461538461538464, 'categories': {'physical_object_interaction': 33, 'officialcat:spatial': 16, 'officialcat:object_properties_interactions': 15, 'direction_spatial': 9, 'officialcat:object_properties_interactions, affordances': 9, 'temporal_arithmetic_or_order': 6, 'tool_affordance': 5, 'officialcat:counting': 5, 'officialcat:time, counting': 3, 'officialcat:object_properties_interactions, spatial': 2, 'officialcat:time': 1, 'officialcat:object_properties': 1}}
- nonparallel all rows: {'n': 100, 'control_correct': 51, 'treatment_correct': 49, 'net_correct_treat_minus_control': -2, 'transitions': {'C->C': 48, 'C->W': 3, 'W->W': 48, 'W->C': 1}, 'rank_delta_stats': {'n': 100, 'min': -1.0, 'p10': 0.0, 'mean': 0.02, 'median': 0.0, 'p90': 0.0, 'max': 1.0}, 'margin_delta_stats_treat_minus_control': {'n': 100, 'min': -0.22899100894019675, 'p10': -0.0730601106371202, 'mean': -0.01218409014926334, 'median': 0.0, 'p90': 0.017317490144209594, 'max': 0.14430345807756684}, 'changed_choice': 4, 'changed_choice_frac': 0.04, 'categories': {'all': 100}}

EWoK row transitions:
- {'n_common': 7618, 'accuracy_transition_counts': {'W->W': 3649, 'C->C': 3728, 'C->W': 124, 'W->C': 117}, 'net_correct_treat_minus_control': -7, 'stable_transition_counts': {'S->S': 2436, 'N->N': 4900, 'N->S': 131, 'S->N': 151}, 'net_stable_failures_treat_minus_control': -20, 'stable_removed_domains': {'agent-properties': 34, 'social-relations': 27, 'material-dynamics': 22, 'social-properties': 18, 'physical-interactions': 14, 'social-interactions': 8, 'material-properties': 7, 'physical-relations': 7, 'quantitative-properties': 6, 'spatial-relations': 6, 'physical-dynamics': 2}, 'stable_added_domains': {'agent-properties': 29, 'material-dynamics': 25, 'social-relations': 22, 'physical-interactions': 13, 'spatial-relations': 9, 'social-interactions': 8, 'physical-relations': 7, 'material-properties': 6, 'social-properties': 6, 'quantitative-properties': 4, 'physical-dynamics': 2}, 'correct_gained_domains': {'agent-properties': 24, 'social-relations': 22, 'material-dynamics': 15, 'physical-interactions': 14, 'social-properties': 12, 'social-interactions': 8, 'material-properties': 6, 'spatial-relations': 6, 'quantitative-properties': 5, 'physical-relations': 4, 'physical-dynamics': 1}, 'correct_lost_domains': {'agent-properties': 30, 'material-dynamics': 24, 'social-relations': 17, 'physical-interactions': 15, 'spatial-relations': 9, 'material-properties': 8, 'quantitative-properties': 8, 'social-interactions': 5, 'physical-relations': 4, 'social-properties': 3, 'physical-dynamics': 1}, 'interaction_delta_all_stats': {'n': 7618, 'min': -3.6760499058291316, 'p10': -0.467339280154556, 'mean': -0.0048572335307737035, 'median': 0.002639050828292966, 'p90': 0.4444511428475384, 'max': 3.2949514884967357}, 'interaction_delta_control_wrong_stats': {'n': 3766, 'min': -3.6760499058291316, 'p10': -0.3829278266057372, 'mean': 0.044158820268113394, 'median': 0.023162666708230972, 'p90': 0.5084702743333764, 'max': 3.2949514884967357}, 'interaction_delta_control_stable_failure_stats': {'n': 2587, 'min': -3.6760499058291316, 'p10': -0.3409772136074025, 'mean': 0.07977759940849094, 'median': 0.052221391815692186, 'p90': 0.594316296745092, 'max': 3.2949514884967357}, 'within_both_positive_wrong_delta_count': -4}

## Mechanism diagnostics

- Treatment mean CE loss 2.4919223584158416; control 2.4625090772277227; delta 0.029413281188118923.
- Treatment clipping fraction 0.039603960396039604; control 0.0.
- Matched-batch selected-set mass: treatment 0.7036445736885071; control 0.7618348598480225.
- Matched-batch ACS--CE gradient cosine: treatment 0.46282202426379965; control 0.7710956573594201.

## Interpretation

ACS alpha0.5/topk8 completed, but fixed-coordinate readout does not show the required interaction-specific natural transfer: GlobalPIQA_parallel and fixed hard52 correctness worsen, EWoK stable failures improve only 20 rows while EWoK accuracy slightly drops, and diagnostics show ACS increases CE loss/gradient clipping and lowers selected-set probability mass on a matched batch.
Do not start alpha/K tuning or from-beginning ACS training from this result; treat ACS-alpha0.5-topk8 as a negative or at best non-decisive single-context hard-negative CE screen unless a separate paired-context coupling mechanism is invented and screened.

Files:
- synthesis JSON: `experiments/archive/representation_and_objectives/data/acs_synthesis/acs_screen_synthesis.json`
- transitions: `experiments/archive/representation_and_objectives/data/acs_synthesis/globalpiqa_parallel_transitions.jsonl`, `experiments/archive/representation_and_objectives/data/acs_synthesis/ewok_transitions.jsonl`
