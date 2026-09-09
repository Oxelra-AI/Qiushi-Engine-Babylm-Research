# pvdm full readout synthesis — PVDM 80M partial readout synthesis before EWoK

This note stays within the staged PVDM experiment and summarizes the completed broad sentinels, GlobalPIQA all-option margins, and treatment/control training traces. EWoK four-cell is still the pending relation readout.

## Integrity

- `reference`: {'run_dir': 'experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022', 'checkpoint_actual_cumulative_word_exposure': 80011326, 'checkpoint_path': 'experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022/hf_model/chck_80M'}

- `treatment`: {'run_dir': 'experiments/archive/representation_and_objectives/training/runs/pvdm_treatment_70M_to_80M_seed43022', 'word_exposure': 80011326, 'continuation_words': 9971289, 'actual_training_steps': 250, 'stage_stop_name': 'chck_80M', 'checkpoint_exists': True, 'model_safetensors_bytes': 137890400, 'aggregate_mask_stats': {'rows': 64000, 'valid_groups': 9762170, 'label_events_total': 208837, 'events_tokenization_usable': 150074, 'events_selected': 89785, 'rows_with_label_events': 61384, 'rows_with_usable_events': 57333, 'rows_with_selected_events': 49190, 'desired_group_count_sum': 1464247, 'treatment_group_count_sum': 1464247, 'control_group_count_sum': 1464247, 'treatment_token_count_sum': 2147486, 'control_token_count_sum': 2147486, 'target_group_count_sum': 89785, 'background_group_count_sum': 1284677, 'anchor_swap_group_count_sum': 89785, 'rows_forced_above_desired_k': 0, 'selected_tokens': 2147486}, 'selected_event_categories': {'temporal': 18744, 'spatial': 16546, 'physical_change': 23303, 'causal_connector': 19084, 'comparative': 1638, 'negation': 10470}, 'replacement_action_counts': {'mask_token': 1717758, 'keep_original': 214377, 'random_token': 215351}}

- `control`: {'run_dir': 'experiments/archive/representation_and_objectives/training/runs/pvdm_control_70M_to_80M_seed43022', 'word_exposure': 80011326, 'continuation_words': 9971289, 'actual_training_steps': 250, 'stage_stop_name': 'chck_80M', 'checkpoint_exists': True, 'model_safetensors_bytes': 137890400, 'aggregate_mask_stats': {'rows': 64000, 'valid_groups': 9762170, 'label_events_total': 208837, 'events_tokenization_usable': 150074, 'events_selected': 89785, 'rows_with_label_events': 61384, 'rows_with_usable_events': 57333, 'rows_with_selected_events': 49190, 'desired_group_count_sum': 1464247, 'treatment_group_count_sum': 1464247, 'control_group_count_sum': 1464247, 'treatment_token_count_sum': 2147486, 'control_token_count_sum': 2147486, 'target_group_count_sum': 89785, 'background_group_count_sum': 1284677, 'anchor_swap_group_count_sum': 89785, 'rows_forced_above_desired_k': 0, 'selected_tokens': 2147486}, 'selected_event_categories': {'temporal': 18744, 'spatial': 16546, 'physical_change': 23303, 'causal_connector': 19084, 'comparative': 1638, 'negation': 10470}, 'replacement_action_counts': {'mask_token': 1717758, 'keep_original': 214377, 'random_token': 215351}}


## Supplement/Entity sentinels

- `reference`: Supplement=59.21, Entity=27.72

- `control`: Supplement=61.49, Entity=27.73

- `treatment`: Supplement=58.83, Entity=28.78

- treatment-control delta: {'Supplement': -2.6600000000000037, 'Entity': 1.0500000000000007}

- control-reference delta: {'Supplement': 2.280000000000001, 'Entity': 0.010000000000001563}

- treatment-reference delta: {'Supplement': -0.38000000000000256, 'Entity': 1.0600000000000023}


## GlobalPIQA summaries

- `reference` parallel acc=24.2718, hard52 acc=3.8462, hard52 ranks={'3': 21, '2': 8, '4': 21, '1': 2}, hard52 mean top-minus-correct=1.7169; nonparallel acc=53.0000

- `control` parallel acc=24.2718, hard52 acc=5.7692, hard52 ranks={'4': 15, '2': 14, '3': 20, '1': 3}, hard52 mean top-minus-correct=1.6026; nonparallel acc=51.0000

- `treatment` parallel acc=22.3301, hard52 acc=3.8462, hard52 ranks={'4': 23, '3': 20, '2': 7, '1': 2}, hard52 mean top-minus-correct=1.8200; nonparallel acc=48.0000


## Pairwise row movement

### parallel

- `treatment_vs_control`: both_correct=19, a_only=4, b_only=6, both_wrong=74, a_better_rank=9, same_rank=69, b_better_rank=25, mean_margin_delta=0.1327474963963849

  - hard52: {'n': 52, 'a_correct': 2, 'b_correct': 3, 'a_accuracy': 3.8461538461538463, 'b_accuracy': 5.769230769230769, 'a_better_rank': 2, 'same_rank': 34, 'b_better_rank': 16, 'a_lower_margin': 18, 'b_lower_margin': 32, 'mean_margin_delta_a_minus_b': 0.21746542073465774, 'mean_rank_delta_a_minus_b': 0.3269230769230769}

- `control_vs_reference`: both_correct=18, a_only=7, b_only=7, both_wrong=71, a_better_rank=29, same_rank=61, b_better_rank=13, mean_margin_delta=-0.05384348694063987

  - hard52: {'n': 52, 'a_correct': 3, 'b_correct': 2, 'a_accuracy': 5.769230769230769, 'b_accuracy': 3.8461538461538463, 'a_better_rank': 16, 'same_rank': 32, 'b_better_rank': 4, 'a_lower_margin': 28, 'b_lower_margin': 23, 'mean_margin_delta_a_minus_b': -0.11434053592429828, 'mean_rank_delta_a_minus_b': -0.2692307692307692}

- `treatment_vs_reference`: both_correct=18, a_only=5, b_only=7, both_wrong=73, a_better_rank=21, same_rank=59, b_better_rank=23, mean_margin_delta=0.07890400945574504

  - hard52: {'n': 52, 'a_correct': 2, 'b_correct': 2, 'a_accuracy': 3.8461538461538463, 'b_accuracy': 3.8461538461538463, 'a_better_rank': 10, 'same_rank': 31, 'b_better_rank': 11, 'a_lower_margin': 26, 'b_lower_margin': 25, 'mean_margin_delta_a_minus_b': 0.10312488481035947, 'mean_rank_delta_a_minus_b': 0.057692307692307696}

### nonparallel

- `treatment_vs_control`: both_correct=44, a_only=4, b_only=7, both_wrong=45, a_better_rank=4, same_rank=89, b_better_rank=7, mean_margin_delta=-0.023660777962175885

- `control_vs_reference`: both_correct=38, a_only=13, b_only=15, both_wrong=34, a_better_rank=13, same_rank=72, b_better_rank=15, mean_margin_delta=0.0021518195190204682

- `treatment_vs_reference`: both_correct=40, a_only=8, b_only=13, both_wrong=39, a_better_rank=8, same_rank=79, b_better_rank=13, mean_margin_delta=-0.021508958443155417


## Training loss contrast

Matched steps=250, treatment lower loss steps=250, control lower=0, masked tokens identical each step=True. Loss delta treatment-control stats={'n': 250, 'min': -0.11099658387242917, 'p05': -0.09126476519148405, 'mean': -0.06439471160245132, 'median': -0.06353630476602556, 'p95': -0.04017447136468426, 'max': -0.023504348458620594}


## Current interpretation

- The paired treatment has lower PVDM training loss than the matched control at every recorded step, so the true-pivot-visible objective is locally easier on its own masked targets.

- The lower training loss does not transfer to GlobalPIQA: treatment is worse than control on parallel accuracy, nonparallel accuracy, and hard52 mean top-minus-correct; control also softens hard52 margin relative to uninterrupted compact 80M whereas treatment worsens it.

- The treatment-control contrast isolates true-pivot visibility on identical dependent targets and matched mask mass; the partial result therefore disfavors the current dense PVDM formulation as a world-relation repair, even before EWoK arrives.

- The result does not close the broader context-conditioned relation problem. It specifically suggests that making the pivot visible while predicting dependents may reduce pressure to represent relation pivots; the control, which masks true pivots, may preserve or improve relation-word learning at lower broad damage.


Files: `experiments/archive/representation_and_objectives/data/pvdm_partial_readout_synthesis/pvdm_partial_readout_synthesis.json`, `experiments/archive/representation_and_objectives/data/pvdm_partial_readout_synthesis/globalpiqa_parallel_row_shifts.csv`, `experiments/archive/representation_and_objectives/data/pvdm_partial_readout_synthesis/globalpiqa_nonparallel_row_shifts.csv`
