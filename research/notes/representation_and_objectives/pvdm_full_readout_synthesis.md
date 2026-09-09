# pvdm full readout synthesis — PVDM 80M full readout synthesis

The staged PVDM experiment tested the accumulated relation question with treatment/control/reference readouts at the same 80,011,326-word boundary.

## Integrity

- `reference`: {'word_exposure': 80011326, 'path': 'experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022/hf_model/chck_80M'}

- `treatment`: {'word_exposure': 80011326, 'continuation_words': 9971289, 'steps': 250, 'stage_stop_name': 'chck_80M', 'checkpoint_exists': True, 'selected_event_categories': {'temporal': 18744, 'spatial': 16546, 'physical_change': 23303, 'causal_connector': 19084, 'comparative': 1638, 'negation': 10470}, 'aggregate_mask_stats': {'rows': 64000, 'valid_groups': 9762170, 'label_events_total': 208837, 'events_tokenization_usable': 150074, 'events_selected': 89785, 'rows_with_label_events': 61384, 'rows_with_usable_events': 57333, 'rows_with_selected_events': 49190, 'desired_group_count_sum': 1464247, 'treatment_group_count_sum': 1464247, 'control_group_count_sum': 1464247, 'treatment_token_count_sum': 2147486, 'control_token_count_sum': 2147486, 'target_group_count_sum': 89785, 'background_group_count_sum': 1284677, 'anchor_swap_group_count_sum': 89785, 'rows_forced_above_desired_k': 0, 'selected_tokens': 2147486}}

- `control`: {'word_exposure': 80011326, 'continuation_words': 9971289, 'steps': 250, 'stage_stop_name': 'chck_80M', 'checkpoint_exists': True, 'selected_event_categories': {'temporal': 18744, 'spatial': 16546, 'physical_change': 23303, 'causal_connector': 19084, 'comparative': 1638, 'negation': 10470}, 'aggregate_mask_stats': {'rows': 64000, 'valid_groups': 9762170, 'label_events_total': 208837, 'events_tokenization_usable': 150074, 'events_selected': 89785, 'rows_with_label_events': 61384, 'rows_with_usable_events': 57333, 'rows_with_selected_events': 49190, 'desired_group_count_sum': 1464247, 'treatment_group_count_sum': 1464247, 'control_group_count_sum': 1464247, 'treatment_token_count_sum': 2147486, 'control_token_count_sum': 2147486, 'target_group_count_sum': 89785, 'background_group_count_sum': 1284677, 'anchor_swap_group_count_sum': 89785, 'rows_forced_above_desired_k': 0, 'selected_tokens': 2147486}}


## Main table

- `reference`: Supplement=59.21, Entity=27.72; GP_parallel=24.2718, GP_nonparallel=53.0000, GP_hard52_margin=1.7169, GP_hard52_ranks={'3': 21, '2': 8, '4': 21, '1': 2}; EWoK_acc=0.501050, EWoK_stable_frac_wrong=0.692712, EWoK_wrong_median=-0.927948

- `control`: Supplement=61.49, Entity=27.73; GP_parallel=24.2718, GP_nonparallel=51.0000, GP_hard52_margin=1.6026, GP_hard52_ranks={'4': 15, '2': 14, '3': 20, '1': 3}; EWoK_acc=0.498556, EWoK_stable_frac_wrong=0.697906, EWoK_wrong_median=-1.199480

- `treatment`: Supplement=58.83, Entity=28.78; GP_parallel=22.3301, GP_nonparallel=48.0000, GP_hard52_margin=1.8200, GP_hard52_ranks={'4': 23, '3': 20, '2': 7, '1': 2}; EWoK_acc=0.495537, EWoK_stable_frac_wrong=0.707520, EWoK_wrong_median=-1.200681


## Deltas

- `treatment_minus_control`: {'Supplement': -2.6600000000000037, 'Entity': 1.0500000000000007, 'GlobalPIQA_parallel': -1.9417475728155367, 'GlobalPIQA_nonparallel': -3.0, 'GP_hard52_mean_margin': 0.2174654207346578, 'EWoK_accuracy': -0.003019165135206059, 'EWoK_stable_failure_frac_wrong': 0.00961440737425634, 'EWoK_wrong_median_interaction': -0.0012007178738713264}

- `control_minus_reference`: {'Supplement': 2.280000000000001, 'Entity': 0.010000000000001563, 'GlobalPIQA_parallel': 0.0, 'GlobalPIQA_nonparallel': -2.0, 'GP_hard52_mean_margin': -0.11434053592429838, 'EWoK_accuracy': -0.002494092937778969, 'EWoK_stable_failure_frac_wrong': 0.0051933150686440666, 'EWoK_wrong_median_interaction': -0.2715316116809845}

- `treatment_minus_reference`: {'Supplement': -0.38000000000000256, 'Entity': 1.0600000000000023, 'GlobalPIQA_parallel': -1.9417475728155367, 'GlobalPIQA_nonparallel': -5.0, 'GP_hard52_mean_margin': 0.10312488481035942, 'EWoK_accuracy': -0.005513258072985028, 'EWoK_stable_failure_frac_wrong': 0.014807722442900406, 'EWoK_wrong_median_interaction': -0.2727323295548558}


## Row-level movement

- GlobalPIQA hard52 treatment vs control: treatment better rank 2, same 34, control better 16; treatment lower margin 18, control lower margin 32; mean margin delta 0.217465.

- EWoK treatment vs control: treatment correct 3775, control correct 3798, treatment-only correct 911, control-only correct 934; stable treatment 2719, stable control 2666, stable both 1797.


## Training signal

Treatment lower loss steps=250/250; final loss delta=-0.054532; mean delta=-0.064395; masked tokens identical each step=True.


## Mechanism interpretation

- Treatment lower training loss on identical target/mask mass did not translate to relation competence; treatment is worse than control on GlobalPIQA parallel, hard52 margin/ranks, nonparallel GlobalPIQA, Supplement, and EWoK accuracy/stable failures, with only Entity improving.

- Control, which masks the true relation pivot while keeping the surrogate visible, is closer to or better than the uninterrupted compact 80M reference on Supplement and GlobalPIQA hard52 margin, but worse on EWoK than reference. This means the common continuation/target-redistribution cost is nontrivial, and true-pivot visibility adds damage rather than repair.

- The current dense PVDM formulation is closed as a continuation route to 100M. The broader context-conditioned world-relation problem remains open; the mechanism lesson is that hiding the relational pivot may be more valuable than making it visible, because predicting relation words and consequences jointly may be necessary to learn conditional compatibility.

- A coherent next research step should revise the relational objective/representation using this specific failure: preserve broad compact learning and train a relation-conditioned contrast or pivot-prediction signal that strengthens interaction without shifting broad probability mass, rather than spending more exposure on this PVDM treatment.


Files: `experiments/archive/representation_and_objectives/data/pvdm_full_readout_synthesis/pvdm_full_readout_synthesis.json`, `experiments/archive/representation_and_objectives/data/pvdm_full_readout_synthesis/globalpiqa_parallel_threeway_rows.csv`, `experiments/archive/representation_and_objectives/data/pvdm_full_readout_synthesis/globalpiqa_nonparallel_threeway_rows.csv`
