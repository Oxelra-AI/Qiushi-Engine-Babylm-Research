# shared factorization result synthesis shared-coordinate merged analysis

This file merges saved tied/untied raw-text factorization probe outputs. It reports train fit, direct-anchor state learning, graph-transfer h1/h3 state choices, joint changed+unchanged state conservation, and signed relation-comparison readouts. No model is loaded here.

- runs read: 8

## Central rows

| condition | arm | seed | vocab_scope | final_train_state_acc | final_train_cmp_acc | eval_arm_psc_changed_same_direct | eval_arm_psc_changed_same_graph | eval_arm_psc_pair_both_same_graph | eval_arm_psc_unchanged | eval_arm_mixed_held_seen_orientation_acc | eval_arm_mixed_held_seen_orientation_signed_margin | eval_arm_heldheld_unseen_edge_closure_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tied | aligned_state_bridge | 28800 | unknown | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 13.692 | 1.000 |
| untied | aligned_state_bridge | 28800 | unknown | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 1.000 | 0.000 | -7.716 | 1.000 |
| tied | inverted_state_bridge | 28800 | unknown | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 13.769 | 1.000 |
| untied | inverted_state_bridge | 28800 | unknown | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 1.000 | 1.000 | 7.716 | 1.000 |
| tied | aligned_state_bridge | 28801 | train | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 13.809 | 1.000 |
| untied | aligned_state_bridge | 28801 | train | 1.000 | 1.000 | 1.000 | 0.250 | 0.250 | 1.000 | 0.500 | 4.698 | 1.000 |
| tied | inverted_state_bridge | 28801 | train | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 13.600 | 1.000 |
| untied | inverted_state_bridge | 28801 | train | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 1.000 | 0.500 | -4.698 | 1.000 |

## Tied minus untied paired differences

| metric | n | mean | values |
|---|---:|---:|---|
| tied_minus_untied|aligned_state_bridge|eval_arm_heldheld_unseen_edge_closure_acc | 2 | 0.000 | [0.0, 0.0] |
| tied_minus_untied|aligned_state_bridge|eval_arm_mixed_held_seen_orientation_acc | 2 | 0.750 | [1.0, 0.5] |
| tied_minus_untied|aligned_state_bridge|eval_arm_mixed_held_seen_orientation_signed_margin | 2 | 15.260 | [21.40826117541919, 9.111076490461613] |
| tied_minus_untied|aligned_state_bridge|eval_arm_psc_changed_same_graph | 2 | 0.625 | [0.5, 0.75] |
| tied_minus_untied|aligned_state_bridge|eval_arm_psc_pair_both_same_graph | 2 | 0.625 | [0.5, 0.75] |
| tied_minus_untied|aligned_state_bridge|eval_arm_psc_unchanged | 2 | 0.000 | [0.0, 0.0] |
| tied_minus_untied|aligned_state_bridge|final_train_cmp_acc | 2 | 0.000 | [0.0, 0.0] |
| tied_minus_untied|aligned_state_bridge|final_train_state_acc | 2 | 0.000 | [0.0, 0.0] |
| tied_minus_untied|inverted_state_bridge|eval_arm_heldheld_unseen_edge_closure_acc | 2 | 0.000 | [0.0, 0.0] |
| tied_minus_untied|inverted_state_bridge|eval_arm_mixed_held_seen_orientation_acc | 2 | 0.250 | [0.0, 0.5] |
| tied_minus_untied|inverted_state_bridge|eval_arm_mixed_held_seen_orientation_signed_margin | 2 | 12.175 | [6.052688655436777, 18.297859580772062] |
| tied_minus_untied|inverted_state_bridge|eval_arm_psc_changed_same_graph | 2 | 0.500 | [0.5, 0.5] |
| tied_minus_untied|inverted_state_bridge|eval_arm_psc_pair_both_same_graph | 2 | 0.500 | [0.5, 0.5] |
| tied_minus_untied|inverted_state_bridge|eval_arm_psc_unchanged | 2 | 0.000 | [0.0, 0.0] |
| tied_minus_untied|inverted_state_bridge|final_train_cmp_acc | 2 | 0.000 | [0.0, 0.0] |
| tied_minus_untied|inverted_state_bridge|final_train_state_acc | 2 | 0.000 | [0.0, 0.0] |

## Scientific reading

The important pattern is whether tying the event-output coordinate raises same-initial graph-transfer h1/h3 state choice and pair-both conservation while preserving unchanged facts and producing the expected signed mixed held-seen relation orientation. If train comparison fit is weak, the run mainly shows that this raw-text scalar scorer failed to learn the comparison graph, not that the principle is false. If tied beats untied only on direct h0/h2 rows, the sharing constraint has not transported the coordinate through the held-held graph.

## Files
- full JSON: `experiments/archive/representation_and_objectives/data/shared_coordinate_all_merged_analysis/shared_coordinate_merged_analysis.json`
