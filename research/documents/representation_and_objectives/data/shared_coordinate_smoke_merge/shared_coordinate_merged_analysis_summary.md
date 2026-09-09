# shared factorization result synthesis shared-coordinate merged analysis

This file merges saved tied/untied raw-text factorization probe outputs. It reports train fit, direct-anchor state learning, graph-transfer h1/h3 state choices, joint changed+unchanged state conservation, and signed relation-comparison readouts. No model is loaded here.

- runs read: 1

## Central rows

| condition | arm | seed | final_train_state_acc | final_train_cmp_acc | eval_arm_psc_changed_same_direct | eval_arm_psc_changed_same_graph | eval_arm_psc_pair_both_same_graph | eval_arm_psc_unchanged | eval_arm_mixed_held_seen_orientation_acc | eval_arm_mixed_held_seen_orientation_signed_margin | eval_arm_heldheld_unseen_edge_closure_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tied | aligned_state_bridge | 28800 | 0.958 | 0.542 | 0.250 | 0.500 | 0.500 | 1.000 | 0.359 | 0.000 | 0.453 |

## Tied minus untied paired differences

| metric | n | mean | values |
|---|---:|---:|---|

## Scientific reading

The important pattern is whether tying the event-output coordinate raises same-initial graph-transfer h1/h3 state choice and pair-both conservation while preserving unchanged facts and producing the expected signed mixed held-seen relation orientation. If train comparison fit is weak, the run mainly shows that this raw-text scalar scorer failed to learn the comparison graph, not that the principle is false. If tied beats untied only on direct h0/h2 rows, the sharing constraint has not transported the coordinate through the held-held graph.

## Files
- full JSON: `experiments/archive/representation_and_objectives/data/shared_coordinate_smoke_merge/shared_coordinate_merged_analysis.json`
