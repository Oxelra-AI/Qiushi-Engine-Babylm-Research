# shared factorization result synthesis shared relation-coordinate probe

This controlled probe trains small raw-text models on the repaired static slot confounded factorial and balanced budget preparation/284 `replace_k16_spread` substrate.  Both tied and untied models receive identical strings and labels.  The tied model uses one candidate-event scorer for both relation comparisons and changed-state updates; the untied model has separate comparison and state event scorers with at least as much capacity.  Neither model receives relation ids, voices, subject/object roles, correct slots, or event-role labels as input; candidate names are extracted from the surface and marked only as `<cand>` versus `<other>` inside the raw event string.

- data root: `experiments/archive/representation_and_objectives/data/information_budget_substrate/replace_k16_spread`
- conditions: ['tied', 'untied']
- arms: ['aligned_state_bridge']
- seeds: [28800]
- epochs/lr: 220 / 0.003

## Central per-result readout

| condition | arm | seed | train_state_acc | train_cmp_acc | direct_same | arm_psc_same_graph | arm_pair_both_same_graph | unchanged | arm_mixed_acc | arm_mixed_margin | true_mixed_margin | hh_closure_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tied | aligned_state_bridge | 28800 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 13.692 | 13.692 | 1.000 |
| untied | aligned_state_bridge | 28800 | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 1.000 | 0.000 | -7.716 | -7.716 | 1.000 |

## Group means

| group | n | train_state_acc | train_cmp_acc | direct_same | arm_psc_same_graph | arm_pair_both_same_graph | unchanged | arm_mixed_acc | arm_mixed_margin | hh_closure_acc |
|---|---|---|---|---|---|---|---|---|---|---|
| tied|aligned_state_bridge | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 13.692 | 1.000 |
| untied|aligned_state_bridge | 1 | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 1.000 | 0.000 | -7.716 | 1.000 |

## Tied minus untied paired differences

| comparison | n | mean | values |
|---|---:|---:|---|
| tied_minus_untied|aligned_state_bridge|arm_mixed_acc | 1 | 1.0 | [1.0] |
| tied_minus_untied|aligned_state_bridge|arm_mixed_margin | 1 | 21.40826117541919 | [21.40826117541919] |
| tied_minus_untied|aligned_state_bridge|arm_pair_both_same_graph | 1 | 0.5 | [0.5] |
| tied_minus_untied|aligned_state_bridge|arm_psc_same_graph | 1 | 0.5 | [0.5] |
| tied_minus_untied|aligned_state_bridge|hh_closure_acc | 1 | 0.0 | [0.0] |
| tied_minus_untied|aligned_state_bridge|true_mixed_margin | 1 | 21.40826117541919 | [21.40826117541919] |

## Scientific reading

A positive factorization result requires the tied model, but not the untied equal-capacity model, to recover graph-transfer h1/h3 changed-state choices on same-initial rows under the arm's installed coordinate, preserve unchanged facts, and show corresponding signed mixed held-seen orientation.  Direct-anchor success alone is not enough; it only means h0/h2 bridge rows were fitted.  If tied succeeds while untied fails, the missing ingredient in the DeBERTa independent-head probes is shared computational factorization rather than more evidence or supplied role addresses.  If tied also fails after train fit, the surface parser/encoder or objective still lacks the needed latent role assignment; if both tied and untied succeed, the result would indicate extra capacity or easier optimization rather than the sharing constraint.

## Files
- summary JSON: `experiments/archive/representation_and_objectives/data/shared_coordinate_aligned_seed28800/shared_coordinate_summary.json`
- per-run directories: `experiments/archive/representation_and_objectives/data/shared_coordinate_aligned_seed28800`
