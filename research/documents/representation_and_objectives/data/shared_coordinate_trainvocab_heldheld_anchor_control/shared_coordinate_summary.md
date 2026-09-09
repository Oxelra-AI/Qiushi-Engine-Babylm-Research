# shared factorization result synthesis shared relation-coordinate probe

This controlled probe trains small raw-text models on the repaired static slot confounded factorial and balanced budget preparation/284 `replace_k16_spread` substrate.  Both tied and untied models receive identical strings and labels.  The tied model uses one candidate-event scorer for both relation comparisons and changed-state updates; the untied model has separate comparison and state event scorers with at least as much capacity.  Neither model receives relation ids, voices, subject/object roles, correct slots, or event-role labels as input; candidate names are extracted from the surface and marked only as `<cand>` versus `<other>` inside the raw event string.

- data root: `experiments/archive/representation_and_objectives/data/information_budget_substrate/replace_k16_spread`
- conditions: ['tied', 'untied']
- arms: ['heldheld_only']
- seeds: [28801, 28802]
- epochs/lr: 220 / 0.003
- vocab scope: `train`

## Central per-result readout

| condition | arm | seed | train_state_acc | train_cmp_acc | direct_same | arm_psc_same_graph | arm_pair_both_same_graph | unchanged | arm_mixed_acc | arm_mixed_margin | true_mixed_margin | hh_closure_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tied | heldheld_only | 28801 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | -11.493 | -11.493 | 1.000 |
| untied | heldheld_only | 28801 | 1.000 | 1.000 | 0.500 | 0.500 | 0.500 | 1.000 | 0.000 | -8.974 | -8.974 | 1.000 |
| tied | heldheld_only | 28802 | 1.000 | 0.875 | 0.000 | 0.250 | 0.250 | 1.000 | 0.125 | -4.897 | -4.897 | 0.625 |
| untied | heldheld_only | 28802 | 1.000 | 1.000 | 0.500 | 0.250 | 0.250 | 1.000 | 0.500 | -2.468 | -2.468 | 1.000 |

## Group means

| group | n | train_state_acc | train_cmp_acc | direct_same | arm_psc_same_graph | arm_pair_both_same_graph | unchanged | arm_mixed_acc | arm_mixed_margin | hh_closure_acc |
|---|---|---|---|---|---|---|---|---|---|---|
| tied|heldheld_only | 2 | 1.000 | 0.938 | 0.000 | 0.125 | 0.125 | 1.000 | 0.062 | -8.195 | 0.812 |
| untied|heldheld_only | 2 | 1.000 | 1.000 | 0.500 | 0.375 | 0.375 | 1.000 | 0.250 | -5.721 | 1.000 |

## Tied minus untied paired differences

| comparison | n | mean | values |
|---|---:|---:|---|
| tied_minus_untied|heldheld_only|arm_mixed_acc | 2 | -0.1875 | [0.0, -0.375] |
| tied_minus_untied|heldheld_only|arm_mixed_margin | 2 | -2.473902523932726 | [-2.5195966209194776, -2.4282084269459743] |
| tied_minus_untied|heldheld_only|arm_pair_both_same_graph | 2 | -0.25 | [-0.5, 0.0] |
| tied_minus_untied|heldheld_only|arm_psc_same_graph | 2 | -0.25 | [-0.5, 0.0] |
| tied_minus_untied|heldheld_only|hh_closure_acc | 2 | -0.1875 | [0.0, -0.375] |
| tied_minus_untied|heldheld_only|true_mixed_margin | 2 | -2.473902523932726 | [-2.5195966209194776, -2.4282084269459743] |

## Scientific reading

A positive factorization result requires the tied model, but not the untied equal-capacity model, to recover graph-transfer h1/h3 changed-state choices on same-initial rows under the arm's installed coordinate, preserve unchanged facts, and show corresponding signed mixed held-seen orientation.  Direct-anchor success alone is not enough; it only means h0/h2 bridge rows were fitted.  If tied succeeds while untied fails, the missing ingredient in the DeBERTa independent-head probes is shared computational factorization rather than more evidence or supplied role addresses.  If tied also fails after train fit, the surface parser/encoder or objective still lacks the needed latent role assignment; if both tied and untied succeed, the result would indicate extra capacity or easier optimization rather than the sharing constraint.

## Files
- summary JSON: `experiments/archive/representation_and_objectives/data/shared_coordinate_trainvocab_heldheld_anchor_control/shared_coordinate_summary.json`
- per-run directories: `experiments/archive/representation_and_objectives/data/shared_coordinate_trainvocab_heldheld_anchor_control`
