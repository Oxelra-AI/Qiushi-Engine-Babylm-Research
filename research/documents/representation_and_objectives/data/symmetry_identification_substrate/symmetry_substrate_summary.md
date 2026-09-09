# symmetry identification route symmetry-identification substrate audit

CPU/file-only construction for a natural transfer-of-possession symmetry experiment. No model loading, training, official evaluation, GPU work, upload, or leaderboard action occurred.

## Scientific object

The held lexical component has relation-comparison rows that are internally learnable while remaining invariant under a global swap of the two possessor roles. Natural sparse bridges to the seen transfer coordinate break this ambiguity; inverted bridges select the opposite orientation; decoupled text controls preserve the ambiguity.

## Row counts

- common seen-coordinate rows: 192 labeled rows, 4656 approximate words, true fraction 0.500

| arm | supervised rows | unsup rows | orientation-breaking supervised rows present | true frac | slot0 true frac |
|---|---:|---:|---|---:|---:|
| exposure_only | 0 | 64 | False | nan | nan |
| heldheld_only | 64 | 64 | False | 0.500 | nan |
| aligned_state_bridge | 80 | 64 | True | 0.500 | 0.250 |
| inverted_state_bridge | 80 | 64 | True | 0.500 | 0.750 |
| decoupled_state_bridge | 80 | 64 | False | 0.500 | 0.250 |
| mixed_event_bridge | 68 | 64 | True | 0.500 | nan |

## Evaluation suites

| suite | rows | true frac | slot0 true frac | first-slot shortcut | second-slot shortcut |
|---|---:|---:|---:|---:|---:|
| held_held_coherence | 48 | 0.500 | nan | nan | nan |
| mixed_held_seen_orientation | 96 | 0.500 | nan | nan | nan |
| paired_state_conservation | 192 | 0.500 | 0.500 | 0.500 | 0.500 |
| cross_frame_state_readout | 96 | 0.500 | 0.500 | 0.500 | 0.500 |
| name_permutation_counterfactual | 64 | 0.500 | 0.500 | 0.500 | 0.500 |

## Symmetry audit

- heldheld_train_labels_invariant_under_global_swap: True
- held_seen_mixed_eval_labels_change_under_held_global_swap: True
- aligned_bridge_breaks_symmetry: True
- inverted_bridge_breaks_symmetry_opposite_labels: True
- decoupled_bridge_preserves_ambiguity: True
- train_eval_name_sets_disjoint: True
- train_eval_object_sets_disjoint: True
- static_objects_separate_from_changed_objects: True
- all state-eval first-slot shortcut: 0.500; second-slot shortcut: 0.500; changed/static same-owner fraction: 0.500

## Files

- common_seen_train_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/common_seen_train.jsonl`
- arm_exposure_only_supervised_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/arms/exposure_only/train_supervised.jsonl`
- arm_exposure_only_unsup_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/arms/exposure_only/train_unsup_text.jsonl`
- arm_heldheld_only_supervised_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/arms/heldheld_only/train_supervised.jsonl`
- arm_heldheld_only_unsup_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/arms/heldheld_only/train_unsup_text.jsonl`
- arm_aligned_state_bridge_supervised_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/arms/aligned_state_bridge/train_supervised.jsonl`
- arm_aligned_state_bridge_unsup_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/arms/aligned_state_bridge/train_unsup_text.jsonl`
- arm_inverted_state_bridge_supervised_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/arms/inverted_state_bridge/train_supervised.jsonl`
- arm_inverted_state_bridge_unsup_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/arms/inverted_state_bridge/train_unsup_text.jsonl`
- arm_decoupled_state_bridge_supervised_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/arms/decoupled_state_bridge/train_supervised.jsonl`
- arm_decoupled_state_bridge_unsup_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/arms/decoupled_state_bridge/train_unsup_text.jsonl`
- arm_mixed_event_bridge_supervised_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/arms/mixed_event_bridge/train_supervised.jsonl`
- arm_mixed_event_bridge_unsup_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/arms/mixed_event_bridge/train_unsup_text.jsonl`
- eval_held_held_coherence_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/eval/held_held_coherence.jsonl`
- eval_mixed_held_seen_orientation_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/eval/mixed_held_seen_orientation.jsonl`
- eval_paired_state_conservation_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/eval/paired_state_conservation.jsonl`
- eval_cross_frame_state_readout_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/eval/cross_frame_state_readout.jsonl`
- eval_name_permutation_counterfactual_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/eval/name_permutation_counterfactual.jsonl`
- audit_json: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/symmetry_substrate_audit.json`
- arm_summary_csv: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/arm_summary.csv`
- eval_summary_csv: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/eval_summary.csv`

