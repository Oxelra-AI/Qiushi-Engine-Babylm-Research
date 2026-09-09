# symmetry identification route v2 repaired symmetry-identification substrate

CPU/file-only rebuilt substrate after independent_review found the natural-verb prototype unsafe. No model loading, training, official evaluation, GPU work, upload, or leaderboard action occurred.

## Scientific object

Held relation words are nonce lexical components in otherwise natural possession-state language. Held-held training rows constrain only relative orientation and admit exactly the two global assignments related by a role swap. Sparse aligned or mixed bridges identify the true assignment; inverted bridges identify the opposite assignment; the neutral decoupled control leaves the same two assignments.

## Formal sign assignments

- true assignment: `{'h0_dax': 1, 'h1_mep': 1, 'h2_norp': 0, 'h3_ziv': 0}`
- inverted assignment: `{'h0_dax': 0, 'h1_mep': 0, 'h2_norp': 1, 'h3_ziv': 1}`

| arm | supervised rows | words | satisfying assignments | true? | inverted? | slot0 true changed | surface-first true changed |
|---|---:|---:|---:|---|---|---:|---:|
| exposure_only | 0 | 0 | 16 | True | True | nan | nan |
| heldheld_only | 48 | 2832 | 2 | True | True | nan | nan |
| aligned_state_bridge | 80 | 4208 | 1 | True | False | 0.500 | 0.500 |
| inverted_state_bridge | 80 | 4208 | 1 | False | True | 0.500 | 0.500 |
| neutral_decoupled | 80 | 4976 | 2 | True | True | 0.500 | 0.000 |
| mixed_event_bridge | 56 | 3224 | 1 | True | False | nan | nan |

## Evaluation suites

| suite | rows | true frac | slot0 true changed | surface-first true changed | slot shortcut all | same changed/static owner |
|---|---:|---:|---:|---:|---:|---:|
| heldheld_unseen_edge_closure | 24 | 0.500 | nan | nan | nan | nan |
| mixed_held_seen_orientation | 64 | 0.500 | nan | nan | nan | nan |
| paired_state_conservation | 128 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| cross_template_state_readout | 64 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| name_permutation_counterfactual | 64 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 |

## Repaired shortcut checks

- train_eval_names_disjoint: True
- train_eval_changed_objects_disjoint: True
- train_eval_static_objects_disjoint: True
- changed_static_objects_disjoint_within_train: True
- changed_static_objects_disjoint_within_eval: True
- held surface leak scan: 0 hits across 1336 held-related text fields
- heldheld_only_has_exactly_global_Z2: True
- aligned_state_identifies_true: True
- inverted_state_identifies_inverted: True
- neutral_decoupled_preserves_Z2: True
- mixed_event_identifies_true: True

## Remaining scientific caution

This v2 substrate removes the main natural-verb leakage by using nonce held relations. It therefore tests a controlled lexical-component identifiability law, not full natural verb understanding. A positive learned result would justify a later ecological shadow with rare natural verbs only if zero-shot/exposure-only natural orientation is measured and residual ambiguity remains.

## Files

- common_seen_train_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/common_seen_train.jsonl`
- arm_exposure_only_supervised_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/arms/exposure_only/train_supervised.jsonl`
- arm_exposure_only_unsup_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/arms/exposure_only/train_unsup_text.jsonl`
- arm_heldheld_only_supervised_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/arms/heldheld_only/train_supervised.jsonl`
- arm_heldheld_only_unsup_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/arms/heldheld_only/train_unsup_text.jsonl`
- arm_aligned_state_bridge_supervised_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/arms/aligned_state_bridge/train_supervised.jsonl`
- arm_aligned_state_bridge_unsup_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/arms/aligned_state_bridge/train_unsup_text.jsonl`
- arm_inverted_state_bridge_supervised_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/arms/inverted_state_bridge/train_supervised.jsonl`
- arm_inverted_state_bridge_unsup_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/arms/inverted_state_bridge/train_unsup_text.jsonl`
- arm_neutral_decoupled_supervised_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/arms/neutral_decoupled/train_supervised.jsonl`
- arm_neutral_decoupled_unsup_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/arms/neutral_decoupled/train_unsup_text.jsonl`
- arm_mixed_event_bridge_supervised_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/arms/mixed_event_bridge/train_supervised.jsonl`
- arm_mixed_event_bridge_unsup_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/arms/mixed_event_bridge/train_unsup_text.jsonl`
- eval_heldheld_unseen_edge_closure_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/eval/heldheld_unseen_edge_closure.jsonl`
- eval_mixed_held_seen_orientation_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/eval/mixed_held_seen_orientation.jsonl`
- eval_paired_state_conservation_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/eval/paired_state_conservation.jsonl`
- eval_cross_template_state_readout_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/eval/cross_template_state_readout.jsonl`
- eval_name_permutation_counterfactual_jsonl: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/eval/name_permutation_counterfactual.jsonl`
- audit_json: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/symmetry_substrate_v2_audit.json`
- arm_summary_csv: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/arm_summary.csv`
- eval_summary_csv: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2/eval_summary.csv`
