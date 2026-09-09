# equivariant symmetry repair and macro context equivariant symmetry-identification substrate

This CPU/file-only construction repairs the symmetry pilot boundary findings 0.750 order solution by replacing order-free conjoined nonce events with active/passive syntactic role orbits, a balanced held-held relation graph, and static-owner counterbalancing for unaffected facts. The learner still sees a reusable argument interface, but visible name order, event side, voice, relation-pair orientation type, changed-owner position, static-owner position, and candidate position are balanced against labels.

## Formal assignments

- true assignment: `{'h0_dax': 1, 'h1_mep': 1, 'h2_norp': 0, 'h3_ziv': 0}`
- inverted assignment: `{'h0_dax': 0, 'h1_mep': 0, 'h2_norp': 1, 'h3_ziv': 1}`

| arm | supervised rows | satisfying assignments | true? | inverted? | best comparison order rule | best changed-state order rule | slot0 true changed | surface-first true changed |
|---|---:|---:|---|---|---:|---:|---:|---:|
| exposure_only | 0 | 16 | True | True | nan | nan | nan | nan |
| heldheld_only | 192 | 2 | True | True | 0.500 | nan | nan | nan |
| aligned_state_bridge | 320 | 1 | True | False | 0.500 | 0.500 | 0.500 | 0.500 |
| inverted_state_bridge | 320 | 1 | False | True | 0.500 | 0.500 | 0.500 | 0.500 |
| neutral_decoupled | 704 | 2 | True | True | 0.500 | 0.500 | 0.500 | 0.500 |
| mixed_event_bridge | 256 | 1 | True | False | 0.500 | nan | nan | nan |

## Evaluation surface readout

| suite | rows | true frac | best comparison order rule | best changed-state order rule | slot0 true changed | surface-first true changed | same changed/static owner |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldheld_unseen_edge_closure | 128 | 0.500 | 0.500 | nan | nan | nan | nan |
| mixed_held_seen_orientation | 512 | 0.500 | 0.500 | nan | nan | nan | nan |
| paired_state_conservation | 512 | 0.500 | nan | 0.500 | 0.500 | 0.500 | 0.500 |
| cross_template_state_readout | 256 | 0.500 | nan | 0.500 | 0.500 | 0.500 | 0.500 |
| name_permutation_counterfactual | 256 | 0.500 | nan | 0.500 | 0.500 | 0.500 | 0.500 |

## Global checks

- train_eval_names_disjoint: True
- train_eval_changed_objects_disjoint: True
- train_eval_static_objects_disjoint: True
- changed_static_objects_disjoint_within_train: True
- changed_static_objects_disjoint_within_eval: True
- held surface transfer-word scan: 0 hits across 5120 held-related fields
- heldheld_only_has_exactly_global_Z2: True
- aligned_state_identifies_true: True
- inverted_state_identifies_inverted: True
- neutral_decoupled_preserves_Z2: True
- mixed_event_identifies_true: True
- mixed_eval_order_rules_at_chance: True
- changed_state_eval_order_rules_at_chance: True

## Files

- common_seen_train_jsonl: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/common_seen_train.jsonl`
- arm_exposure_only_supervised_jsonl: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/arms/exposure_only/train_supervised.jsonl`
- arm_exposure_only_unsup_jsonl: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/arms/exposure_only/train_unsup_text.jsonl`
- arm_heldheld_only_supervised_jsonl: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/arms/heldheld_only/train_supervised.jsonl`
- arm_heldheld_only_unsup_jsonl: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/arms/heldheld_only/train_unsup_text.jsonl`
- arm_aligned_state_bridge_supervised_jsonl: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/arms/aligned_state_bridge/train_supervised.jsonl`
- arm_aligned_state_bridge_unsup_jsonl: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/arms/aligned_state_bridge/train_unsup_text.jsonl`
- arm_inverted_state_bridge_supervised_jsonl: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/arms/inverted_state_bridge/train_supervised.jsonl`
- arm_inverted_state_bridge_unsup_jsonl: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/arms/inverted_state_bridge/train_unsup_text.jsonl`
- arm_neutral_decoupled_supervised_jsonl: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/arms/neutral_decoupled/train_supervised.jsonl`
- arm_neutral_decoupled_unsup_jsonl: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/arms/neutral_decoupled/train_unsup_text.jsonl`
- arm_mixed_event_bridge_supervised_jsonl: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/arms/mixed_event_bridge/train_supervised.jsonl`
- arm_mixed_event_bridge_unsup_jsonl: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/arms/mixed_event_bridge/train_unsup_text.jsonl`
- eval_heldheld_unseen_edge_closure_jsonl: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/eval/heldheld_unseen_edge_closure.jsonl`
- eval_mixed_held_seen_orientation_jsonl: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/eval/mixed_held_seen_orientation.jsonl`
- eval_paired_state_conservation_jsonl: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/eval/paired_state_conservation.jsonl`
- eval_cross_template_state_readout_jsonl: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/eval/cross_template_state_readout.jsonl`
- eval_name_permutation_counterfactual_jsonl: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/eval/name_permutation_counterfactual.jsonl`
- report_json: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/equivariant_substrate_report.json`
- arm_summary_csv: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/arm_summary.csv`
- eval_summary_csv: `experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate/eval_summary.csv`
