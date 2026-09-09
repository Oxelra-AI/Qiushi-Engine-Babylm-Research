# earlier analysis exposure-preserving rewired connectivity substrate

This rebuild removes the connectivity substrate construction and audit name-exposure flaw.  Bridge state supervision is identical in the two conditions; held-held comparison exposure is rewired so individual names keep the same role/relation/label counts while exact bridge-pair overlap changes.

## Pair construction

- bridge/base pairs: [['Mira', 'Omar'], ['Noel', 'Iris'], ['Lena', 'Pavel'], ['Rina', 'Tomas'], ['Nia', 'Felix'], ['Ava', 'Jonas'], ['Keira', 'Milo'], ['Sara', 'Theo']]
- rewired comparison pairs: [['Mira', 'Iris'], ['Noel', 'Pavel'], ['Lena', 'Tomas'], ['Rina', 'Felix'], ['Nia', 'Jonas'], ['Ava', 'Milo'], ['Keira', 'Theo'], ['Sara', 'Omar']]
- connected condition: held-held comparisons use the bridge/base pairs
- rewired condition: held-held comparisons use the degree-preserving second-name permutation

## Central structural readout

| arm | condition | common | held-held | bridge | model-train rows | exact bridge/comparison pair overlap | formal assignments | anti-copy opposite | anti-copy same |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned_state_bridge | connected | 2048 | 256 | 512 | 2816 | 8 | 1 | 1.0 | 0.0 |
| aligned_state_bridge | rewired | 2048 | 256 | 512 | 2816 | 0 | 1 | 1.0 | 0.0 |
| inverted_state_bridge | connected | 2048 | 256 | 512 | 2816 | 8 | 1 | 1.0 | 0.0 |
| inverted_state_bridge | rewired | 2048 | 256 | 512 | 2816 | 0 | 1 | 1.0 | 0.0 |
| heldheld_only | connected | 2048 | 256 | 0 | 2304 | 0 | 2 | 1.0 | n/a |
| heldheld_only | rewired | 2048 | 256 | 0 | 2304 | 0 | 2 | 1.0 | n/a |

## Matched exposure checks

| arm | token unigrams | name role/label/relation counts | structural relation/voice/label counts |
|---|---:|---:|---:|
| aligned_state_bridge | True | True | True |
| inverted_state_bridge | True | True | True |
| heldheld_only | True | True | True |

## Global checks
- base_rewired_exact_pair_overlap_zero: True
- same_training_names: True
- first_side_names_preserved: True
- second_side_names_permuted: True
- aligned_state_bridge_token_unigram_counts_equal: True
- aligned_state_bridge_name_role_label_relation_counts_equal: True
- aligned_state_bridge_structural_relation_voice_label_counts_equal: True
- aligned_state_bridge_formal_assignment_count_equal: True
- aligned_state_bridge_supervised_rows_equal: True
- aligned_state_bridge_model_train_rows_equal: True
- aligned_state_bridge_connected_exact_pair_overlap_positive: True
- aligned_state_bridge_rewired_exact_pair_overlap_zero: True
- inverted_state_bridge_token_unigram_counts_equal: True
- inverted_state_bridge_name_role_label_relation_counts_equal: True
- inverted_state_bridge_structural_relation_voice_label_counts_equal: True
- inverted_state_bridge_formal_assignment_count_equal: True
- inverted_state_bridge_supervised_rows_equal: True
- inverted_state_bridge_model_train_rows_equal: True
- inverted_state_bridge_connected_exact_pair_overlap_positive: True
- inverted_state_bridge_rewired_exact_pair_overlap_zero: True
- heldheld_only_token_unigram_counts_equal: True
- heldheld_only_name_role_label_relation_counts_equal: True
- heldheld_only_structural_relation_voice_label_counts_equal: True
- heldheld_only_formal_assignment_count_equal: True
- heldheld_only_supervised_rows_equal: True
- heldheld_only_model_train_rows_equal: True
- heldheld_only_connected_exact_pair_overlap_positive: True
- heldheld_only_rewired_exact_pair_overlap_zero: True

## Eval suites
- heldheld_unseen_edge_closure: 128 rows, true_frac=0.5
- mixed_held_seen_orientation: 512 rows, true_frac=0.5
- paired_state_conservation: 1024 rows, true_frac=0.5
- cross_template_state_readout: 512 rows, true_frac=0.5

## Scientific reading

This file-only construction is the corrected substrate for a future minimal learned comparison.  A useful positive learned result would require the connected condition, but not the rewired condition, to show same-initial changed exact choice, pair-both conservation, and signed mixed held-seen orientation.  If both fail after local train fit, the evidence points away from data topology on this surface and toward an explicit role/entity/state interface.

## Files
- manifest: `experiments/archive/representation_and_objectives/data/rewired_connectivity_substrate/manifest.json`
- connected data: `experiments/archive/representation_and_objectives/data/rewired_connectivity_substrate/connected`
- rewired data: `experiments/archive/representation_and_objectives/data/rewired_connectivity_substrate/rewired`
- eval data: `experiments/archive/representation_and_objectives/data/rewired_connectivity_substrate/eval`
