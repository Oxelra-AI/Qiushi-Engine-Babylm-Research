# earlier analysis neutral-block pair-binding substrate

This is the corrected no-GPU substrate after the connectivity substrate construction and audit exposure flaw.  Both conditions have identical model-input text multisets, common seen-coordinate rows, bridge rows, token counts, exact pair degrees, per-name role/label counts, relation/voice/label counts, and no duplicate input with conflicting labels.  The only intended difference is which exact participant pairs carry orientation-informative held-held labels; the complementary pairs carry rank-zero neutral labels with the same surface exposure.

## Pair rewiring
- B bridge/base pairs: [['Mira', 'Omar'], ['Noel', 'Iris'], ['Lena', 'Pavel'], ['Rina', 'Tomas'], ['Nia', 'Felix'], ['Ava', 'Jonas'], ['Keira', 'Milo'], ['Sara', 'Theo']]
- R rewired pairs: [['Mira', 'Iris'], ['Noel', 'Pavel'], ['Lena', 'Tomas'], ['Rina', 'Felix'], ['Nia', 'Jonas'], ['Ava', 'Milo'], ['Keira', 'Theo'], ['Sara', 'Omar']]
- pair_connected: B informative, R neutral
- pair_rewired: B neutral, R informative

## Central readout

| arm | condition | rows | heldheld info | heldheld neutral | bridge | formal assignments | neutral rank-zero | anti-copy same | anti-copy opposite |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned_state_bridge | pair_connected | 3072 | 256 | 256 | 512 | 1 | True | 0.0 | 1.0 |
| aligned_state_bridge | pair_rewired | 3072 | 256 | 256 | 512 | 1 | True | 0.0 | 1.0 |
| inverted_state_bridge | pair_connected | 3072 | 256 | 256 | 512 | 1 | True | 0.0 | 1.0 |
| inverted_state_bridge | pair_rewired | 3072 | 256 | 256 | 512 | 1 | True | 0.0 | 1.0 |
| heldheld_only | pair_connected | 2560 | 256 | 256 | 0 | 2 | True | n/a | 1.0 |
| heldheld_only | pair_rewired | 2560 | 256 | 256 | 0 | 2 | True | n/a | 1.0 |

## Matched checks

| arm | input text multiset | token unigrams | name role/label | exact pair degree | pair relation/voice/label | structural no-neutral-flag | input+label differs | geometry-label differs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned_state_bridge | True | True | True | True | True | True | True | True |
| inverted_state_bridge | True | True | True | True | True | True | True | True |
| heldheld_only | True | True | True | True | True | True | True | True |

The final two columns are expected to differ: they are the intervention assigning orientation-informative versus rank-zero labels to B or R while preserving the same inputs and simple exposure counts.

## Global checks
- B_R_exact_pair_overlap_zero: True
- B_R_same_individual_name_set: True
- aligned_state_bridge_input_text_multiset_equal: True
- aligned_state_bridge_token_unigram_counts_equal: True
- aligned_state_bridge_name_role_label_counts_equal: True
- aligned_state_bridge_exact_pair_degree_counts_equal: True
- aligned_state_bridge_pair_label_voice_relation_counts_equal: True
- aligned_state_bridge_structural_counts_without_neutral_flag_equal: True
- aligned_state_bridge_input_text_plus_label_multiset_not_equal_expected: True
- aligned_state_bridge_pair_geometry_label_counts_differ_expected: True
- pair_connected_aligned_state_bridge_no_duplicate_input_label_conflicts: True
- pair_connected_aligned_state_bridge_neutral_rank_zero: True
- pair_rewired_aligned_state_bridge_no_duplicate_input_label_conflicts: True
- pair_rewired_aligned_state_bridge_neutral_rank_zero: True
- aligned_state_bridge_formal_assignment_count_equal: True
- aligned_state_bridge_rows_equal: True
- aligned_state_bridge_train_token_total_equal: True
- inverted_state_bridge_input_text_multiset_equal: True
- inverted_state_bridge_token_unigram_counts_equal: True
- inverted_state_bridge_name_role_label_counts_equal: True
- inverted_state_bridge_exact_pair_degree_counts_equal: True
- inverted_state_bridge_pair_label_voice_relation_counts_equal: True
- inverted_state_bridge_structural_counts_without_neutral_flag_equal: True
- inverted_state_bridge_input_text_plus_label_multiset_not_equal_expected: True
- inverted_state_bridge_pair_geometry_label_counts_differ_expected: True
- pair_connected_inverted_state_bridge_no_duplicate_input_label_conflicts: True
- pair_connected_inverted_state_bridge_neutral_rank_zero: True
- pair_rewired_inverted_state_bridge_no_duplicate_input_label_conflicts: True
- pair_rewired_inverted_state_bridge_neutral_rank_zero: True
- inverted_state_bridge_formal_assignment_count_equal: True
- inverted_state_bridge_rows_equal: True
- inverted_state_bridge_train_token_total_equal: True
- heldheld_only_input_text_multiset_equal: True
- heldheld_only_token_unigram_counts_equal: True
- heldheld_only_name_role_label_counts_equal: True
- heldheld_only_exact_pair_degree_counts_equal: True
- heldheld_only_pair_label_voice_relation_counts_equal: True
- heldheld_only_structural_counts_without_neutral_flag_equal: True
- heldheld_only_input_text_plus_label_multiset_not_equal_expected: True
- heldheld_only_pair_geometry_label_counts_differ_expected: True
- pair_connected_heldheld_only_no_duplicate_input_label_conflicts: True
- pair_connected_heldheld_only_neutral_rank_zero: True
- pair_rewired_heldheld_only_no_duplicate_input_label_conflicts: True
- pair_rewired_heldheld_only_neutral_rank_zero: True
- heldheld_only_formal_assignment_count_equal: True
- heldheld_only_rows_equal: True
- heldheld_only_train_token_total_equal: True

## Eval suites
- heldheld_unseen_edge_closure: 128 rows, true_frac=0.5
- mixed_held_seen_orientation: 512 rows, true_frac=0.5
- paired_state_conservation: 1024 rows, true_frac=0.5
- cross_template_state_readout: 512 rows, true_frac=0.5

## Scientific reading

This substrate no longer supports the connectivity substrate construction and audit criticism that bridge-supervised names or dyads differ in exposure.  A future learned comparison would be the minimum-cost test of pair-binding alignment: whether the same orientation information becomes reusable when the informative held-held labels sit on the exact bridge-supervised dyads rather than on degree-matched rewired dyads.  The result would be narrower than full graph connectivity, but cleaner: it addresses whether finite experience must align discriminative relation constraints with the learner's pair-binding interface.  A positive result still needs three simultaneous readouts: same-initial changed exact choice, pair-both conservation, and aligned-vs-inverted signed mixed margins.  A null after local train fit would again push toward architectural role/entity/state factorization.

## Files
- manifest: `experiments/archive/representation_and_objectives/data/pair_binding_neutral_substrate/manifest.json`
- data root: `experiments/archive/representation_and_objectives/data/pair_binding_neutral_substrate`
