# permuted companion correspondence control independent audit of permuted companion control

Status: `PERMUTED_COMPANION_INDEPENDENT_AUDIT_PASS`

The written permuted companion correspondence control control preserves source/rewrite multisets and row/filler/order geometry while breaking same-row/same-document source-view correspondence.

## Critical invariants

- assignment_rows_equal_pair_count: True
- target_set_is_all_pairs: True
- donor_set_is_all_pairs: True
- same_pair_assignments: 0
- same_doc_assignments: 0
- same_row_assignments: 0
- donor_doc_in_target_row_docset: 0
- row_assignment_counts_match_view_slots: True
- pool_row_count_matches_view: True
- pool_exact_10M_words: True
- row_length_sequence_matches_view: True
- reconstructed_prefix_matches_permuted_pool: True
- suffix_rows_exactly_equal_view_pool: True
- all_row_word_residuals_zero: True
- source_multiset_digest_matches_view: True
- rewrite_multiset_digest_matches_view: True
- metadata_status: PERMUTED_COMPANION_MAX_ROWHOLDOUT_MATERIALIZED
- metadata_stream_sha256: f5416621a1647f867d2112e1bfcb175b2461ea2c6dbc5413e8ed4daf53122edc
- actual_stream_sha256: f5416621a1647f867d2112e1bfcb175b2461ea2c6dbc5413e8ed4daf53122edc
- metadata_stream_sha_matches_actual: True
- training_order_matches_step256_formula: True

## Token/WWM shift versus MAX view

- tokens_legal16k_minus_max_view: 0
- tokens_legal16k_pct_vs_max_view: 0.0
- tokens_visible_seq256_minus_max_view: 1629
- tokens_visible_seq256_pct_vs_max_view: 0.01139871226039613
- wwm_groups_visible_minus_max_view: 881
- wwm_groups_visible_pct_vs_max_view: 0.008972586862992163
- words_minus_max_view: 0
- words_pct_vs_max_view: 0.0
