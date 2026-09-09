# earlier analysis relation-graph connectivity substrate

This construction preserves the names, exact bridge state rows, common seen-coordinate rows, exact pair degrees, token exposure, and per-name role/label counts.  It rewires only the held-held relation graph while preserving each held relation's degree and side-position counts.

## Relation edge sets
- rel_connected: [['h0_dax', 'h1_mep'], ['h1_mep', 'h2_norp'], ['h2_norp', 'h3_ziv'], ['h3_ziv', 'h0_dax']] -> components [['h0_dax', 'h1_mep', 'h2_norp', 'h3_ziv']]
- rel_disconnected: [['h0_dax', 'h2_norp'], ['h2_norp', 'h0_dax'], ['h1_mep', 'h3_ziv'], ['h3_ziv', 'h1_mep']] -> components [['h0_dax', 'h2_norp'], ['h1_mep', 'h3_ziv']]
- degree counts connected/disconnected: {'h0_dax': 2, 'h1_mep': 2, 'h2_norp': 2, 'h3_ziv': 2} / {'h0_dax': 2, 'h1_mep': 2, 'h2_norp': 2, 'h3_ziv': 2}

## Central readout

| arm | condition | rows | held-held | bridge | formal assignments | true satisfies | inverted satisfies | anti-copy same | anti-copy opposite |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned_state_bridge | rel_connected | 1792 | 256 | 512 | 1 | True | False | 0.0 | 1.0 |
| aligned_state_bridge | rel_disconnected | 1792 | 256 | 512 | 2 | True | False | 0.0 | 1.0 |
| inverted_state_bridge | rel_connected | 1792 | 256 | 512 | 1 | False | True | 0.0 | 1.0 |
| inverted_state_bridge | rel_disconnected | 1792 | 256 | 512 | 2 | False | True | 0.0 | 1.0 |
| heldheld_only | rel_connected | 1280 | 256 | 0 | 2 | True | True | n/a | 1.0 |
| heldheld_only | rel_disconnected | 1280 | 256 | 0 | 4 | True | True | n/a | 1.0 |

## Matched checks

| arm | token unigrams | name role/label | exact pair degree | structural row counts |
|---|---:|---:|---:|---:|
| aligned_state_bridge | True | False | True | False |
| inverted_state_bridge | True | False | True | False |
| heldheld_only | True | False | True | False |

Structural row counts are intentionally not equal because the relation-pair edge set is the intervention.  The degree/side counts of individual relations are equal.

## Global checks
- same_exact_pairs: True
- connected_one_relation_component: True
- disconnected_two_relation_components: True
- relation_degrees_equal: True
- relation_side1_counts_equal: True
- relation_side2_counts_equal: True
- aligned_state_bridge_token_unigram_counts_equal: True
- aligned_state_bridge_name_role_label_counts_equal: False
- aligned_state_bridge_exact_pair_degree_counts_equal: True
- aligned_state_bridge_rows_equal: True
- aligned_state_bridge_supervised_rows_equal: True
- aligned_state_bridge_tokens_total_equal: True
- inverted_state_bridge_token_unigram_counts_equal: True
- inverted_state_bridge_name_role_label_counts_equal: False
- inverted_state_bridge_exact_pair_degree_counts_equal: True
- inverted_state_bridge_rows_equal: True
- inverted_state_bridge_supervised_rows_equal: True
- inverted_state_bridge_tokens_total_equal: True
- heldheld_only_token_unigram_counts_equal: True
- heldheld_only_name_role_label_counts_equal: False
- heldheld_only_exact_pair_degree_counts_equal: True
- heldheld_only_rows_equal: True
- heldheld_only_supervised_rows_equal: True
- heldheld_only_tokens_total_equal: True

## Eval suites
- heldheld_unseen_edge_closure: 128 rows, true_frac=0.5
- mixed_held_seen_orientation: 512 rows, true_frac=0.5
- paired_state_conservation: 1024 rows, true_frac=0.5
- cross_template_state_readout: 512 rows, true_frac=0.5

## Scientific reading

This is a clean CPU substrate for a future minimal learned test of relation-coordinate connectivity.  Because rel_disconnected has no relation path from bridge anchors h0/h2 to targets h1/h3, target transfer is not formally determined there; the useful contrast is whether rel_connected can exploit the path without any extra filler exposure, while rel_disconnected should at most learn direct anchors.  A positive result still requires exact same-initial changed choice and pair-both conservation, plus signed mixed held-seen orientation.  If rel_connected fails after local fit, the evidence points toward missing architectural factorization rather than more counterexamples.

## Files
- manifest: `experiments/archive/representation_and_objectives/data/relation_graph_connectivity_substrate/manifest.json`
- data root: `experiments/archive/representation_and_objectives/data/relation_graph_connectivity_substrate`
