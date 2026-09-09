# repaired relation first acquisition synthesis repaired relation-first harness

## Why this repair was necessary

The multiseed and relation first relation-first extraction is useful, but its first scorer/trainer did not preserve the previous experimental substrate. Generic `AutoModelForMaskedLM` loading without trusted custom loading selects stock `DebertaV2ForMaskedLM`; the answer-only trainer also optimized `model.parameters()`. The multiseed and relation first construction selected separate new values for the two update recipients, so changing recipient also changed the competing answer phrase. Step039b masked candidate tokens one at a time and allowed a token-search fallback that could drop the first candidate token. This script repairs those problems before interpreting acquisition.

## Repaired construction

- Pairs: 120 (90 train, 30 held); by relation: {'death_place': 108, 'birthplace': 11, 'founded_year': 1}
- Rows per role: {'UPDATE': 240, 'RETAIN': 240, 'NEUTRAL': 240}
- Original multiseed and relation first pairs with different new values now repaired: 120
- For query A, `query_a_update_a` and `query_a_update_b` have the same query entity, source answer, and shared new-value candidate; only the update recipient changes. Query B is analyzed the same way.
- Pair success is `min(u_a, r_a, u_b, r_b) > 0`, not gamma after averaging.

## Bounded answer-only private-adapter pilot

- Trainable params: 995584 in 48 tensors; non-private trainable tensors 0
- Final train: {'n_pairs': 90, 'n_four_condition_success': 81, 'n_query_orientation_success': 171, 'n_query_orientations': 180, 'mean_U': 5.261062518479163, 'mean_R': 8.367170453195081, 'mean_beta_pair_average': 6.814116485837121, 'mean_abs_alpha_pair_average': 2.065453360137078, 'mean_min_four_signed_margin': 2.0558593218466394}
- Final held: {'n_pairs': 30, 'n_four_condition_success': 28, 'n_query_orientation_success': 58, 'n_query_orientations': 60, 'mean_U': 5.670430682427282, 'mean_R': 7.206103460261689, 'mean_beta_pair_average': 6.438267071344486, 'mean_abs_alpha_pair_average': 1.7682361124285995, 'mean_min_four_signed_margin': 2.0355687225183807}

## Files

- `experiments/archive/functional_learning/data/relation_first_repaired_e80/repaired_pairs.jsonl`
- `experiments/archive/functional_learning/data/relation_first_repaired_e80/repaired_scoring_rows.jsonl`
