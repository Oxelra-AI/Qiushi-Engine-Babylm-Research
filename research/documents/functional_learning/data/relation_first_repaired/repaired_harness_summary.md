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
- Final train: {'n_pairs': 90, 'n_four_condition_success': 2, 'n_query_orientation_success': 14, 'n_query_orientations': 180, 'mean_U': 1.1225840999041814, 'mean_R': -0.8088646101471062, 'mean_beta_pair_average': 0.15685974487853754, 'mean_abs_alpha_pair_average': 1.1497722737729137, 'mean_min_four_signed_margin': -1.7789889916185617}
- Final held: {'n_pairs': 30, 'n_four_condition_success': 0, 'n_query_orientation_success': 4, 'n_query_orientations': 60, 'mean_U': 1.3177178345244223, 'mean_R': -0.8929451443007889, 'mean_beta_pair_average': 0.2123863451118167, 'mean_abs_alpha_pair_average': 1.5252211829830453, 'mean_min_four_signed_margin': -2.3939983263745024}

## Files

- `experiments/archive/functional_learning/data/relation_first_repaired/repaired_pairs.jsonl`
- `experiments/archive/functional_learning/data/relation_first_repaired/repaired_scoring_rows.jsonl`
- `experiments/archive/functional_learning/data/relation_first_repaired/trusted_parent_summary.json`
- `experiments/archive/functional_learning/data/relation_first_repaired/trusted_parent_pair_metrics.jsonl`
