# earlier analysis rel_eq0 operation-content partition

On rel_eq0 flips where chck82 is correct and the endpoint is wrong, pred_minus_gold operation-item/token hits tests whether the wrong option contains more irrelevant-operation-mentioned content than the gold option. Positive values support operation-content pull; near-zero or mixed values keep the remaining damage as a broader initial-state confidence loss.

rel_eq0 items in official no-nothing universe: `1541`

## Summary

| label | group | n | pred_item_hits | gold_item_hits | pred-gold item | pred_more_items | pred_token_hits | gold_token_hits | pred-gold token | pred_more_tokens | net_loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coherent86 | flips_base_correct_to_label_wrong | 15 | 0.400 | 0.000 | 0.400 | 26.667 | 1.933 | 1.400 | 0.533 | 40.000 | 4 |
| coherent86 | reverse_flips_base_wrong_to_label_correct | 11 | 0.000 | 0.000 | 0.000 | 0.000 | 0.818 | 0.818 | 0.000 | 0.000 |  |
| coherent86 | all_rel_eq0_predictions | 1541 | 0.339 | 0.000 | 0.339 | 21.609 | 1.262 | 0.933 | 0.329 | 21.415 |  |
| coherent86 | flips_irrelevant_ops_0 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| coherent86 | flips_irrelevant_ops_1-3 | 6 | 0.167 | 0.000 | 0.167 | 16.667 | 1.667 | 1.500 | 0.167 | 16.667 |  |
| coherent86 | flips_irrelevant_ops_4-6 | 6 | 0.833 | 0.000 | 0.833 | 50.000 | 2.500 | 1.667 | 0.833 | 66.667 |  |
| coherent86 | flips_irrelevant_ops_7+ | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 2.000 | 1.000 | 1.000 | 50.000 |  |
| binding_ep25_alpha0p50 | flips_base_correct_to_label_wrong | 199 | 0.764 | 0.000 | 0.764 | 46.734 | 1.714 | 0.985 | 0.729 | 45.729 | 109 |
| binding_ep25_alpha0p50 | reverse_flips_base_wrong_to_label_correct | 90 | 0.000 | 0.000 | 0.000 | 0.000 | 0.622 | 0.622 | 0.000 | 0.000 |  |
| binding_ep25_alpha0p50 | all_rel_eq0_predictions | 1541 | 0.456 | 0.000 | 0.456 | 27.969 | 1.371 | 0.933 | 0.438 | 27.385 |  |
| binding_ep25_alpha0p50 | flips_irrelevant_ops_0 | 27 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| binding_ep25_alpha0p50 | flips_irrelevant_ops_1-3 | 62 | 0.387 | 0.000 | 0.387 | 24.194 | 1.032 | 0.742 | 0.290 | 24.194 |  |
| binding_ep25_alpha0p50 | flips_irrelevant_ops_4-6 | 62 | 1.048 | 0.000 | 1.048 | 69.355 | 2.081 | 1.081 | 1.000 | 64.516 |  |
| binding_ep25_alpha0p50 | flips_irrelevant_ops_7+ | 48 | 1.312 | 0.000 | 1.312 | 72.917 | 3.083 | 1.729 | 1.354 | 75.000 |  |
| binding_ep25_alpha0p75 | flips_base_correct_to_label_wrong | 292 | 0.747 | 0.000 | 0.747 | 43.836 | 1.729 | 0.986 | 0.743 | 42.808 | 173 |
| binding_ep25_alpha0p75 | reverse_flips_base_wrong_to_label_correct | 119 | 0.000 | 0.000 | 0.000 | 0.000 | 0.798 | 0.798 | 0.000 | 0.000 |  |
| binding_ep25_alpha0p75 | all_rel_eq0_predictions | 1541 | 0.500 | 0.000 | 0.500 | 30.305 | 1.411 | 0.933 | 0.479 | 29.721 |  |
| binding_ep25_alpha0p75 | flips_irrelevant_ops_0 | 42 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| binding_ep25_alpha0p75 | flips_irrelevant_ops_1-3 | 105 | 0.381 | 0.000 | 0.381 | 23.810 | 1.029 | 0.705 | 0.324 | 22.857 |  |
| binding_ep25_alpha0p75 | flips_irrelevant_ops_4-6 | 85 | 1.106 | 0.000 | 1.106 | 71.765 | 2.224 | 1.153 | 1.071 | 67.059 |  |
| binding_ep25_alpha0p75 | flips_irrelevant_ops_7+ | 60 | 1.400 | 0.000 | 1.400 | 70.000 | 3.467 | 1.933 | 1.533 | 73.333 |  |
| binding_ep25_alpha1p00 | flips_base_correct_to_label_wrong | 386 | 0.674 | 0.000 | 0.674 | 40.933 | 1.630 | 0.969 | 0.661 | 39.637 | 250 |
| binding_ep25_alpha1p00 | reverse_flips_base_wrong_to_label_correct | 136 | 0.000 | 0.000 | 0.000 | 0.000 | 0.735 | 0.735 | 0.000 | 0.000 |  |
| binding_ep25_alpha1p00 | all_rel_eq0_predictions | 1541 | 0.507 | 0.000 | 0.507 | 30.824 | 1.410 | 0.933 | 0.478 | 30.240 |  |
| binding_ep25_alpha1p00 | flips_irrelevant_ops_0 | 57 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| binding_ep25_alpha1p00 | flips_irrelevant_ops_1-3 | 156 | 0.417 | 0.000 | 0.417 | 26.282 | 1.071 | 0.686 | 0.385 | 25.000 |  |
| binding_ep25_alpha1p00 | flips_irrelevant_ops_4-6 | 111 | 1.000 | 0.000 | 1.000 | 66.667 | 2.243 | 1.306 | 0.937 | 60.360 |  |
| binding_ep25_alpha1p00 | flips_irrelevant_ops_7+ | 62 | 1.355 | 0.000 | 1.355 | 69.355 | 3.435 | 1.968 | 1.468 | 75.806 |  |

## Files

- Detail flips/reverse flips: `experiments/archive/relation_learning/data/rel_eq0_operation_content_partition/operation_content_detail.csv`
- All rel_eq0 prediction rows: `experiments/archive/relation_learning/data/rel_eq0_operation_content_partition/operation_content_all_rel_eq0.csv`
- Summary CSV: `experiments/archive/relation_learning/data/rel_eq0_operation_content_partition/operation_content_summary.csv`
