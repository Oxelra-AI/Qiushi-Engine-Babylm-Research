# supplement newline symmetry and posthoc slices Supplement row-slice comparison

This comparison uses only existing saved Supplement predictions. It validates the post-evaluation slice tool and gives a reference for interpreting future compliant-tokenizer endpoint differences.

## Overall and affected-row surface

| tag | Supplement macro | affected acc | unaffected acc | affected n | unaffected n |
|---|---:|---:|---:|---:|---:|
| oldtok_compact_view_reinvest_seed43022 | 63.275764 | 58.140 | 77.724 | 473 | 4745 |
| oldtok_qwen_clean_aligned_seed43022 | 62.842544 | 59.619 | 75.111 | 473 | 4745 |

## Pairwise deltas in percentage points (first minus second)

| first - second | Supplement macro | affected acc | unaffected acc | qa easy aff | qa tricky aff | turn-taking aff | hypernym | subject-aux |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oldtok_compact_view_reinvest_seed43022 - oldtok_qwen_clean_aligned_seed43022 | 0.433 | -1.480 | 2.613 | 1.562 | -3.030 | -1.230 | 1.069 | 2.922 |

## Interpretation

For the two already-existing old-tokenizer endpoints, the compact-view-reinvest model's Supplement macro advantage over clean-Qwen is not carried by the spatial repair route status newline-affected rows: affected-row accuracy is slightly lower, while unaffected-row accuracy is higher. Thus future spatial repair route status-versus-byte-alphabet compliant endpoint differences should be localized with this script rather than attributed wholesale to newline `<unk>` exposure.

Full JSON: `experiments/archive/frontier_consolidation/data/supplement_prediction_slices/existing_oldtok_reinvest_vs_clean_comparison.json`
