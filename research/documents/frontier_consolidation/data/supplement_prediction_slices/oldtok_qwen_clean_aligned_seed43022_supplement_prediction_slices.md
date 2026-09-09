# supplement newline symmetry and posthoc slices Supplement prediction slices — oldtok_qwen_clean_aligned_seed43022

This CPU-only join maps official Supplement predictions to the spatial repair route status-tokenizer newline-affected row structure. It is an interpretation layer; official column scores still come from the standard evaluator and pristine collation.

Predictions: `experiments/archive/compact_experience/data/full_eval/official_outputs/qwen_clean_aligned/Supplement/chck_100M/full_qwen_clean_aligned_Supplement/zero_shot/mlm/blimp/supplement_filtered/predictions.json`
Rows joined: 5218 / 5218 (missing 0, extra 0).
Subtask-macro Supplement accuracy: 62.842544

## By subtask

| subtask | n | correct | accuracy | affected n | affected accuracy | unaffected accuracy |
|---|---:|---:|---:|---:|---:|---:|
| hypernym | 842 | 416 | 49.406 | 0 |  | 49.406 |
| qa_congruence_easy | 64 | 45 | 70.312 | 64 | 70.312 |  |
| qa_congruence_tricky | 165 | 81 | 49.091 | 165 | 49.091 |  |
| subject_aux_inversion | 3867 | 3123 | 80.760 | 0 |  | 80.760 |
| turn_taking | 280 | 181 | 64.643 | 244 | 63.934 | 69.444 |

## Affected versus unaffected

| slice | n | correct | accuracy |
|---|---:|---:|---:|
| affected | 473 | 282 | 59.619 |
| unaffected | 4745 | 3564 | 75.111 |

## Structural slices among spatial repair route status-affected rows

| structural slice | n | correct | accuracy |
|---|---:|---:|---:|
| affected_offsetdiff_local8diff | 9 | 5 | 55.556 |
| affected_offsetdiff_local8same | 95 | 58 | 61.053 |
| affected_same_span_offset | 369 | 219 | 59.350 |
| unaffected | 4745 | 3564 | 75.111 |

## Interpretation

Use this after each compliant endpoint finishes official Supplement scoring. If spatial repair route status and byte-alphabet endpoints differ mostly on `affected` rows, especially QA or turn-taking, the difference is consistent with the tokenizer-surface mechanism quantified in dual compliant tokenizer endpoint policy/43. If they differ mainly on unaffected rows or other columns, the movement reflects broader training/tokenization interactions rather than newline `<unk>` exposure alone.

Full JSON: `experiments/archive/frontier_consolidation/data/supplement_prediction_slices/oldtok_qwen_clean_aligned_seed43022_supplement_prediction_slices.json`
