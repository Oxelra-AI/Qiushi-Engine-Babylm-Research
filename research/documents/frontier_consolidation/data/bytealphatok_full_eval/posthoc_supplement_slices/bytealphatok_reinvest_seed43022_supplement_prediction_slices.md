# supplement newline symmetry and posthoc slices Supplement prediction slices — bytealphatok_reinvest_seed43022

This CPU-only join maps official Supplement predictions to the spatial repair route status-tokenizer newline-affected row structure. It is an interpretation layer; official column scores still come from the standard evaluator and pristine collation.

Predictions: `experiments/archive/frontier_consolidation/data/bytealphatok_full_eval/official_outputs/bytealphatok_reinvest_seed43022/Supplement/chck_100M/full_bytealphatok_reinvest_seed43022_Supplement/zero_shot/mlm/blimp/supplement_filtered/predictions.json`
Rows joined: 5218 / 5218 (missing 0, extra 0).
Subtask-macro Supplement accuracy: 59.280593

## By subtask

| subtask | n | correct | accuracy | affected n | affected accuracy | unaffected accuracy |
|---|---:|---:|---:|---:|---:|---:|
| hypernym | 842 | 426 | 50.594 | 0 |  | 50.594 |
| qa_congruence_easy | 64 | 39 | 60.938 | 64 | 60.938 |  |
| qa_congruence_tricky | 165 | 56 | 33.939 | 165 | 33.939 |  |
| subject_aux_inversion | 3867 | 3323 | 85.932 | 0 |  | 85.932 |
| turn_taking | 280 | 182 | 65.000 | 244 | 64.754 | 66.667 |

## Affected versus unaffected

| slice | n | correct | accuracy |
|---|---:|---:|---:|
| affected | 473 | 253 | 53.488 |
| unaffected | 4745 | 3773 | 79.515 |

## Structural slices among spatial repair route status-affected rows

| structural slice | n | correct | accuracy |
|---|---:|---:|---:|
| affected_offsetdiff_local8diff | 9 | 4 | 44.444 |
| affected_offsetdiff_local8same | 95 | 59 | 62.105 |
| affected_same_span_offset | 369 | 190 | 51.491 |
| unaffected | 4745 | 3773 | 79.515 |

## Interpretation

Use this after each compliant endpoint finishes official Supplement scoring. If spatial repair route status and byte-alphabet endpoints differ mostly on `affected` rows, especially QA or turn-taking, the difference is consistent with the tokenizer-surface mechanism quantified in dual compliant tokenizer endpoint policy/43. If they differ mainly on unaffected rows or other columns, the movement reflects broader training/tokenization interactions rather than newline `<unk>` exposure alone.

Full JSON: `experiments/archive/frontier_consolidation/data/bytealphatok_full_eval/posthoc_supplement_slices/bytealphatok_reinvest_seed43022_supplement_prediction_slices.json`
