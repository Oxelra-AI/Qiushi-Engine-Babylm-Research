# supplement newline symmetry and posthoc slices Supplement prediction slices — complianttok_reinvest_seed43022

This CPU-only join maps official Supplement predictions to the spatial repair route status-tokenizer newline-affected row structure. It is an interpretation layer; official column scores still come from the standard evaluator and pristine collation.

Predictions: `experiments/archive/frontier_consolidation/data/compliant_full_eval/official_outputs/complianttok_reinvest_seed43022/Supplement/chck_100M/full_complianttok_reinvest_seed43022_Supplement/zero_shot/mlm/blimp/supplement_filtered/predictions.json`
Rows joined: 5218 / 5218 (missing 0, extra 0).
Subtask-macro Supplement accuracy: 61.165661

## By subtask

| subtask | n | correct | accuracy | affected n | affected accuracy | unaffected accuracy |
|---|---:|---:|---:|---:|---:|---:|
| hypernym | 842 | 436 | 51.781 | 0 |  | 51.781 |
| qa_congruence_easy | 64 | 38 | 59.375 | 64 | 59.375 |  |
| qa_congruence_tricky | 165 | 70 | 42.424 | 165 | 42.424 |  |
| subject_aux_inversion | 3867 | 3291 | 85.105 | 0 |  | 85.105 |
| turn_taking | 280 | 188 | 67.143 | 244 | 66.803 | 69.444 |

## Affected versus unaffected

| slice | n | correct | accuracy |
|---|---:|---:|---:|
| affected | 473 | 271 | 57.294 |
| unaffected | 4745 | 3752 | 79.073 |

## Structural slices among spatial repair route status-affected rows

| structural slice | n | correct | accuracy |
|---|---:|---:|---:|
| affected_offsetdiff_local8diff | 9 | 5 | 55.556 |
| affected_offsetdiff_local8same | 95 | 63 | 66.316 |
| affected_same_span_offset | 369 | 203 | 55.014 |
| unaffected | 4745 | 3752 | 79.073 |

## Interpretation

Use this after each compliant endpoint finishes official Supplement scoring. If spatial repair route status and byte-alphabet endpoints differ mostly on `affected` rows, especially QA or turn-taking, the difference is consistent with the tokenizer-surface mechanism quantified in dual compliant tokenizer endpoint policy/43. If they differ mainly on unaffected rows or other columns, the movement reflects broader training/tokenization interactions rather than newline `<unk>` exposure alone.

Full JSON: `experiments/archive/frontier_consolidation/data/compliant_full_eval/posthoc_supplement_slices/complianttok_reinvest_seed43022_supplement_prediction_slices.json`
