# supplement newline symmetry and posthoc slices Supplement prediction slices — oldtok_compact_view_reinvest_seed43022

This CPU-only join maps official Supplement predictions to the spatial repair route status-tokenizer newline-affected row structure. It is an interpretation layer; official column scores still come from the standard evaluator and pristine collation.

Predictions: `experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/official_outputs/compact_view_reinvest/Supplement/chck_100M/full_compact_view_reinvest_Supplement/zero_shot/mlm/blimp/supplement_filtered/predictions.json`
Rows joined: 5218 / 5218 (missing 0, extra 0).
Subtask-macro Supplement accuracy: 63.275764

## By subtask

| subtask | n | correct | accuracy | affected n | affected accuracy | unaffected accuracy |
|---|---:|---:|---:|---:|---:|---:|
| hypernym | 842 | 425 | 50.475 | 0 |  | 50.475 |
| qa_congruence_easy | 64 | 46 | 71.875 | 64 | 71.875 |  |
| qa_congruence_tricky | 165 | 76 | 46.061 | 165 | 46.061 |  |
| subject_aux_inversion | 3867 | 3236 | 83.682 | 0 |  | 83.682 |
| turn_taking | 280 | 180 | 64.286 | 244 | 62.705 | 75.000 |

## Affected versus unaffected

| slice | n | correct | accuracy |
|---|---:|---:|---:|
| affected | 473 | 275 | 58.140 |
| unaffected | 4745 | 3688 | 77.724 |

## Structural slices among spatial repair route status-affected rows

| structural slice | n | correct | accuracy |
|---|---:|---:|---:|
| affected_offsetdiff_local8diff | 9 | 5 | 55.556 |
| affected_offsetdiff_local8same | 95 | 58 | 61.053 |
| affected_same_span_offset | 369 | 212 | 57.453 |
| unaffected | 4745 | 3688 | 77.724 |

## Interpretation

Use this after each compliant endpoint finishes official Supplement scoring. If spatial repair route status and byte-alphabet endpoints differ mostly on `affected` rows, especially QA or turn-taking, the difference is consistent with the tokenizer-surface mechanism quantified in dual compliant tokenizer endpoint policy/43. If they differ mainly on unaffected rows or other columns, the movement reflects broader training/tokenization interactions rather than newline `<unk>` exposure alone.

Full JSON: `experiments/archive/frontier_consolidation/data/supplement_prediction_slices/oldtok_compact_view_reinvest_seed43022_supplement_prediction_slices.json`
