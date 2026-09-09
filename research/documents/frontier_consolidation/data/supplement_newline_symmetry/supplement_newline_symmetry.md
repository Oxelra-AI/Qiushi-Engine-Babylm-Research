# supplement newline symmetry and posthoc slices Supplement newline symmetry under the spatial repair route status tokenizer

This CPU-only analysis asks whether the spatial repair route status same-pool tokenizer's Supplement `<unk>` targets are matched between the good and bad alternatives in the official MLM scoring path. It does not run model inference and does not alter either compliant tokenizer.

spatial repair route status tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`
Byte-alphabet tokenizer SHA: `b2b317e655a96f2c14eb559ce9fa7573f180904d84819cacfb5f61174cf355cf`

## Aggregate Supplement surface

Rows scanned: 5218; rows with spatial repair route status target `<unk>`: 473 (0.090648).
spatial repair route status target `<unk>` tokens: 946 out of 172736 paired candidate target tokens (0.005477).
Byte-alphabet / spatial repair route status target-token ratio: 1.000973.
Affected rows all have both candidates affected: 473/473.
Affected rows with same `<unk>` count in good and bad: 473/473.
Affected rows with same `<unk>` span multiset: 473/473.
Affected rows with same span+offset: 369/473.
Affected rows where every spatial repair route status `<unk>` span starts with newline: 473/473.
Affected rows with identical 8-character local windows around the spatial repair route status `<unk>` target: 201/473.
Affected rows with identical 24-character local windows around the spatial repair route status `<unk>` target: 64/473.

If every affected Supplement row flipped solely because of this tokenizer surface, the maximum exposed Supplement movement would be 57.429 column points, or 6.381 Overall points. This is only a structural bound; symmetry means the realized effect could be much smaller and must be measured by official scoring.

## By Supplement subtask

| subtask | rows | affected rows | affected rate | target `<unk>` frac | same count | same span | same span+offset | local8 same | top spatial repair route status span | max Supplement pts | Overall per row flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| hypernym | 842 | 0 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 | `` | 0.000 | 0.002639 |
| qa_congruence_easy | 64 | 64 | 1.000000 | 0.076831 | 64 | 64 | 64 | 26 | `'\n'` | 20.000 | 0.034722 |
| qa_congruence_tricky | 165 | 165 | 1.000000 | 0.083460 | 165 | 165 | 165 | 10 | `'\n'` | 20.000 | 0.013468 |
| subject_aux_inversion | 3867 | 0 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 | `` | 0.000 | 0.000575 |
| turn_taking | 280 | 244 | 0.871429 | 0.048557 | 244 | 244 | 140 | 165 | `'\n'` | 17.429 | 0.007937 |

## Interpretation for endpoint comparison

The spatial repair route status tokenizer weakness is real but highly structured. In all affected rows the good and bad candidates both contain a spatial repair route status target `<unk>`; in the affected QA rows the unknown span and offset are identical, while turn-taking contains the same newline-starting target in both candidates but local context can differ because the contrast changes speaker/pronoun material before or after the line break. Thus the newline/speaker `<unk>` is not a simple one-sided poison token, and its model-score effect cannot be inferred from tokenization alone.

For later pristine results, a spatial repair route status-versus-byte-alphabet difference concentrated in `qa_congruence_easy`, `qa_congruence_tricky`, or `turn_taking` should be read as tokenizer-surface interaction. A difference in hypernym, subject-aux inversion, BLiMP, EWoK, Entity, COMPS, or GlobalPIQA cannot be explained by the spatial repair route status newline `<unk>` exposure shown here.

Full JSON: `experiments/archive/frontier_consolidation/data/supplement_newline_symmetry/supplement_newline_symmetry.json`
