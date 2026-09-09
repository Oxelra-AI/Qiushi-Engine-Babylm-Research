# supplement newline symmetry and posthoc slices — Supplement newline symmetry and posthoc slice tooling

## Scientific Motivation

The two compliant reinvest training comparisons remain pending:

- spatial repair route status same-pool compliant tokenizer endpoint, target `complianttok_reinvest_seed43022`.
- bytealphabet tokenizer repair and retrain priority byte-alphabet tokenizer endpoint, target `bytealphatok_reinvest_seed43022`.

Active training outputs were not inspected. This analysis characterizes the original legal tokenizer's newline/`<unk>` issue before official endpoint scores become available.

dual compliant tokenizer endpoint policy had shown that the spatial repair route status tokenizer is legal but creates direct target-span `<unk>` exposure in 473/5,218 official Supplement rows, concentrated in `qa_congruence_easy`, `qa_congruence_tricky`, and `turn_taking`. What remained unclear was whether this exposure was one-sided enough to explain future spatial repair route status-versus-byte-alphabet score movement, or whether it is mostly a shared structural nuisance in both good and bad alternatives.

## New CPU-only evidence

### 1. Supplement newline symmetry

Script:
- `experiments/archive/frontier_consolidation/scripts/supplement_newline_symmetry.py`

Outputs:
- `experiments/archive/frontier_consolidation/data/supplement_newline_symmetry/supplement_newline_symmetry.json`
- `research/documents/frontier_consolidation/data/supplement_newline_symmetry/supplement_newline_symmetry.md`

Key results:

- Official Supplement rows scanned: 5,218.
- Rows with spatial repair route status target `<unk>`: 473 (9.0648%).
- spatial repair route status target `<unk>` tokens: 946 / 172,736 paired candidate target tokens (0.5477%).
- Every affected row has both good and bad alternatives affected: 473/473.
- Every affected row has the same `<unk>` count in good and bad: 473/473.
- Every affected row has the same `<unk>` span multiset in good and bad: 473/473.
- Same span+offset holds for 369/473 rows.
- Every affected `<unk>` span starts with newline: 473/473.
- Identical 8-character local windows around the spatial repair route status `<unk>` target occur in 201/473 rows; identical 24-character windows in 64/473.

Subtask localization:

| subtask | rows | affected rows | same span+offset | local8 same | max column movement if all affected flip |
|---|---:|---:|---:|---:|---:|
| `qa_congruence_easy` | 64 | 64 | 64 | 26 | 20.000 |
| `qa_congruence_tricky` | 165 | 165 | 165 | 10 | 20.000 |
| `turn_taking` | 280 | 244 | 140 | 165 | 17.429 |
| `hypernym` | 842 | 0 | 0 | 0 | 0.000 |
| `subject_aux_inversion` | 3867 | 0 | 0 | 0 | 0.000 |

Interpretation:

The spatial repair route status tokenizer weakness is real and can affect the Supplement column, but it is not a simple one-sided poison token. Good and bad alternatives always share the newline-starting `<unk>` target count and span string. QA rows have identical unknown span and offset; turn-taking has more local context asymmetry because the contrast changes speaker/pronoun material around the line break. Therefore tokenization alone cannot predict the sign of spatial repair route status-versus-byte-alphabet Supplement movement. Official scoring must decide, and future interpretation must localize row subsets.

The structural upper bound is large only in a formal sense: if every affected row flipped, the exposed Supplement movement could be 57.429 column points, equal to 6.381 Overall points. Because the exposure is symmetric, realized movement may be far smaller.

### 2. Supplement prediction-slice tool

Script:
- `experiments/archive/frontier_consolidation/scripts/supplement_prediction_slices.py`

Purpose:
- Join any future official Supplement `predictions.json` to the supplement newline symmetry and posthoc slices affected-row map.
- Compute official subtask-macro Supplement accuracy plus affected/unaffected and structural-slice accuracy.
- Preserve examples of affected-row errors for interpretation.

Validation on existing old-tokenizer compact-view-reinvest seed43022 predictions:

Output:
- `experiments/archive/frontier_consolidation/data/supplement_prediction_slices/oldtok_compact_view_reinvest_seed43022_supplement_prediction_slices.json`
- `.md`

Result:
- Joined 5,218 / 5,218 rows.
- Reproduced known Supplement macro exactly: 63.27576417952157.
- Affected-row accuracy 58.140%; unaffected-row accuracy 77.724%.

Validation on inherited old-tokenizer clean-Qwen seed43022 predictions:

Output:
- `experiments/archive/frontier_consolidation/data/supplement_prediction_slices/oldtok_qwen_clean_aligned_seed43022_supplement_prediction_slices.json`
- `.md`

Result:
- Joined 5,218 / 5,218 rows.
- Reproduced known Supplement macro: 62.84254425840123.
- Affected-row accuracy 59.619%; unaffected-row accuracy 75.111%.

### 3. Existing old-tokenizer reinvest-versus-clean slice comparison

Script:
- `experiments/archive/frontier_consolidation/scripts/compare_supplement_slices.py`

Output:
- `experiments/archive/frontier_consolidation/data/supplement_prediction_slices/existing_oldtok_reinvest_vs_clean_comparison.json`
- `.md`

Existing old-tokenizer comparison:

| comparison | Supplement macro delta | affected delta | unaffected delta | QA easy affected | QA tricky affected | turn-taking affected | hypernym | subject-aux |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact-view-reinvest - clean-Qwen | +0.433 pp | -1.480 pp | +2.613 pp | +1.562 pp | -3.030 pp | -1.230 pp | +1.069 pp | +2.922 pp |

This means the old-tokenizer compact-view-reinvest Supplement gain over clean-Qwen was **not** carried by the newline-affected rows; it came from unaffected Supplement rows, especially hypernym/subject-aux and generally non-newline surface. This is important because it prevents a future shortcut interpretation that any compliant-tokenizer Supplement shift must be due to newline `<unk>` exposure alone.

## Post-delivery driver update

Patched:
- `experiments/archive/frontier_consolidation/scripts/compliant_postdelivery_driver.py`

Changes:

- Added `SUPPLEMENT_SLICER = scripts/supplement_prediction_slices.py`.
- Added reusable `latest_prediction_for_column(...)` while retaining `latest_prediction_for_ewok(...)` for pristine collation.
- After cheap columns finish, the driver now runs a CPU-only `supplement_prediction_slices` stage using the actual saved Supplement predictions for that target.
- Dry-runs passed for both endpoint identities:
  - `complianttok_reinvest_seed43022` with spatial repair route status tokenizer SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`.
  - `bytealphatok_reinvest_seed43022` with byte-alphabet tokenizer SHA `b2b317e655a96f2c14eb559ce9fa7573f180904d84819cacfb5f61174cf355cf`.
- The dry-run JSONs show isolated target, per-target, posthoc Supplement-slice, AoA, and pristine-collate paths:
  - `experiments/archive/frontier_consolidation/data/compliant_postdelivery_driver/complianttok_reinvest_seed43022_postdelivery_driver.json`
  - `experiments/archive/frontier_consolidation/data/compliant_postdelivery_driver/bytealphatok_reinvest_seed43022_postdelivery_driver.json`

This patch does not change official scoring. It only ensures future official Supplement predictions are immediately interpretable by the newline-affected structure.

## Consequence for the research route

The endpoint policy from dual compliant tokenizer endpoint policy remains intact:

1. The byte-alphabet endpoint has resource priority because it is the standard ByteLevel BPE construction and removes avoidable `<unk>` coverage on official strings.
2. The spatial repair route status tokenizer endpoint remains legal and should be evaluated if it completes without delaying the byte-alphabet priority path.
3. Endpoint choice must be made by pristine official nine-column collation, not by tokenization speculation.
4. If spatial repair route status and byte-alphabet endpoints differ mostly in `qa_congruence_easy`, `qa_congruence_tricky`, or `turn_taking` affected rows, the difference is consistent with the newline-tokenizer-surface mechanism. If they differ mainly in unaffected Supplement rows or other leaderboard columns, the movement reflects broader legal-tokenizer training dynamics rather than the spatial repair route status newline `<unk>` issue alone.

No model score was produced or inferred in supplement newline symmetry and posthoc slices.
