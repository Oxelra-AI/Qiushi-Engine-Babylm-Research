# earlier analysis register stream audit

File-only stream audit; no model loading, training, evaluation, GPU, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard.

Tokenizer SHA match: True (`91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`).
Clean rows/words match expected: True / True.

## Arm checks

| arm | rows10 | words10 | changed rows | changed words | unchanged mismatches | length mismatches | replacement-index mismatches | fineweb multiset hash ok | mean token diff FineWeb-clean | FineWeb fertility | clean fertility |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| childsub_posmatched | 65313 | 10000000 | 758 | 105962 | 0 | 0 | 0 | False | -18.2480 | 1.4426 | 1.5729 |
| adult_posmatched | 65313 | 10000000 | 758 | 105962 | 0 | 0 | 0 | False | -1.6860 | 1.4426 | 1.4541 |

## Pair checks

- FineWeb text multiset identical by text hash: True.
- Changed row count child/adult: 758 / 758.
- Same changed positions across arms: 23; union changed positions: 1493.
- All rows word-count sequence identical between arms: True; identical to clean: True.
- Rows outside union of changed positions equal clean in both arms: True.

## 100M repetition

| arm | rows | words | ordered 10x repetition | mismatches |
|---|---:|---:|---|---:|
| childsub_posmatched | 653130 | 100000000 | True | 0 |
| adult_posmatched | 653130 | 100000000 | True | 0 |

## Interpretation

The actual stream files support the intended pair if all checks above are true: the contrast varies the directly removed clean register while keeping the admitted FineWeb multiset and word-count/row-count geometry fixed. Token fertility differences between admitted FineWeb and replaced clean rows remain part of the substitution mechanism and should be reported with score results.

Replacement map: `experiments/archive/frontier_consolidation/data/register_stream_audit/replacement_map_rows.csv`
JSON: `experiments/archive/frontier_consolidation/data/register_stream_audit/register_stream_audit_summary.json`
