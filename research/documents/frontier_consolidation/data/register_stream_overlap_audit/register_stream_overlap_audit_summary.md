# earlier analysis register stream overlap audit

File-only overlap/effective-contrast audit; no model loading, training, evaluation, GPU, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard.

## Effective row contrast

- Child/subtitle-removal arm changed rows: 758.
- Adult-prose-removal arm changed rows: 758.
- Positions changed in both arms: 23; positions changed in only child arm: 735; only adult arm: 735; union: 1493.
- Actual cross-arm text-different rows: 1481.
- FineWeb text multiset identical across arms: True.
- Overlapping replacement positions with same FineWeb text: 12; with different FineWeb text: 11.
- Unexpected child/adult text differences outside replacement markers: 0 / 0.

The metadata position-match field `same_clean_row_both_arms_count=3` counts matched selection pairs that used the same clean row; it is not the same as the actual number of output positions that are replacement rows in both arms, which is 23. The score contrast remains mostly direct register substitution because 735/758 replacement positions per arm are not shared, all rows keep the same word-count sequence, and rows outside the changed union are identical clean text.

## Subset source mixtures (clean words at positions)

| subset | rows | words | row mean | source groups |
|---|---:|---:|---:|---|
| child_replaced_only | 735 | 103044 | 4019.8639455782313 | `{"adult_gut_simple": 661, "child_sub": 100871, "other": 1512}` |
| adult_replaced_only | 735 | 103044 | 4035.5442176870747 | `{"adult_gut_simple": 88299, "child_sub": 6837, "other": 7908}` |
| replaced_in_both_arms | 23 | 2918 | 2618.304347826087 | `{"adult_gut_simple": 589, "child_sub": 1922, "other": 407}` |
| changed_union | 1493 | 209006 | 4005.9919624916274 | `{"adult_gut_simple": 89549, "child_sub": 109630, "other": 9827}` |
| unchanged_in_both_arms | 63820 | 9790994 | 33326.236007521155 | `{"adult_gut_simple": 2986802, "child_sub": 4529344, "other": 2274848}` |
| cross_arm_text_diff_positions | 1481 | 207536 | 4025.8541525995947 | `{"adult_gut_simple": 89198, "child_sub": 108777, "other": 9561}` |

## Reading consequence

The direct pair is a valid high-contrast test of removal-side opportunity cost, but not a pure abstract register axis. Interpret pair deltas as conditional on these selected mixtures and on the small amount of shared replacement-position overlap.

JSON: `experiments/archive/frontier_consolidation/data/register_stream_overlap_audit/register_stream_overlap_audit_summary.json`
Subset CSV: `experiments/archive/frontier_consolidation/data/register_stream_overlap_audit/overlap_subset_summaries.csv`
Examples CSV: `experiments/archive/frontier_consolidation/data/register_stream_overlap_audit/overlap_examples.csv`
