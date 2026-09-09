# earlier analysis selected pair interval: extractive_wide minus legal_compact

Created UTC: `2026-09-02T07:57:54Z`

CPU/file-only reading of existing selected predictions; no model scoring, training, SuperGLUE, AoA, upload, or leaderboard submission.

## Official/wrapper selected deltas

- cheap7: -0.1193
- cheap6_no_GlobalPIQA: -0.8075
- cheap5_no_GlobalPIQA_Reading: -0.8880
- EWoK_plus_Entity: -3.2700
- BLiMP: 0.9400
- Supplement: -2.1000
- EWoK: -0.9700
- Entity: -2.3000
- COMPS: -0.0100
- GlobalPIQA: 4.0100
- Reading: -0.4050

## Raw item paired intervals, RIGHT minus LEFT (percentage points)

| subset | n items | item point | item 95% | clusters | cluster point | cluster 95% |
|---|---:|---:|---:|---:|---:|---:|
| stable_five_item_pool | 170519 | 0.0158 | [-0.2349, 0.3322] | 105 | 0.0158 | [-0.9230, 1.2604] |
| EWoK_plus_Entity_item_pool | 14398 | -1.0001 | [-1.8101, -0.2252] | 29 | -1.0001 | [-2.3897, 0.3244] |
| BLiMP | 59875 | 0.9370 | [0.6075, 1.2965] | 67 | 0.9370 | [-0.4494, 2.2158] |
| Supplement | 5218 | -3.3921 | [-4.3906, -2.2897] | 5 | -3.3921 | [-3.7894, -1.4643] |
| EWoK | 7618 | -0.7351 | [-1.9103, 0.4608] | 11 | -0.7351 | [-3.1414, 1.0143] |
| Entity | 6780 | -1.2979 | [-2.2939, -0.3540] | 18 | -1.2979 | [-3.0211, 0.2850] |
| COMPS | 91028 | -0.2340 | [-0.5989, 0.1813] | 4 | -0.2340 | [-1.7731, 1.7377] |
| GlobalPIQA_item_pool | 203 | 3.9409 | [-1.7365, 9.8522] | 2 | 3.9409 | [-0.9709, 9.0000] |

## Coverage and caution
- common items: `170722`, left-only `0`, right-only `0`.
- loader warnings: `0`; skipped columns: {'left': {'Reading': 'continuous reading correlations, not item correctness flips'}, 'right': {'Reading': 'continuous reading correlations, not item correctness flips'}}.
- Use these intervals to interpret item movement; the selected panel's official/wrapper scores remain the primary readout because BabyLM aggregation is not always raw item averaging.

## Files
- json: `experiments/archive/frontier_consolidation/data/extractive_selected_interval_analysis/extractive_wide/chck_80M/selected_pair_interval_analysis.json`
- md: `research/documents/frontier_consolidation/data/extractive_selected_interval_analysis/extractive_wide/chck_80M/selected_pair_interval_analysis.md`
- csv: `experiments/archive/frontier_consolidation/data/extractive_selected_interval_analysis/extractive_wide/chck_80M/subset_intervals.csv`
