# earlier analysis selected pair interval: nodis_compact minus nodis_repeat

Created UTC: `2026-09-02T13:54:45Z`

CPU/file-only reading of existing selected predictions; no model scoring, training, SuperGLUE, AoA, upload, or leaderboard submission.

## Official/wrapper selected deltas

- cheap7: -0.0429
- cheap6_no_GlobalPIQA: -0.1308
- cheap5_no_GlobalPIQA_Reading: -0.0660
- EWoK_plus_Entity: -1.0600
- BLiMP: 0.0400
- Supplement: 0.0800
- EWoK: -0.9700
- Entity: -0.0900
- COMPS: 0.6100
- GlobalPIQA: 0.4850
- Reading: -0.4550

## Raw item paired intervals, RIGHT minus LEFT (percentage points)

| subset | n items | item point | item 95% | clusters | cluster point | cluster 95% |
|---|---:|---:|---:|---:|---:|---:|
| stable_five_item_pool | 170519 | 0.3003 | [0.0660, 0.5337] | 105 | 0.3003 | [-0.7938, 1.2135] |
| EWoK_plus_Entity_item_pool | 14398 | -0.6320 | [-1.4413, 0.2269] | 29 | -0.6320 | [-2.0088, 0.4115] |
| BLiMP | 59875 | 0.0251 | [-0.3141, 0.3441] | 67 | 0.0251 | [-2.1244, 2.1812] |
| Supplement | 5218 | 4.3503 | [2.9796, 5.7695] | 5 | 4.3503 | [-1.4675, 5.3029] |
| EWoK | 7618 | -1.2733 | [-2.5991, 0.3570] | 11 | -1.2733 | [-3.8399, 0.6973] |
| Entity | 6780 | 0.0885 | [-0.7895, 0.9664] | 18 | 0.0885 | [-0.6651, 0.7893] |
| COMPS | 91028 | 0.3966 | [0.0387, 0.7571] | 4 | 0.3966 | [-0.4680, 1.7307] |
| GlobalPIQA_item_pool | 203 | 0.4926 | [-3.9409, 4.6921] | 2 | 0.4926 | [0.0000, 0.9709] |

## Coverage and caution
- common items: `170722`, left-only `0`, right-only `0`.
- loader warnings: `0`; skipped columns: {'left': {'Reading': 'continuous reading correlations, not item correctness flips'}, 'right': {'Reading': 'continuous reading correlations, not item correctness flips'}}.
- Use these intervals to interpret item movement; the selected panel's official/wrapper scores remain the primary readout because BabyLM aggregation is not always raw item averaging.

## Files
- json: `experiments/archive/frontier_consolidation/data/architecture_interaction_pair_intervals/nodis_compact_minus_repeat/chck_100M/selected_pair_interval_analysis.json`
- md: `research/documents/frontier_consolidation/data/architecture_interaction_pair_intervals/nodis_compact_minus_repeat/chck_100M/selected_pair_interval_analysis.md`
- csv: `experiments/archive/frontier_consolidation/data/architecture_interaction_pair_intervals/nodis_compact_minus_repeat/chck_100M/subset_intervals.csv`
