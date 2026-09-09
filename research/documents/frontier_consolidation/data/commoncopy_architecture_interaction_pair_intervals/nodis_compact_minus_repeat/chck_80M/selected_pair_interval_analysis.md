# earlier analysis selected pair interval: nodis_compact minus nodis_repeat

Created UTC: `2026-09-02T15:14:43Z`

CPU/file-only reading of existing selected predictions; no model scoring, training, SuperGLUE, AoA, upload, or leaderboard submission.

## Official/wrapper selected deltas

- cheap7: 0.7064
- cheap6_no_GlobalPIQA: 0.3267
- cheap5_no_GlobalPIQA_Reading: 0.3760
- EWoK_plus_Entity: 0.6100
- BLiMP: 0.4300
- Supplement: 0.5200
- EWoK: 0.1200
- Entity: 0.4900
- COMPS: 0.3200
- GlobalPIQA: 2.9850
- Reading: 0.0800

## Raw item paired intervals, RIGHT minus LEFT (percentage points)

| subset | n items | item point | item 95% | clusters | cluster point | cluster 95% |
|---|---:|---:|---:|---:|---:|---:|
| stable_five_item_pool | 170519 | 0.3806 | [0.1737, 0.6217] | 105 | 0.3806 | [-0.2826, 0.9916] |
| EWoK_plus_Entity_item_pool | 14398 | 0.5765 | [-0.1912, 1.3346] | 29 | 0.5765 | [-0.0718, 1.3754] |
| BLiMP | 59875 | 0.4526 | [0.1268, 0.8063] | 67 | 0.4526 | [-1.3037, 1.9475] |
| Supplement | 5218 | -0.8049 | [-1.8791, 0.2875] | 5 | -0.8049 | [-1.2950, 0.9535] |
| EWoK | 7618 | 0.3150 | [-1.0121, 1.4902] | 11 | 0.3150 | [-0.6721, 1.4591] |
| Entity | 6780 | 0.8702 | [0.0435, 1.8956] | 18 | 0.8702 | [-0.0731, 1.8643] |
| COMPS | 91028 | 0.3702 | [-0.0141, 0.7724] | 4 | 0.3702 | [-0.3598, 0.7340] |
| GlobalPIQA_item_pool | 203 | 2.9557 | [-2.2291, 9.3596] | 2 | 2.9557 | [0.9709, 5.0000] |

## Coverage and caution
- common items: `170722`, left-only `0`, right-only `0`.
- loader warnings: `0`; skipped columns: {'left': {'Reading': 'continuous reading correlations, not item correctness flips'}, 'right': {'Reading': 'continuous reading correlations, not item correctness flips'}}.
- Use these intervals to interpret item movement; the selected panel's official/wrapper scores remain the primary readout because BabyLM aggregation is not always raw item averaging.

## Files
- json: `experiments/archive/frontier_consolidation/data/commoncopy_architecture_interaction_pair_intervals/nodis_compact_minus_repeat/chck_80M/selected_pair_interval_analysis.json`
- md: `research/documents/frontier_consolidation/data/commoncopy_architecture_interaction_pair_intervals/nodis_compact_minus_repeat/chck_80M/selected_pair_interval_analysis.md`
- csv: `experiments/archive/frontier_consolidation/data/commoncopy_architecture_interaction_pair_intervals/nodis_compact_minus_repeat/chck_80M/subset_intervals.csv`
