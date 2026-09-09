# earlier analysis selected pair interval: nodis_compact minus nodis_repeat

Created UTC: `2026-09-02T15:15:04Z`

CPU/file-only reading of existing selected predictions; no model scoring, training, SuperGLUE, AoA, upload, or leaderboard submission.

## Official/wrapper selected deltas

- cheap7: 0.8679
- cheap6_no_GlobalPIQA: 0.3533
- cheap5_no_GlobalPIQA_Reading: 0.4220
- EWoK_plus_Entity: 1.3000
- BLiMP: 0.8400
- Supplement: 0.0700
- EWoK: 0.3900
- Entity: 0.9100
- COMPS: -0.1000
- GlobalPIQA: 3.9550
- Reading: 0.0100

## Raw item paired intervals, RIGHT minus LEFT (percentage points)

| subset | n items | item point | item 95% | clusters | cluster point | cluster 95% |
|---|---:|---:|---:|---:|---:|---:|
| stable_five_item_pool | 170519 | 0.3613 | [0.1433, 0.6098] | 105 | 0.3613 | [-0.2574, 1.1707] |
| EWoK_plus_Entity_item_pool | 14398 | 0.6807 | [-0.2401, 1.4702] | 29 | 0.6807 | [0.0128, 1.4526] |
| BLiMP | 59875 | 0.8685 | [0.5702, 1.1925] | 67 | 0.8685 | [-0.3381, 2.3581] |
| Supplement | 5218 | -1.4182 | [-2.6830, -0.1231] | 5 | -1.4182 | [-2.8266, 0.9562] |
| EWoK | 7618 | 0.6170 | [-0.7614, 1.8184] | 11 | 0.6170 | [-0.4614, 1.9982] |
| Entity | 6780 | 0.7522 | [-0.2290, 1.5782] | 18 | 0.7522 | [-0.1336, 1.7046] |
| COMPS | 91028 | 0.0791 | [-0.2625, 0.4430] | 4 | 0.0791 | [-1.0111, 0.5433] |
| GlobalPIQA_item_pool | 203 | 3.9409 | [-1.4778, 9.6182] | 2 | 3.9409 | [2.9126, 5.0000] |

## Coverage and caution
- common items: `170722`, left-only `0`, right-only `0`.
- loader warnings: `0`; skipped columns: {'left': {'Reading': 'continuous reading correlations, not item correctness flips'}, 'right': {'Reading': 'continuous reading correlations, not item correctness flips'}}.
- Use these intervals to interpret item movement; the selected panel's official/wrapper scores remain the primary readout because BabyLM aggregation is not always raw item averaging.

## Files
- json: `experiments/archive/frontier_consolidation/data/commoncopy_architecture_interaction_pair_intervals/nodis_compact_minus_repeat/chck_100M/selected_pair_interval_analysis.json`
- md: `research/documents/frontier_consolidation/data/commoncopy_architecture_interaction_pair_intervals/nodis_compact_minus_repeat/chck_100M/selected_pair_interval_analysis.md`
- csv: `experiments/archive/frontier_consolidation/data/commoncopy_architecture_interaction_pair_intervals/nodis_compact_minus_repeat/chck_100M/subset_intervals.csv`
