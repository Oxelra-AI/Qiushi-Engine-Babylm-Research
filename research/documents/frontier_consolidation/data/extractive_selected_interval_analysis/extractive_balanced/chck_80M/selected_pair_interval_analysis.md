# earlier analysis selected pair interval: extractive_balanced minus legal_compact

Created UTC: `2026-09-02T07:57:44Z`

CPU/file-only reading of existing selected predictions; no model scoring, training, SuperGLUE, AoA, upload, or leaderboard submission.

## Official/wrapper selected deltas

- cheap7: 0.2600
- cheap6_no_GlobalPIQA: -0.1983
- cheap5_no_GlobalPIQA_Reading: -0.2340
- EWoK_plus_Entity: -0.9400
- BLiMP: 0.0300
- Supplement: 0.4800
- EWoK: -0.3900
- Entity: -0.5500
- COMPS: -0.7400
- GlobalPIQA: 3.0100
- Reading: -0.0200

## Raw item paired intervals, RIGHT minus LEFT (percentage points)

| subset | n items | item point | item 95% | clusters | cluster point | cluster 95% |
|---|---:|---:|---:|---:|---:|---:|
| stable_five_item_pool | 170519 | -0.4434 | [-0.7059, -0.2037] | 105 | -0.4434 | [-2.0746, 0.9627] |
| EWoK_plus_Entity_item_pool | 14398 | -0.4931 | [-1.2786, 0.2679] | 29 | -0.4931 | [-1.1554, 0.4231] |
| BLiMP | 59875 | 0.1052 | [-0.2298, 0.3992] | 67 | 0.1052 | [-1.2037, 1.3643] |
| Supplement | 5218 | -5.3852 | [-6.3535, -4.1477] | 5 | -5.3852 | [-6.5111, 4.3956] |
| EWoK | 7618 | -0.9057 | [-2.1728, 0.3357] | 11 | -0.9057 | [-1.8411, 0.1692] |
| Entity | 6780 | -0.0295 | [-0.8927, 0.9152] | 18 | -0.0295 | [-1.2801, 1.2027] |
| COMPS | 91028 | -0.5130 | [-0.9196, -0.0790] | 4 | -0.5130 | [-4.1829, 2.1553] |
| GlobalPIQA_item_pool | 203 | 2.9557 | [-2.4631, 8.6330] | 2 | 2.9557 | [-0.9709, 7.0000] |

## Coverage and caution
- common items: `170722`, left-only `0`, right-only `0`.
- loader warnings: `0`; skipped columns: {'left': {'Reading': 'continuous reading correlations, not item correctness flips'}, 'right': {'Reading': 'continuous reading correlations, not item correctness flips'}}.
- Use these intervals to interpret item movement; the selected panel's official/wrapper scores remain the primary readout because BabyLM aggregation is not always raw item averaging.

## Files
- json: `experiments/archive/frontier_consolidation/data/extractive_selected_interval_analysis/extractive_balanced/chck_80M/selected_pair_interval_analysis.json`
- md: `research/documents/frontier_consolidation/data/extractive_selected_interval_analysis/extractive_balanced/chck_80M/selected_pair_interval_analysis.md`
- csv: `experiments/archive/frontier_consolidation/data/extractive_selected_interval_analysis/extractive_balanced/chck_80M/subset_intervals.csv`
