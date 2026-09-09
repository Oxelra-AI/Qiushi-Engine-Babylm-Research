# earlier analysis selected pair interval: full_compact minus full_repeat

Created UTC: `2026-09-02T13:54:35Z`

CPU/file-only reading of existing selected predictions; no model scoring, training, SuperGLUE, AoA, upload, or leaderboard submission.

## Official/wrapper selected deltas

- cheap7: -0.3900
- cheap6_no_GlobalPIQA: 0.1283
- cheap5_no_GlobalPIQA_Reading: 0.0680
- EWoK_plus_Entity: 0.7700
- BLiMP: -1.1200
- Supplement: 0.2100
- EWoK: -0.8600
- Entity: 1.6300
- COMPS: 0.4800
- GlobalPIQA: -3.5000
- Reading: 0.4300

## Raw item paired intervals, RIGHT minus LEFT (percentage points)

| subset | n items | item point | item 95% | clusters | cluster point | cluster 95% |
|---|---:|---:|---:|---:|---:|---:|
| stable_five_item_pool | 170519 | -0.1859 | [-0.4377, 0.0542] | 105 | -0.1859 | [-1.4332, 1.2522] |
| EWoK_plus_Entity_item_pool | 14398 | 0.7084 | [-0.1816, 1.5078] | 29 | 0.7084 | [-0.8306, 2.0942] |
| BLiMP | 59875 | -1.1441 | [-1.5510, -0.8405] | 67 | -1.1441 | [-2.4258, 0.1739] |
| Supplement | 5218 | 1.0349 | [0.0455, 2.0506] | 5 | 1.0349 | [-0.4357, 1.4385] |
| EWoK | 7618 | 0.4463 | [-0.8470, 1.6609] | 11 | 0.4463 | [-1.9513, 2.3772] |
| Entity | 6780 | 1.0029 | [0.1327, 2.0656] | 18 | 1.0029 | [-1.0765, 3.1105] |
| COMPS | 91028 | 0.2329 | [-0.1434, 0.6324] | 4 | 0.2329 | [-1.9829, 2.9343] |
| GlobalPIQA_item_pool | 203 | -3.4483 | [-8.8670, 1.9704] | 2 | -3.4483 | [-7.0000, 0.0000] |

## Coverage and caution
- common items: `170722`, left-only `0`, right-only `0`.
- loader warnings: `0`; skipped columns: {'left': {'Reading': 'continuous reading correlations, not item correctness flips'}, 'right': {'Reading': 'continuous reading correlations, not item correctness flips', 'SuperGLUE': 'unsupported_or_nonselected_column', 'AoA': 'unsupported_or_nonselected_column'}}.
- Use these intervals to interpret item movement; the selected panel's official/wrapper scores remain the primary readout because BabyLM aggregation is not always raw item averaging.

## Files
- json: `experiments/archive/frontier_consolidation/data/architecture_interaction_pair_intervals/full_compact_minus_repeat/chck_100M/selected_pair_interval_analysis.json`
- md: `research/documents/frontier_consolidation/data/architecture_interaction_pair_intervals/full_compact_minus_repeat/chck_100M/selected_pair_interval_analysis.md`
- csv: `experiments/archive/frontier_consolidation/data/architecture_interaction_pair_intervals/full_compact_minus_repeat/chck_100M/subset_intervals.csv`
