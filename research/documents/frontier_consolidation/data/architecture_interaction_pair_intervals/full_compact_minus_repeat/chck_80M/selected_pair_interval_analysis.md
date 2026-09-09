# earlier analysis selected pair interval: full_compact minus full_repeat

Created UTC: `2026-09-02T13:54:14Z`

CPU/file-only reading of existing selected predictions; no model scoring, training, SuperGLUE, AoA, upload, or leaderboard submission.

## Official/wrapper selected deltas

- cheap7: -0.1414
- cheap6_no_GlobalPIQA: 0.4158
- cheap5_no_GlobalPIQA_Reading: 0.4200
- EWoK_plus_Entity: 1.9100
- BLiMP: -0.4700
- Supplement: 0.3200
- EWoK: 1.2700
- Entity: 0.6400
- COMPS: 0.3400
- GlobalPIQA: -3.4850
- Reading: 0.3950

## Raw item paired intervals, RIGHT minus LEFT (percentage points)

| subset | n items | item point | item 95% | clusters | cluster point | cluster 95% |
|---|---:|---:|---:|---:|---:|---:|
| stable_five_item_pool | 170519 | 0.0088 | [-0.2138, 0.2809] | 105 | 0.0088 | [-1.4812, 1.3050] |
| EWoK_plus_Entity_item_pool | 14398 | 0.7848 | [0.0066, 1.6398] | 29 | 0.7848 | [-0.5279, 2.0459] |
| BLiMP | 59875 | -0.5344 | [-0.8595, -0.1845] | 67 | -0.5344 | [-1.9073, 0.8054] |
| Supplement | 5218 | -0.4599 | [-1.4958, 0.4791] | 5 | -0.4599 | [-0.8106, 1.8827] |
| EWoK | 7618 | 1.3521 | [0.0981, 2.5991] | 11 | 1.3521 | [-0.7680, 3.0575] |
| Entity | 6780 | 0.1475 | [-0.7010, 1.0767] | 18 | 0.1475 | [-1.3991, 1.9244] |
| COMPS | 91028 | 0.2702 | [-0.1447, 0.7171] | 4 | 0.2702 | [-2.2992, 3.2509] |
| GlobalPIQA_item_pool | 203 | -3.4483 | [-9.3596, 0.9852] | 2 | -3.4483 | [-6.0000, -0.9709] |

## Coverage and caution
- common items: `170722`, left-only `0`, right-only `0`.
- loader warnings: `0`; skipped columns: {'left': {'Reading': 'continuous reading correlations, not item correctness flips'}, 'right': {'Reading': 'continuous reading correlations, not item correctness flips'}}.
- Use these intervals to interpret item movement; the selected panel's official/wrapper scores remain the primary readout because BabyLM aggregation is not always raw item averaging.

## Files
- json: `experiments/archive/frontier_consolidation/data/architecture_interaction_pair_intervals/full_compact_minus_repeat/chck_80M/selected_pair_interval_analysis.json`
- md: `research/documents/frontier_consolidation/data/architecture_interaction_pair_intervals/full_compact_minus_repeat/chck_80M/selected_pair_interval_analysis.md`
- csv: `experiments/archive/frontier_consolidation/data/architecture_interaction_pair_intervals/full_compact_minus_repeat/chck_80M/subset_intervals.csv`
