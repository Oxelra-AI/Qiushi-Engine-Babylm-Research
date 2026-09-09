# earlier analysis selected pair interval: nodis_compact minus nodis_repeat

Created UTC: `2026-09-02T13:54:24Z`

CPU/file-only reading of existing selected predictions; no model scoring, training, SuperGLUE, AoA, upload, or leaderboard submission.

## Official/wrapper selected deltas

- cheap7: -0.6793
- cheap6_no_GlobalPIQA: -0.7067
- cheap5_no_GlobalPIQA_Reading: -0.6620
- EWoK_plus_Entity: -2.5000
- BLiMP: -0.6500
- Supplement: -0.5100
- EWoK: -2.6300
- Entity: 0.1300
- COMPS: 0.3500
- GlobalPIQA: -0.5150
- Reading: -0.9300

## Raw item paired intervals, RIGHT minus LEFT (percentage points)

| subset | n items | item point | item 95% | clusters | cluster point | cluster 95% |
|---|---:|---:|---:|---:|---:|---:|
| stable_five_item_pool | 170519 | -0.0575 | [-0.3337, 0.1884] | 105 | -0.0575 | [-1.2450, 1.0380] |
| EWoK_plus_Entity_item_pool | 14398 | -0.9793 | [-1.7370, -0.0761] | 29 | -0.9793 | [-2.2422, 0.1922] |
| BLiMP | 59875 | -0.6965 | [-1.0063, -0.3063] | 67 | -0.6965 | [-3.1655, 1.6184] |
| Supplement | 5218 | 3.2388 | [2.0405, 4.4754] | 5 | 3.2388 | [-2.8333, 4.8253] |
| EWoK | 7618 | -1.9690 | [-3.3342, -0.5044] | 11 | -1.9690 | [-4.7346, -0.4191] |
| Entity | 6780 | 0.1327 | [-0.8717, 1.0767] | 18 | 0.1327 | [-0.7045, 0.8464] |
| COMPS | 91028 | 0.3197 | [-0.0508, 0.6731] | 4 | 0.3197 | [-1.1532, 1.7523] |
| GlobalPIQA_item_pool | 203 | -0.4926 | [-5.4187, 5.4187] | 2 | -0.4926 | [-2.0000, 0.9709] |

## Coverage and caution
- common items: `170722`, left-only `0`, right-only `0`.
- loader warnings: `0`; skipped columns: {'left': {'Reading': 'continuous reading correlations, not item correctness flips'}, 'right': {'Reading': 'continuous reading correlations, not item correctness flips'}}.
- Use these intervals to interpret item movement; the selected panel's official/wrapper scores remain the primary readout because BabyLM aggregation is not always raw item averaging.

## Files
- json: `experiments/archive/frontier_consolidation/data/architecture_interaction_pair_intervals/nodis_compact_minus_repeat/chck_80M/selected_pair_interval_analysis.json`
- md: `research/documents/frontier_consolidation/data/architecture_interaction_pair_intervals/nodis_compact_minus_repeat/chck_80M/selected_pair_interval_analysis.md`
- csv: `experiments/archive/frontier_consolidation/data/architecture_interaction_pair_intervals/nodis_compact_minus_repeat/chck_80M/subset_intervals.csv`
