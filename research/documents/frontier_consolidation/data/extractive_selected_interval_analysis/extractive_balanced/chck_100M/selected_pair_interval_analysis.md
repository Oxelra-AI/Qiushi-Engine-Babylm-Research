# earlier analysis selected pair interval: extractive_balanced minus legal_compact

Created UTC: `2026-09-02T07:58:05Z`

CPU/file-only reading of existing selected predictions; no model scoring, training, SuperGLUE, AoA, upload, or leaderboard submission.

## Official/wrapper selected deltas

- cheap7: 0.2314
- cheap6_no_GlobalPIQA: -0.3158
- cheap5_no_GlobalPIQA_Reading: -0.3400
- EWoK_plus_Entity: -2.2200
- BLiMP: 0.4000
- Supplement: 0.7600
- EWoK: -1.2500
- Entity: -0.9700
- COMPS: -0.6400
- GlobalPIQA: 3.5150
- Reading: -0.1950

## Raw item paired intervals, RIGHT minus LEFT (percentage points)

| subset | n items | item point | item 95% | clusters | cluster point | cluster 95% |
|---|---:|---:|---:|---:|---:|---:|
| stable_five_item_pool | 170519 | -0.3654 | [-0.5827, -0.1267] | 105 | -0.3654 | [-1.7562, 0.7980] |
| EWoK_plus_Entity_item_pool | 14398 | -1.1182 | [-1.8518, -0.3948] | 29 | -1.1182 | [-1.8700, -0.3053] |
| BLiMP | 59875 | 0.4292 | [0.1309, 0.7600] | 67 | 0.4292 | [-0.7904, 1.5517] |
| Supplement | 5218 | -5.5385 | [-6.4876, -4.5511] | 5 | -5.5385 | [-6.8624, 5.1363] |
| EWoK | 7618 | -1.7852 | [-3.0792, -0.4201] | 11 | -1.7852 | [-3.0276, -0.7134] |
| Entity | 6780 | -0.3687 | [-1.2979, 0.5092] | 18 | -0.3687 | [-1.6272, 0.8122] |
| COMPS | 91028 | -0.4724 | [-0.8882, -0.0996] | 4 | -0.4724 | [-4.1654, 2.0071] |
| GlobalPIQA_item_pool | 203 | 3.4483 | [-1.2438, 8.3744] | 2 | 3.4483 | [-0.9709, 8.0000] |

## Coverage and caution
- common items: `170722`, left-only `0`, right-only `0`.
- loader warnings: `0`; skipped columns: {'left': {'Reading': 'continuous reading correlations, not item correctness flips', 'SuperGLUE': 'unsupported_or_nonselected_column', 'AoA': 'unsupported_or_nonselected_column'}, 'right': {'Reading': 'continuous reading correlations, not item correctness flips'}}.
- Use these intervals to interpret item movement; the selected panel's official/wrapper scores remain the primary readout because BabyLM aggregation is not always raw item averaging.

## Files
- json: `experiments/archive/frontier_consolidation/data/extractive_selected_interval_analysis/extractive_balanced/chck_100M/selected_pair_interval_analysis.json`
- md: `research/documents/frontier_consolidation/data/extractive_selected_interval_analysis/extractive_balanced/chck_100M/selected_pair_interval_analysis.md`
- csv: `experiments/archive/frontier_consolidation/data/extractive_selected_interval_analysis/extractive_balanced/chck_100M/subset_intervals.csv`
