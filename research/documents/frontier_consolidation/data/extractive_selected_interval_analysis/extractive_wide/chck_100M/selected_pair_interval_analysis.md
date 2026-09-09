# earlier analysis selected pair interval: extractive_wide minus legal_compact

Created UTC: `2026-09-02T07:58:15Z`

CPU/file-only reading of existing selected predictions; no model scoring, training, SuperGLUE, AoA, upload, or leaderboard submission.

## Official/wrapper selected deltas

- cheap7: -0.0771
- cheap6_no_GlobalPIQA: -0.5158
- cheap5_no_GlobalPIQA_Reading: -0.6040
- EWoK_plus_Entity: -2.3700
- BLiMP: 1.4400
- Supplement: -2.0300
- EWoK: 0.0400
- Entity: -2.4100
- COMPS: -0.0600
- GlobalPIQA: 2.5550
- Reading: -0.0750

## Raw item paired intervals, RIGHT minus LEFT (percentage points)

| subset | n items | item point | item 95% | clusters | cluster point | cluster 95% |
|---|---:|---:|---:|---:|---:|---:|
| stable_five_item_pool | 170519 | 0.1800 | [-0.0647, 0.4193] | 105 | 0.1800 | [-1.0474, 1.8172] |
| EWoK_plus_Entity_item_pool | 14398 | -1.0557 | [-1.7469, -0.2212] | 29 | -1.0557 | [-2.0721, 0.0465] |
| BLiMP | 59875 | 1.4246 | [1.1181, 1.6977] | 67 | 1.4246 | [0.2462, 2.7763] |
| Supplement | 5218 | -4.5036 | [-5.5596, -3.4376] | 5 | -4.5036 | [-5.6330, -0.4228] |
| EWoK | 7618 | -0.4594 | [-1.5358, 0.7289] | 11 | -0.4594 | [-1.8799, 1.4501] |
| Entity | 6780 | -1.7257 | [-2.5819, -0.8842] | 18 | -1.7257 | [-2.9793, -0.4439] |
| COMPS | 91028 | -0.1747 | [-0.5512, 0.1595] | 4 | -0.1747 | [-3.1918, 3.2224] |
| GlobalPIQA_item_pool | 203 | 2.4631 | [-2.7217, 7.6478] | 2 | 2.4631 | [-3.8835, 9.0000] |

## Coverage and caution
- common items: `170722`, left-only `0`, right-only `0`.
- loader warnings: `0`; skipped columns: {'left': {'Reading': 'continuous reading correlations, not item correctness flips', 'SuperGLUE': 'unsupported_or_nonselected_column', 'AoA': 'unsupported_or_nonselected_column'}, 'right': {'Reading': 'continuous reading correlations, not item correctness flips'}}.
- Use these intervals to interpret item movement; the selected panel's official/wrapper scores remain the primary readout because BabyLM aggregation is not always raw item averaging.

## Files
- json: `experiments/archive/frontier_consolidation/data/extractive_selected_interval_analysis/extractive_wide/chck_100M/selected_pair_interval_analysis.json`
- md: `research/documents/frontier_consolidation/data/extractive_selected_interval_analysis/extractive_wide/chck_100M/selected_pair_interval_analysis.md`
- csv: `experiments/archive/frontier_consolidation/data/extractive_selected_interval_analysis/extractive_wide/chck_100M/subset_intervals.csv`
