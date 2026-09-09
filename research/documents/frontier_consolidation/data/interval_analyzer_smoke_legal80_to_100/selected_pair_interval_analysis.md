# earlier analysis selected pair interval: legal_compact_100M minus legal_compact_80M

Created UTC: `2026-09-02T06:51:21Z`

CPU/file-only reading of existing selected predictions; no model scoring, training, SuperGLUE, AoA, upload, or leaderboard submission.

## Official/wrapper selected deltas

- cheap7: 0.0571
- cheap6_no_GlobalPIQA: -0.0142
- cheap5_no_GlobalPIQA_Reading: 0.0140
- EWoK_plus_Entity: -0.2800
- BLiMP: -0.2400
- Supplement: 0.5100
- EWoK: -0.6200
- Entity: 0.3400
- COMPS: 0.0800
- GlobalPIQA: 0.4850
- Reading: -0.1550

## Raw item paired intervals, RIGHT minus LEFT (percentage points)

| subset | n items | item point | item 95% | clusters | cluster point | cluster 95% |
|---|---:|---:|---:|---:|---:|---:|
| stable_five_item_pool | 170519 | -0.0211 | [-0.1616, 0.1044] | 105 | -0.0211 | [-0.5332, 0.3744] |
| EWoK_plus_Entity_item_pool | 14398 | 0.1250 | [-0.2410, 0.4244] | 29 | 0.1250 | [-0.1964, 0.6303] |
| BLiMP | 59875 | -0.2238 | [-0.3547, -0.0235] | 67 | -0.2238 | [-0.6514, 0.1676] |
| Supplement | 5218 | 1.1499 | [0.6851, 1.4378] | 5 | 1.1499 | [-0.1747, 1.5300] |
| EWoK | 7618 | -0.1706 | [-0.8411, 0.2511] | 11 | -0.1706 | [-1.4588, 0.4059] |
| Entity | 6780 | 0.4572 | [-0.1047, 0.9886] | 18 | 0.4572 | [-0.0968, 0.9870] |
| COMPS | 91028 | 0.0220 | [-0.2293, 0.1602] | 4 | 0.0220 | [-0.8044, 0.7684] |
| GlobalPIQA_item_pool | 203 | 0.4926 | [-1.9704, 2.9557] | 2 | 0.4926 | [0.0000, 0.9709] |

## Coverage and caution
- common items: `170722`, left-only `0`, right-only `0`.
- loader warnings: `0`; skipped columns: {'left': {'Reading': 'continuous reading correlations, not item correctness flips'}, 'right': {'Reading': 'continuous reading correlations, not item correctness flips', 'SuperGLUE': 'unsupported_or_nonselected_column', 'AoA': 'unsupported_or_nonselected_column'}}.
- Use these intervals to interpret item movement; the selected panel's official/wrapper scores remain the primary readout because BabyLM aggregation is not always raw item averaging.

## Files
- json: `experiments/archive/frontier_consolidation/data/interval_analyzer_smoke_legal80_to_100/selected_pair_interval_analysis.json`
- md: `research/documents/frontier_consolidation/data/interval_analyzer_smoke_legal80_to_100/selected_pair_interval_analysis.md`
- csv: `experiments/archive/frontier_consolidation/data/interval_analyzer_smoke_legal80_to_100/subset_intervals.csv`
