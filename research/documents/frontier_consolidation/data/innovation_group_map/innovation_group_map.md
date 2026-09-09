# earlier analysis — innovation group map

CPU-only; no model, no training, no evaluation.

- changed rows processed: `3005`
- total innovation groups (one pass): `26435`
- total copyable groups (one pass): `131675`
- skipped empty/single: `3314`

## Per-row statistics
- total groups: mean=140.82, median=143, min=83, max=160
- innovation: mean=8.8, median=9, min=0, max=19
- copyable: mean=43.82, median=44, min=24, max=62
- source: mean=87.07, median=88, min=50, max=107
- other: mean=1.14, median=1, min=0, max=28

## Budget analysis (p_innov=0.5, p_copy=0.0)
- avg innovation selected per changed row: 4.4
- avg standard total selected per row: 21.12
- avg p_other for budget match: 0.1896

Map: `experiments/archive/frontier_consolidation/data/innovation_group_map/innovation_group_map.json`
Summary: `experiments/archive/frontier_consolidation/data/innovation_group_map/innovation_group_map_summary.json`
