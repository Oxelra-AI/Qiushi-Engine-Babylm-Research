# Compact Recurrence Schedule Audit
## Pair text multiset
- Pairs: `12155`; source words `261803`; rewrite words `161708`; pair words `423511`.
- Max individual source/view unit length: `48` words; all below 256 words: `True`.
- Source word stats: `{'n': 12155, 'min': 10, 'mean': 21.538708350473055, 'median': 20, 'p10': 12, 'p90': 33, 'max': 48, 'sum': 261803}`.
- Rewrite word stats: `{'n': 12155, 'min': 4, 'mean': 13.303825586178528, 'median': 13, 'p10': 8, 'p90': 20, 'max': 39, 'sum': 161708}`.

## Current packing
- Changed rows: `3006`; rows with exact source+view substrings for every listed pair: `3006`.
- Pair count per changed row: `{0: 1, 2: 40, 3: 727, 4: 1406, 5: 727, 6: 100, 7: 5}`.
- Source-before-view count: `12154/12155`.
- Word gap between source and its compact view in current stream: `{'n': 12155, 'min': 0, 'mean': 0.0, 'median': 0, 'p10': 0, 'p90': 0, 'max': 0, 'sum': 0}`.
- Row word stats for changed block: `{'n': 3006, 'min': 9, 'mean': 140.8915502328676, 'median': 143.0, 'p10': 122, 'p90': 157, 'max': 160, 'sum': 423520}`.

## Schedule feasibility
- Common filler rows: `61734`.
- If each source/view pair stays one row plus filler: `73889` rows.
- If every source and view becomes a separate row plus filler: `86044` rows.
- The schedule route is mechanically feasible as an exact-text intervention, but any training comparison must include a `repacked_adjacent_split_units` arm so row-boundary effects are not misread as delayed recurrence.

JSON: `experiments/archive/frontier_consolidation/data/compact_recurrence_schedule_audit/compact_recurrence_schedule_audit.json`
