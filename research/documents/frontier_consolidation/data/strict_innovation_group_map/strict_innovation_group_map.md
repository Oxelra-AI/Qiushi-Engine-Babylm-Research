# earlier analysis — strict content-innovation group map

CPU-only; no model update, no official evaluation, no H100 work.

This replaces the earlier analysis loose map. Strict innovation means a rewrite word group that is source-absent relative to its paired source, absent from every other word-group occurrence in the row, and content-like. Duplicate source-absent and relation/function-like groups remain ordinary.

- changed rows processed: `3005`
- rows with strict innovation: `2999`
- strict content-innovation groups per 10M pass: `17968`
- copyable rewrite groups per 10M pass: `131675`
- duplicate source-absent rewrite groups kept ordinary: `3035`
- unique relation/function-like source-absent rewrite groups kept ordinary: `5432`
- short/empty source-absent rewrite groups kept ordinary: `3314`

## Category totals per 10M pass
- total groups in changed rows: `423170`
- strict content innovations: `17968`
- copyable rewrite: `131675`
- source-only groups: `261644`
- non-strict rewrite-other groups: `11781`
  - duplicate source-absent: `3035`
  - unique relation/function-like source-absent: `5432`
  - short/empty source-absent: `3314`
- nonrewrite filler groups: `102`

## Default isolated mass reallocation
- baseline WWM rate: `0.15`
- strict innovation rate: `0.5`
- copyable rewrite rate needed for aggregate mass match: `0.10223998`
- expected extra strict groups: `6288.800000`
- expected removed copyable groups: `6288.800000`
- aggregate balance error: `-0.000000000001`
- all ordinary groups, including source/filler/duplicate/function-like groups, remain at baseline 0.15.

## Per-row means
- total groups: mean=140.8220, median=143.0, min=83.0, max=160.0
- strict: mean=5.9794, median=6.0, min=0.0, max=16.0
- copyable: mean=43.8186, median=44.0, min=24.0, max=62.0
- source-only: mean=87.0696, median=88.0, min=50.0, max=107.0
- rewrite-other: mean=3.9205, median=4.0, min=0.0, max=13.0
- duplicate source-absent: mean=1.0100, median=1.0, min=0.0, max=7.0
- function/relation source-absent: mean=1.8077, median=2.0, min=0.0, max=7.0
- nonrewrite filler: mean=0.0339, median=0.0, min=0.0, max=26.0

## Top strict normalized targets
- `use`: 134
- `shows`: 125
- `says`: 116
- `show`: 96
- `many`: 93
- `using`: 93
- `causing`: 89
- `due`: 83
- `helps`: 83
- `uses`: 78
- `lets`: 77
- `people`: 70
- `where`: 68
- `though`: 60
- `help`: 58
- `yet`: 58
- `needs`: 58
- `found`: 56
- `exist`: 55
- `one`: 51
- `making`: 51
- `about`: 51
- `say`: 51
- `cannot`: 50
- `yearly`: 50

Map: `experiments/archive/frontier_consolidation/data/strict_innovation_group_map/strict_innovation_group_map.json`
Summary JSON: `experiments/archive/frontier_consolidation/data/strict_innovation_group_map/strict_innovation_group_map_summary.json`
