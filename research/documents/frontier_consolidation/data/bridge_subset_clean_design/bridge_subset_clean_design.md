# earlier analysis bridge matched-subset clean design

## Immediate correction
A bridge/compact fallback stream would be mostly natural compact and cannot measure the bridge factor. The clean unit is a whole changed-block row: every pair in the row must have a transformation-like bridge candidate; otherwise the row is excluded or must be regenerated.

## Existing mechanism evidence synthesis/237 material
- Transformation-like bridge pairs available: 55 / 12155
- Whole changed-block rows fully fillable now: 0 / 3005
- Whole-row pair count now: 0 pairs
- Existing fillable row words per 10M: 0 (0.0000% of total; 0.0000% of changed-block words)
- Partial rows with at least one bridge pair: 54

## Geometry on existing whole-row subset
| arm | n pairs | pooled gap1 | pooled skip | pooled absent content | pooled compression |
|---|---:|---:|---:|---:|---:|
| compact | 0 | None | None | 0.0 | None |
| bridge | 0 | None | None | 0.0 | None |
| extractive_balanced | 0 | None | None | 0.0 | None |
| extractive_wide | 0 | None | None | 0.0 | None |

## Interpretation boundaries
- A positive bridge-vs-compact result would not isolate novel vocabulary alone because bridge also changes compression, retained-source fraction, and generation distribution.
- A negative bridge-vs-compact result would implicate the natural-compact bundle: source-absent content, stronger compression, lower source retention, surface quality/distribution, and any remaining row-selection differences.
- Bridge-vs-extractive on the same rows is the test of whether compact-like structural re-expression recovers any of the extractive stable-family deficit.
- Do not run 100M pretraining from a compact/bridge blend; the scale gate must count whole fillable rows and produce matched bridge/compact/extractive subset arms.

JSON: `experiments/archive/frontier_consolidation/data/bridge_subset_clean_design/bridge_subset_clean_design.json`
CSV: `experiments/archive/frontier_consolidation/data/bridge_subset_clean_design/existing_clean_whole_rows.csv`
