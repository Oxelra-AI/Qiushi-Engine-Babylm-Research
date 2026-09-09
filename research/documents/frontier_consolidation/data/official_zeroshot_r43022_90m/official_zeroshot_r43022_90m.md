# existing ladder checkpoint route state — raw seed43022 90M official-coordinate zero-shot probe

Existing checkpoint only. Full BLiMP, full Supplement, and pristine 7,618-row EWoK were evaluated before any SuperGLUE or AoA work.

Summary JSON: `experiments/archive/frontier_consolidation/data/official_zeroshot_r43022_90m/official_zeroshot_r43022_90m_summary.json`

## Measured official-coordinate zero-shot columns
| column | score | data rows | prediction items | returncode |
|---|---:|---:|---:|---:|
| BLiMP | 67.02 | 59875 | 59875 | 0 |
| Supplement | 63.05 | 5218 | 5218 | 0 |
| EWoK | 53.68 | 7618 | 7618 | 0 |

## Projection using measured zero-shot plus known raw90 no-AoA columns
- Seven measured/known columns excluding SuperGLUE and AoA sum: 306.710000.
- Required SuperGLUE+AoA for 41.8: 69.490000.
- Required SuperGLUE+AoA for 42.0: 71.290000.
- Required SuperGLUE+AoA to exceed seed43022 100M official 42.033135: 71.588213.
- If SuperGLUE matches the seed43022 100M official value and AoA remains 0.0, projected Overall is 41.971783.

## Scientific read
- With full BLiMP/Supplement/EWoK measured and existing ladder checkpoint route state raw90 Entity/COMPS/GlobalPIQA/Reading reused, raw90 needs SuperGLUE+AoA 69.490 to clear 41.8 and 71.588 to exceed the frozen 100M official endpoint.
- If SuperGLUE equals the frozen 100M seed43022 value 71.036 and AoA remains 0.0, raw90 projects to Overall 41.9718.
- The raw90 checkpoint does not improve enough on official zero-shot columns to justify SuperGLUE/AoA escalation unless another measured column changes.
