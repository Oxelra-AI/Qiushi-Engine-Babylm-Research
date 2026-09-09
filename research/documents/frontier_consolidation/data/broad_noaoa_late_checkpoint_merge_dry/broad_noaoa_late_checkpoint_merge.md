# existing ladder checkpoint route state — merged broad no-AoA late-checkpoint screen

This merge combines the two split GPU evaluation roots for selected existing compact_view_reinvest checkpoints. It is not a full official score because SuperGLUE, AoA, and full current official collation are absent.

Summary JSON: `experiments/archive/frontier_consolidation/data/broad_noaoa_late_checkpoint_merge_dry/broad_noaoa_late_checkpoint_merge.json`


## Candidate checkpoint scores
| target | BLiMP | Supp | EWoK-fast | Entity full | COMPS | GPIQA | Reading | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r43022_80M |  |  |  |  |  |  |  |  |
| r43022_90M |  |  |  |  |  |  |  |  |
| r43122_45M |  |  |  |  |  |  |  |  |
| r43122_80M |  |  |  |  |  |  |  |  |

## Known 100M fast references
| reference | BLiMP | Supp | EWoK-fast | Entity full | COMPS | GPIQA | Reading | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r43022_100M_fast | 66.630 | 66.400 | 53.090 | 27.750 | 51.970 | 35.620 | 8.240 | 44.243 |
| r43122_100M_fast | 65.750 | 63.200 | 49.360 | 26.290 | 51.540 | 35.135 | 8.865 | 42.877 |

## Contrasts vs own 100M fast reference
- **r43022_80M_minus_r43022_100M_fast**: 
- **r43022_90M_minus_r43022_100M_fast**: 
- **r43122_45M_minus_r43122_100M_fast**: 
- **r43122_80M_minus_r43122_100M_fast**: 
- **r43122_80M_minus_r43022_80M**: 
- **r43022_90M_minus_r43022_80M**: 
- **r43122_45M_minus_r43122_80M**: 

## Scientific read
- Some target payloads or scores are absent, so no stopping or averaging decision should be made from this merge alone.
- incomplete scores for r43022_80M: equal7_full_entity is absent; incomplete scores for r43122_80M: equal7_full_entity is absent; incomplete scores for r43022_90M: equal7_full_entity is absent; incomplete scores for r43122_45M: equal7_full_entity is absent
