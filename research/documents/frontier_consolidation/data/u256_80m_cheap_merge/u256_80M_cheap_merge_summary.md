# earlier analysis — U256 chck_80M cheap-column merge

This merges the two split cheap-column screens from endpoint synthesis scale1p75 u256. It is not a full official checkpoint evaluation; it is the low-cost test of whether U256 had an intermediate cheap-surface peak worth SuperGLUE/AoA follow-up.

| Column | U256 80M | spatial repair route status 100M | scale1.75 100M | U256 100M | Δ80M-spatial repair route status | Δ80M-scale | Δ80M-U256100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 66.950000 | 65.870718 | 68.631753 | 66.817235 | 1.079282 | -1.681753 | 0.132765 |
| Supplement | 61.080000 | 61.165661 | 62.895171 | 60.847700 | -0.085661 | -1.815171 | 0.232300 |
| EWoK | 49.880000 | 50.393237 | 49.080093 | 50.386096 | -0.513237 | 0.799907 | -0.506096 |
| Entity | 27.500000 | 27.400834 | 27.464549 | 28.433145 | 0.099166 | 0.035451 | -0.933145 |
| COMPS | 51.430000 | 52.008345 | 52.303917 | 51.645492 | -0.578345 | -0.873917 | -0.215492 |
| GlobalPIQA | 36.605000 | 36.063107 | 36.106796 | 36.135922 | 0.541893 | 0.498204 | 0.469078 |
| Reading | 7.465000 | 8.138168 | 8.319840 | 7.322907 | -0.673168 | -0.854840 | 0.142093 |
| cheap7 | 42.987143 | 43.005724 | 43.543160 | 43.084071 | -0.018581 | -0.556017 | -0.096928 |

Projected Overall if spatial repair route status SuperGLUE/AoA are borrowed only for triage: `41.243319`.
Projected Overall if U256-100M SuperGLUE/AoA are borrowed only for triage: `41.253854`.
Required SuperGLUE at AoA=0 for Overall 41.8: `75.290000`.

Decision: full U256-80M SuperGLUE/AoA follow-up supported? `False`.
Reason: U256 80M cheap7 is below spatial repair route status and far below scale1.75 100M; with AoA=0 it would require SuperGLUE above any observed U256/spatial repair route status/scale1.75 value to approach 41.8.

JSON: `experiments/archive/frontier_consolidation/data/u256_80m_cheap_merge/u256_80M_cheap_merge_summary.json`
