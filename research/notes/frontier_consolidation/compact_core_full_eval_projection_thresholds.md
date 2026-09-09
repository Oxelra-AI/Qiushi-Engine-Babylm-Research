# compact core full eval projection thresholds compact-core full-evaluation projection thresholds

JSON: `experiments/archive/frontier_consolidation/data/projection_thresholds/compact_core_full_eval_thresholds.json`
Fast-screen source: `experiments/archive/frontier_consolidation/data/density_noaoa_eval_compact_core/density_noaoa_eval_summary.json`

This is arithmetic context only. The running full official-compatible evaluation remains the evidence.

| arm | fast seven-column mean | need SG+AoA for 41.344 | need SG+AoA for 41.8 | need SG+AoA for 42.0 | projected Overall if SG=clean and AoA=0 |
|---|---:|---:|---:|---:|---:|
| compact_repeat_core | 41.7221 | 80.044 | 84.145 | 85.945 | 40.2626 |
| compact_view_core | 43.8493 | 65.154 | 69.255 | 71.055 | 41.9171 |

For compact_view_core, preserving a clean-Qwen-like SuperGLUE with AoA=0 would project above the visible 41.8 surface from the fast seven-column screen, but a negative AoA or full-split degradation can erase that margin. This is why the full evaluation is decisive before any recipe changes.
