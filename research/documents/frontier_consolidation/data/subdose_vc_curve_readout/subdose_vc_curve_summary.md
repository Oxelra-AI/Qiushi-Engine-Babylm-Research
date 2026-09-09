# earlier analysis geometry-matched sub-dose V-C curve readout

File-only readout; no model loading, training, official evaluation, GPU work, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.

## Geometry

clean, quarter_1x, half_1x, full_1x_maxgeom, max_2p64x all use 65,313-row MAX geometry

Excluded 1.82x context: rows=65041 versus MAX rows=65313; earlier analysis 1.82x has a different 65,041-row sequence, so it is not part of the strict row-matched curve.

## Trajectory means over complete common-window checkpoints

| label | rho | admitted words | docs | complete ck count | cheap6 V-C | ex-Entity V-C | Entity V-C | clean spread cheap6 mean window |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| quarter_1x | 0.0106 | 105962 | 2861 | 0 |  |  |  |  |
| half_1x | 0.0212 | 211853 | 4247 | 0 |  |  |  |  |
| full_1x_maxgeom | 0.0424 | 423559 | 4529 | 0 |  |  |  |  |
| max_2p64x | 0.1119 | 1118587 | 5261 | 4 | 0.5306 | 0.4313 | 1.0275 |  |

JSON: `experiments/archive/frontier_consolidation/data/subdose_vc_curve_readout/subdose_vc_curve_summary.json`
Deltas: `experiments/archive/frontier_consolidation/data/subdose_vc_curve_readout/vc_delta_rows.csv`
