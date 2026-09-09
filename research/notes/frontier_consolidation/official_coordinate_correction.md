# clean full trajectory seed control official-coordinate correction for compact_view_reinvest

Use this local correction instead of the aoa overall update `26_aoa_overall_update` Overall arithmetic.

## Current seed43022 official coordinate
- BLiMP: 66.87232315173485
- Supplement: 63.27576417952158
- EWoK: 53.536575594886855
- Entity: 27.745741097952372
- COMPS: 51.968828052457084
- SuperGLUE: 71.03604952825312
- GlobalPIQA: 35.62135922330097
- Reading: 8.241572282566393
- AoA: 0.0
- Overall: 42.0331347900748
- Margin over visible 41.8 leader: 0.23313479007479998
- NLP average: 52.86523440401526
- Human-like average: 4.1207861412831965

## Correction
Earlier seed43122 fast localization/aoa overall update arithmetic used 42.086785719138156. The current official-coordinate value is lower by 0.05365092906335889. The coordinate changed because EWoK was re-scored on the pristine 7618-row official data and SuperGLUE was aggregated by the official primary metrics.

## Independent AoA confirmation
- reinvest_seed43022: AoA=0.0, row_count_values=[8005], num_rows=152095, n_words=241
- reinvest_seed43122: AoA=0.0, row_count_values=[8005], num_rows=152095, n_words=233

Seed43022 remains above the visible leader as a single official-coordinate endpoint, but seed43122 robustness and the mechanism/source of seed spread are still active research questions.

Machine-readable output: `experiments/archive/frontier_consolidation/data/official_coordinate_correction/official_coordinate_correction.json`
