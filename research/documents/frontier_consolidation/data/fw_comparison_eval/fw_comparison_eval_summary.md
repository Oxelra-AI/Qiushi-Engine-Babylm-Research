# earlier analysis FW compact-view vs whole-sentence source-breadth evaluation

Created UTC: 2026-08-30T21:44:47Z

| Checkpoint | Arm | BLiMP | Suppl | EWoK | Entity | COMPS | GPIQA | Reading | cheap7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 70M | compact_view | 65.62 | 57.94 | 49.42 | 27.74 | 51.76 | 38.61 | 7.515 | 42.6571 |
| 70M | source_breadth | 65.98 | 58.39 | 51.44 | 22.44 | 51.31 | 36.61 | 8.055 | 42.0314 |
| 80M | compact_view | 66.37 | 59.21 | 50.17 | 27.72 | 52.01 | 38.13 | 7.290 | 42.9864 |
| 80M | source_breadth | 67.31 | 60.14 | 51.76 | 23.96 | 51.25 | 38.05 | 8.290 | 42.9657 |
| 100M | compact_view | 66.87 | 58.86 | 50.25 | 28.36 | 51.84 | 38.63 | 7.455 | 43.1814 |
| 100M | source_breadth | 67.89 | 59.69 | 50.50 | 23.88 | 51.33 | 37.06 | 8.105 | 42.6371 |

## Compact minus breadth deltas

| Checkpoint | Δcheap7 | ΔBLiMP | ΔSuppl | ΔEWoK | ΔEntity | ΔCOMPS | ΔGPIQA | ΔReading | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 70M | +0.6257 | -0.36 | -0.45 | -2.02 | +5.30 | +0.45 | +2.00 | -0.540 | compact_view_better_on_predeclared_rule |
| 80M | +0.0207 | -0.94 | -0.93 | -1.59 | +3.76 | +0.76 | +0.09 | -1.000 | too_close_next_source_repeat_test |
| 100M | +0.5443 | -1.02 | -0.83 | -0.25 | +4.48 | +0.51 | +1.57 | -0.650 | compact_view_better_on_predeclared_rule |
