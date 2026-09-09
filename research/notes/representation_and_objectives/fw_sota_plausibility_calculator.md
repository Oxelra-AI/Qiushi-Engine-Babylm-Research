# fw 70m ewok prediction overlap — FW cheap7 to Overall plausibility

This CPU-only calculator translates available cheap-column screens into the SuperGLUE+AoA or cheap7 burden required to reach Overall 41.80. It does not evaluate models.

## Required cheap7 for Overall 41.80

- If SuperGLUE+AoA = 67.00, required cheap7 = 44.1714.
- If SuperGLUE+AoA = 68.00, required cheap7 = 44.0286.
- If SuperGLUE+AoA = 69.79, required cheap7 = 43.7729.
- If SuperGLUE+AoA = 70.00, required cheap7 = 43.7429.
- If SuperGLUE+AoA = 71.04, required cheap7 = 43.5943.
- If SuperGLUE+AoA = 72.00, required cheap7 = 43.4571.
- If SuperGLUE+AoA = 75.00, required cheap7 = 43.0286.
- If SuperGLUE+AoA = 80.00, required cheap7 = 42.3143.

Visible leader cheap7 = 43.7700 with SuperGLUE+AoA = 69.79. Known complete local vectors with full columns have SuperGLUE+AoA range 67.08–71.04.

## Available FW rows

- 70M compact_view: cheap7=42.6571; would need SuperGLUE+AoA=77.60 to reach 41.80 at that cheap7.
- 70M source_breadth_rowblock: cheap7=42.0314; would need SuperGLUE+AoA=81.98 to reach 41.80 at that cheap7.
- 80M compact_view: only 0/7 cheap columns; no cheap7 burden yet.
- 80M source_breadth_rowblock: only 0/7 cheap columns; no cheap7 burden yet.
- 100M compact_view: only 0/7 cheap columns; no cheap7 burden yet.
- 100M source_breadth_rowblock: only 0/7 cheap columns; no cheap7 burden yet.

## Use

Do not launch full official evaluation merely from a partial 70M advantage. Once complete paired 80M or 100M cheap7 exists, use this file to decide whether full official evaluation/readouts can plausibly change the SOTA judgment. If 100M cheap7 is far below about 43.5–43.8 under plausible SuperGLUE+AoA, the arm cannot reach Overall 41.80 without an historically large SuperGLUE jump.

## Evidence files

- JSON: `experiments/archive/representation_and_objectives/data/fw_sota_plausibility/fw_sota_plausibility_calculator.json`
- Threshold CSV: `experiments/archive/representation_and_objectives/data/fw_sota_plausibility/fw_sota_thresholds.csv`
- FW rows CSV: `experiments/archive/representation_and_objectives/data/fw_sota_plausibility/fw_available_plausibility_rows.csv`
- Partial requirements CSV: `experiments/archive/representation_and_objectives/data/fw_sota_plausibility/fw_partial_missing_requirements.csv`
