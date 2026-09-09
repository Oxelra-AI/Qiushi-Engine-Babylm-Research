# fw weight space sweep — first-screen readout for FW weight-space mixtures

This readout tested a fixed, small set of same-initialization parameter mixtures before any new training or full official evaluation. It reads broad preservation through Supplement/Entity, relation movement through current-coordinate EWoK and GlobalPIQA all-option margins, and does not run SuperGLUE/AoA.

## Endpoint references

| endpoint | Supplement | Entity | EWoK | GlobalPIQA | GP-parallel | GP-nonparallel | broad pair | relation pair | hard52 margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 58.86 | 28.36 | 50.25 | 38.635 | 24.27 | 53.00 | 43.610 | 44.442 | 1.717 |
| interleaved | 56.42 | 25.60 | 51.85 | 41.105 | 26.21 | 56.00 | 41.010 | 46.477 | 1.660 |
| rowblock | 59.69 | 23.88 | 50.50 | 37.065 | 29.13 | 45.00 | 41.785 | 43.782 | 1.433 |

## Mixture readout

| mixture | Supplement | Entity | EWoK | GlobalPIQA | GP-parallel | GP-nonparallel | broad pair | relation pair | screen4 | hard52 acc | hard52 margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cr_a0p25 | 57.53 | 23.03 | 49.96 | 35.650 | 23.30 | 48.00 | 40.280 | 42.805 | 41.543 | 0.00 | 1.467 |
| cir_i0p25_r0p25 | 55.25 | 18.94 | 50.58 | 35.209 | 19.42 | 51.00 | 37.095 | 42.894 | 39.995 | 3.85 | 1.384 |

Best mixture by readout: {"screen4": "cr_a0p25", "broad_pair": "cr_a0p25", "relation_pair": "cir_i0p25_r0p25", "globalpiqa_parallel": "cr_a0p25", "hard52_margin_lowest": "cir_i0p25_r0p25"}

Interpretation should compare these mixtures to the endpoints rather than choose a fine-grained official-row lambda. A useful branch-consolidation model would keep compact-like Supplement/Entity while moving EWoK/GlobalPIQA and hard52 margins toward the breadth directions.

Files:
- summary JSON: `experiments/archive/representation_and_objectives/data/weight_sweep_rowblock_screen/weight_sweep_first_screen_summary.json`
- zero surface: `experiments/archive/representation_and_objectives/data/weight_sweep_rowblock_screen/zero_surface`
- EWoK: `experiments/archive/representation_and_objectives/data/weight_sweep_rowblock_screen/official_ewok`
- GlobalPIQA margins: `experiments/archive/representation_and_objectives/data/weight_sweep_rowblock_screen/globalpiqa_margin`
