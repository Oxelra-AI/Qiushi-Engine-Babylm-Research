# aoa overall update — official-rowcount AoA and reinvest Overall update

The repaired AoA run used min_context=0, exactly 8005 rows per checkpoint, and 19 checkpoint steps for both compact_view_reinvest seeds.

## Seed43022 submission-facing arithmetic
- BLiMP: 66.87
- Supplement: 63.28
- EWoK: 53.67
- Entity: 27.75
- COMPS: 51.97
- GlobalPIQA: 35.62
- SuperGLUE: 71.3810714722
- Reading: 8.24
- AoA: 0
- Overall: 42.0867857191
- Delta versus visible 41.8 leader: +0.286785719138

## AoA rowcount result
- reinvest_seed43022: AoA=0.0, row_count_values=[8005], num_rows=152095, curve_fitness={'curve_fitness': 0.0, 'n_words': 241}
- reinvest_seed43122: AoA=0.0, row_count_values=[8005], num_rows=152095, curve_fitness={'curve_fitness': 0.0, 'n_words': 233}

## Consequence for seed43122
Seed43122 also has official-rowcount AoA=0, but its fast seven-column sum excluding SuperGLUE/AoA is 300.140000. With AoA=0 it would need SuperGLUE 76.060000 under the optimistic fast projection, or 78.360000 after applying seed43022 fast-to-full shifts. Thus AoA=0 removes the catastrophic failure mode but does not restore seed43122 as a likely above-41.8 replicate.

## Scientific interpretation
The endpoint is now stronger as a single submission-facing model: its load-bearing AoA convention has been recomputed under the live 8005-row convention and remains 0. The broader recipe, however, is not yet stable: the second seed's no-AoA surface is broadly weaker, so the active research question shifts from whether seed43022 is valid to why the density mechanism produces a high upper-tail seed and how to make it reliable.

Machine-readable output: `experiments/archive/frontier_consolidation/data/aoa_overall_update/aoa_overall_update.json`
