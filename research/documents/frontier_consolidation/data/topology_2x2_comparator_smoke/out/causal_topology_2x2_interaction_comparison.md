# topology 2x2 scaffold and deberta pending state causal topology 2×2 interaction comparison

This file compares completed future trajectories; it does not evaluate or train models.

Common endpoints: 2. Broad positive endpoints: 2; robust without GlobalPIQA/Reading: 2.

## Interaction summaries

- cheap7: mean 0.300000, std 0.000000, positive 2/2, min 0.300000, max 0.300000.
- cheap6_no_global: mean 0.300000, std 0.000000, positive 2/2, min 0.300000, max 0.300000.
- cheap5_no_global_reading: mean 0.300000, std 0.000000, positive 2/2, min 0.300000, max 0.300000.
- syntax_surface: mean 0.300000, std 0.000000, positive 2/2, min 0.300000, max 0.300000.
- relation_state: mean 0.300000, std 0.000000, positive 2/2, min 0.300000, max 0.300000.
- volatile_small: mean 0.300000, std 0.000000, positive 2/2, min 0.300000, max 0.300000.

Best endpoint by cheap7 interaction: chck_20M (0.300000); worst: chck_20M (0.300000).

| endpoint | sem one cheap7 | sem rec cheap7 | topo compact | topo repeat | interaction cheap7 | int cheap6 noG | int cheap5 noG/R | int rel/state | int syntax | int volatile | +int cols | max +share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chck_20M | 0.200000 | 0.500000 | 0.500000 | 0.200000 | 0.300000 | 0.300000 | 0.300000 | 0.300000 | 0.300000 | 0.300000 | 7 | 0.143 |
| chck_40M | 0.200000 | 0.500000 | 0.600000 | 0.300000 | 0.300000 | 0.300000 | 0.300000 | 0.300000 | 0.300000 | 0.300000 | 7 | 0.143 |

## Scientific reading

The topology route remains live only if reciprocal topology specifically increases the compact semantic advantage over repeat, with breadth across official families and persistence after removing GlobalPIQA/Reading. Otherwise the result is recurrence/copy/volatile-column redistribution.

JSON: `experiments/archive/frontier_consolidation/data/topology_2x2_comparator_smoke/out/causal_topology_2x2_interaction_comparison.json`
