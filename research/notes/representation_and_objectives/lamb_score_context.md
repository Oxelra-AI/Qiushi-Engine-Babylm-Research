# lamb config audit — score context for lamb curriculum route decision LAMB endpoints

lamb curriculum route decision endpoints must be interpreted by cheap7 vectors before any full official conclusion.

| reference | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | cheap7 | SuperGLUE | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legal40k_8x480_adamw_fixed256_seed43022 | 67.949 | 60.966 | 51.474 | 27.204 | 51.675 | 34.665 | 7.823 | 43.108 | 68.510 | 41.141 |
| legal40k_12x384_adamw_fixed256_seed43022 | 67.475 | 60.140 | 50.547 | 27.171 | 52.702 | 35.636 | 7.349 | 43.003 | 68.228 | 41.028 |
| visible_leader_41p80 | 67.200 | 56.010 | 56.070 | 28.450 | 53.570 | 39.670 | 5.420 | 43.770 | 69.790 | 41.800 |

Best matched baseline cheap7: **legal40k_8x480_adamw_fixed256_seed43022 = 43.107852**.
Visible leader cheap7: **43.770000** (gap from best matched baseline 0.662148).

Cheap7 needed to reach Overall 41.80 with AoA=0 depends on SuperGLUE:
- if SuperGLUE equals 8×480 baseline (68.510): cheap7 > 43.956
- if SuperGLUE equals 12×384 baseline (68.228): cheap7 > 43.996
- if SuperGLUE equals visible leader (69.790): cheap7 > 43.773

Full official evaluation is scientifically justified only if cheap7 is near or above this crossing region, or if the per-column vector shows a plausible SuperGLUE-backed crossing. Otherwise first analyze the mechanism and do matched cheaper controls.

Summary JSON: `experiments/archive/representation_and_objectives/data/lamb_score_context/lamb_score_context.json`
