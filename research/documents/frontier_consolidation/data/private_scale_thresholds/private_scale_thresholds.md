# coherent86 multiarm mechanism reading private-scale SuperGLUE thresholds

Protected chck82 Overall: `41.942481167385985`; SuperGLUE `69.7661813713118`; cheap7 `43.95944987645173`.
Alpha1 first SuperGLUE: `69.77796826428681`.

| arm | cheap7 | SG needed to match chck82 | slack below chck82 SG | SG needed to match alpha1 | slack below alpha1 SG | Overall if alpha1 SG |
|---|---:|---:|---:|---:|---:|---:|
| coherent86_private_alpha0p5 | 44.17785714285714 | 68.23733050647388 | 1.5288508648379207 | 69.27796826428681 | 0.5 | 42.11366314047631 |
| coherent86_private_alpha0p75 | 44.18142857142857 | 68.2123305064739 | 1.553850864837898 | 69.25296826428684 | 0.5249999999999773 | 42.11644091825409 |
| coherent86_private_alpha1 | 44.10642857142857 | 68.73733050647388 | 1.0288508648379207 | 69.77796826428681 | 0.0 | 42.058107584920755 |

If alpha0.5/0.75 SuperGLUE stays near the already observed coherent/chck82 range, their higher cheap7 makes them stronger no-training materialized endpoints; mechanism remains amplitude-controlled redistribution.

JSON: `experiments/archive/frontier_consolidation/data/private_scale_thresholds/private_scale_thresholds.json`
