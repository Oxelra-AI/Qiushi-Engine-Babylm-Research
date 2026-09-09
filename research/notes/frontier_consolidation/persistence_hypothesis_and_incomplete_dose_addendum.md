# Persistence Hypothesis and an Incomplete Dose Trajectory

Status: completed reference measurements and historical partial-run assessment; the persistence mechanism and outstanding substitution comparisons remained hypotheses.

The full-dose run stopped at its 70M checkpoint after a timeout. Its 10M-to-70M checkpoints could support a bounded dose curve, but no 80M-to-100M full-dose comparison existed. Missing late measurements must not be filled by another arm or inferred from a checkpoint naming convention.

The clean reference values were:

| Exposure | exEntity4 | Entity | cheap5 |
|---|---:|---:|---:|
| 40M | 54.273 | 25.490 | 48.516 |
| 50M | 55.737 | 23.200 | 49.230 |
| 60M | 55.670 | 24.930 | 49.522 |
| 70M | 56.495 | 24.440 | 50.084 |
| 80M | 56.282 | 24.050 | 49.836 |
| 90M | 56.617 | 24.950 | 50.284 |
| 100M | 56.675 | 24.820 | 50.304 |

In the MAX triangle, repetition-minus-clean exEntity5 fell from +0.4361 over 10M-80M to +0.0343 over 80M-100M; view-minus-clean changed +0.4998 to +0.3853; breadth-minus-clean changed +0.6691 to +0.3350. Distinct text retained more late benefit than repetition in these recorded comparisons.

The proposed explanation was that repeated changed-block rows became easier to predict while view/breadth rows retained error and useful learning signal. A changed-versus-unchanged MLM-loss trajectory was designed to test this. Lower late repeat loss would support that explanation; its absence would refute the residual-error account, not erase the measured persistence differences.

Further discriminators remained unfinished: low-dose onset versus threshold, sacrificed-register opportunity cost, and in-corpus adult prose versus FineWeb. The predeclared register prediction was child-minus-adult exEntity5 approximately +0.748. Equal-register outcomes would weaken a register-cost explanation; in-corpus similarity to FineWeb would favor distribution proximity, whereas similarity to clean would favor distinct external content. Architecture dependence and Entity operation-propensity redistribution remained boundaries, not universal learning laws.
