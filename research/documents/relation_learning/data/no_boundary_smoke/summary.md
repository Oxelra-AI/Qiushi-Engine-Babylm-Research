# earlier analysis no-boundary cheap7 diagnostic

This is not a submission path.  It evaluates the same local masked-LM candidate-ranking tasks with candidate inputs tokenized either as the official pipeline does (`with_special`) or with boundary tokens stripped (`no_special`).

## Scores

| endpoint | input form | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | cheap7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chck82_slow_scale1p75 | with_special | 67.1642 | 100.0000 | 36.3636 | 33.3333 | 75.0000 | 50.0000 |  |  |
| chck82_slow_scale1p75 | no_special | 67.1642 | 100.0000 | 45.4545 | 33.3333 | 75.0000 | 100.0000 |  |  |

## No-boundary minus with-boundary

| endpoint | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | cheap7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chck82_slow_scale1p75 | 0.0000 | 0.0000 | 9.0909 | 0.0000 | 0.0000 | 50.0000 |  |  |

Full JSON: `experiments/archive/relation_learning/data/no_boundary_smoke/summary.json`
