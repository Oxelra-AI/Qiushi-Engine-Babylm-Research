# earlier analysis no-boundary cheap7 diagnostic

This is not a submission path.  It evaluates the same local masked-LM candidate-ranking tasks with candidate inputs tokenized either as the official pipeline does (`with_special`) or with boundary tokens stripped (`no_special`).

## Scores

| endpoint | input form | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | cheap7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense64_scale0p60_smoke | with_special | 67.1642 | 100.0000 | 45.4545 | 33.3333 | 100.0000 | 100.0000 |  |  |

## No-boundary minus with-boundary

| endpoint | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | cheap7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

Full JSON: `experiments/archive/relation_learning/data/shrinkage_cheap7_smoke/summary.json`
