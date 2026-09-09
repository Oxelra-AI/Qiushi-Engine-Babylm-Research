# pure order profile interpretation — Pure-order dense profile vs identical-data random full-pool baseline

Evidence JSON: `experiments/archive/initial_model_studies/data/pure_order_profile.json`

Each order run consumes exactly the same selected example set and source mix as its random full-pool dense baseline (earlier analysis/27), changing only training order. 1M exposure, `lr_total_steps=98`, shared-core init, `chck_1M` eval.

## Scores (order runs)

| seed | order | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR | loss_last |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | source_stage | 53.63 | 46.00 | 50.36 | 16.23 | 50.34 | 10.26 | 3.33 | 7.10 |
| 42 | readability_interleave | 54.02 | 49.20 | 48.91 | 17.14 | 50.38 | 9.94 | 3.03 | 6.30 |
| 43 | source_stage | 54.39 | 50.80 | 52.00 | 17.27 | 49.83 | 9.58 | 3.12 | 7.05 |
| 43 | readability_interleave | 53.94 | 48.40 | 50.09 | 18.32 | 49.67 | 9.76 | 3.17 | 6.37 |

## Random full-pool dense baselines (from controlled grid profile interpretation)

| seed | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 53.37 | 48.80 | 46.73 | 18.53 | 50.46 | 10.24 | 3.07 |
| 43 | 53.79 | 51.20 | 51.45 | 18.05 | 50.07 | 9.70 | 3.07 |

## Order minus random deltas

| seed | order | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | source_stage | 0.26 | -2.80 | 3.63 | -2.30 | -0.12 | 0.02 | 0.26 |
| 42 | readability_interleave | 0.65 | 0.40 | 2.18 | -1.39 | -0.08 | -0.30 | -0.04 |
| 43 | source_stage | 0.60 | -0.40 | 0.55 | -0.78 | -0.24 | -0.12 | 0.05 |
| 43 | readability_interleave | 0.15 | -2.80 | -1.36 | 0.27 | -0.40 | 0.06 | 0.10 |

## Mean order-minus-random by mode

| order | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| source_stage | 0.43 | -1.60 | 2.09 | -1.54 | -0.18 | -0.05 | 0.15 |
| readability_interleave | 0.40 | -1.20 | 0.41 | -0.56 | -0.24 | -0.12 | 0.03 |

## Note

Single-pass source_stage places hard Gutenberg/SimpleWiki text at the end of training, so a higher loss_last and any BLiMP/Supplement drop may be an end-of-pass artifact rather than a curriculum failure. Judge order effects by seed-stable profile deltas, and if source_stage looks penalized mainly on syntax, test a multi-epoch or repeated-interleave variant that removes the end-of-pass artifact.

