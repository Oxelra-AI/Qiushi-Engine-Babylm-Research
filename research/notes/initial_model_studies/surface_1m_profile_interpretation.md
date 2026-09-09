# surface 1m profile interpretation — Surface 1M profile vs earlier analysis dense baselines

Evidence JSON: `experiments/archive/initial_model_studies/data/surface_1m_profile.json`

All surface runs use identical consumed examples/order/source mix as their earlier analysis dense pool10M baseline, with pairing seeds derived from baseline configs. 1M exposure, `lr_total_steps=98`, shared-core init, `chck_1M` eval.

## Scores (surface runs)

| seed | mode | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR | loss_last |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | char_ngram | 53.42 | 47.20 | 46.82 | 17.78 | 50.26 | 9.80 | 3.27 | 5.90 |
| 42 | lookup | 53.40 | 46.80 | 47.00 | 18.24 | 50.31 | 9.83 | 3.30 | 5.90 |
| 43 | char_ngram | 54.16 | 48.00 | 48.91 | 18.52 | 49.96 | 9.37 | 3.07 | 5.79 |
| 43 | lookup | 53.80 | 50.80 | 49.91 | 17.57 | 49.89 | 9.29 | 3.08 | 5.83 |

## Dense baselines (controlled grid profile interpretation)

| seed | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 53.37 | 48.80 | 46.73 | 18.53 | 50.46 | 10.24 | 3.07 |
| 43 | 53.79 | 51.20 | 51.45 | 18.05 | 50.07 | 9.70 | 3.07 |

## Surface minus dense deltas

| seed | mode | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | char_ngram | 0.05 | -1.60 | 0.09 | -0.75 | -0.20 | -0.44 | 0.20 |
| 42 | lookup | 0.03 | -2.00 | 0.27 | -0.29 | -0.15 | -0.41 | 0.23 |
| 43 | char_ngram | 0.37 | -3.20 | -2.54 | 0.47 | -0.11 | -0.33 | 0.00 |
| 43 | lookup | 0.01 | -0.40 | -1.54 | -0.48 | -0.18 | -0.41 | 0.01 |

## Char minus lookup deltas (per seed)

| seed | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.02 | 0.40 | -0.18 | -0.46 | -0.05 | -0.03 | -0.03 |
| 43 | 0.36 | -2.80 | -1.00 | 0.95 | 0.07 | 0.08 | -0.01 |

## Mean surface-minus-dense by mode

| mode | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| char_ngram | 0.21 | -2.40 | -1.23 | -0.14 | -0.15 | -0.39 | 0.10 |
| lookup | 0.02 | -1.20 | -0.64 | -0.39 | -0.17 | -0.41 | 0.12 |

## Interpretation guidance

- If char_ngram improves Entity/EWoK/Reading vs dense AND vs lookup, the gain is from shared surface structure.

- If char_ngram and lookup perform similarly, the effect is added capacity/fusion, not character sharing.

- If both hurt BLiMP/Supplement or Entity relative to dense, surface composition at this scale does not solve the tradeoff.

