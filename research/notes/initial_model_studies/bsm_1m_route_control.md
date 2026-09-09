# bsm 1m route control — route-control analysis for legal BSM 1M

Evidence JSON: `experiments/archive/initial_model_studies/data/bsm_1m_confounds.json`

## Actual 1M prefix packing and exposure

| arm | rows | words | binding words | official words | mean words/row | estimated updates (batch64) | recorded updates |
|---|---:|---:|---:|---:|---:|---:|---:|
| official_control | 6250 | 1000000 | 0 | 1000000 | 160.00 | 98 | 98 |
| bsm_coherent_20pct | 18627 | 999874 | 199234 | 800640 | 53.68 | 292 | 292 |
| bsm_swapped_20pct | 18605 | 999989 | 198869 | 801120 | 53.75 | 291 | 291 |

## Binding-row length contrast

| arm | binding rows | mean binding words/row | official rows | mean official words/row |
|---|---:|---:|---:|---:|
| bsm_coherent_20pct | 13623 | 14.62 | 5004 | 160.00 |
| bsm_swapped_20pct | 13598 | 14.62 | 5007 | 160.00 |

## Official fast profile (legal bsm 1m eval)

| arm | BLiMP | Supplement | Entity | EWoK | GlobalPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|
| official_control | 54.31 | 48.80 | 17.99 | 48.73 | 37.225 | 6.47 |
| bsm_coherent_20pct | 55.29 | 47.60 | 19.85 | 51.00 | 33.265 | 6.13 |
| bsm_swapped_20pct | 53.35 | 50.40 | 19.12 | 50.45 | 36.710 | 6.81 |

## Route implication

The current legal BSM recipe should not be scaled as a mechanism-positive result. It did not form binding in probes, and the official task movements are entangled with row length, update count, official-data replacement, and targeted-mask supervision. Coherent-vs-swapped is the best current contrast but it is not an exact text-multiset control because the two BSM corpora were generated independently. A stronger next experiment must either build exact row-matched coherent/swapped controls plus a matched-update official-experience reference, or redirect toward an architecture/objective that makes binding learnable earlier.
