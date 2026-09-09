# debertav2 entity diagnostics — DeBERTa-v2 b256 Entity diagnostics

Evidence JSON: `experiments/archive/initial_model_studies/data/debertav2_entity_diagnostics.json`

Overall exact-option accuracy: 21.95 (1488/6780).

## By split

| split | accuracy | correct/n |
|---|---:|---:|
| ambiref | 22.46 | 520/2315 |
| move_contents | 21.55 | 480/2227 |
| regular | 21.81 | 488/2238 |

## By numops

| numops | accuracy | correct/n |
|---|---:|---:|
| 0 | 23.30 | 359/1541 |
| 1 | 16.33 | 208/1274 |
| 2 | 19.23 | 234/1217 |
| 3 | 22.74 | 282/1240 |
| 4 | 26.47 | 311/1175 |
| 5 | 28.23 | 94/333 |

## Tokenizer effect on Entity options

| quantity | baseline16k | official40k | delta |
|---|---:|---:|---:|
| all option mean tokens | 6.474 | 6.474 | 0.000 |
| correct option mean tokens | 6.474 | 6.474 | 0.000 |
| option strings shorter/equal/longer under 40k | 0 | 33900 | 0 |

Subset details and examples are in the JSON.
