# earlier analysis anchor-margin conditioned alpha-sweep census

Status: **COMPLETE**
n_items: `12`

Pattern counts: `{"0001": 1, "0010": 1, "0011": 1, "0100": 1, "0110": 1, "0111": 1, "1000": 1, "1001": 1, "1011": 1, "1100": 1, "1101": 1, "1110": 1}`

## Margin summaries

Anchor margin = official-style anchor log-score(gold) − max log-score(non-gold).

| group | n | median | mean | q05/q25/q50/q75/q95 | frac margin<0 | frac |margin|<0.5 |
|---|---:|---:|---:|---|---:|---:|
| monotonic_activation | 3 | -0.4247 | -0.6234 | [-1.3036, -0.913, -0.4247, -0.2345, -0.0824] | 1.000 | 0.667 |
| monotonic_damage | 3 | +0.4410 | +0.3791 | [0.0897, 0.2458, 0.441, 0.5432, 0.625] | 0.000 | 0.667 |
| nonmonotonic_changed | 6 | -0.0017 | +0.0275 | [-0.0691, -0.0305, -0.0017, 0.0331, 0.1932] | 0.500 | 1.000 |

Activation-vs-damage separability:
- damage minus activation mean signed margin: `+1.002490`
- damage minus activation median signed margin: `+0.865688`
- AUC(damage by signed margin): `1.000000`
- AUC(damage by abs margin): `0.555556`

## By column

| column | n | act n | dmg n | act median margin | dmg median margin | act |margin|<0.5 | dmg |margin|<0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 10 | 3 | 3 | -0.4246866414323449 | 0.44100141525268555 | 0.6666666666666666 | 0.6666666666666666 |
| COMPS | 2 | 0 | 0 | None | None | None | None |

JSON: `experiments/archive/frontier_consolidation/data/anchor_margin_alpha_census_cpu_smoke/anchor_margin_alpha_census.json`
