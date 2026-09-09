# earlier analysis anchor-margin conditioned alpha-sweep census

Status: **COMPLETE**
n_items: `6363`

Pattern counts: `{"0001": 792, "0010": 2, "0011": 771, "0100": 1, "0110": 6, "0111": 1553, "1000": 1575, "1001": 5, "1011": 1, "1100": 837, "1101": 1, "1110": 819}`

## Margin summaries

Anchor margin = official-style anchor log-score(gold) − max log-score(non-gold).

| group | n | median | mean | q05/q25/q50/q75/q95 | frac margin<0 | frac |margin|<0.5 |
|---|---:|---:|---:|---|---:|---:|
| monotonic_activation | 3116 | -0.1397 | -0.2304 | [-0.7836, -0.3169, -0.1397, -0.0561, -0.009] | 0.999 | 0.868 |
| monotonic_damage | 3231 | +0.1330 | +0.2208 | [0.0093, 0.0526, 0.133, 0.3019, 0.7194] | 0.000 | 0.885 |
| nonmonotonic_changed | 16 | -0.0043 | -0.0463 | [-0.2797, -0.0149, -0.0043, 0.006, 0.0935] | 0.562 | 0.938 |

Activation-vs-damage separability:
- damage minus activation mean signed margin: `+0.451240`
- damage minus activation median signed margin: `+0.272697`
- AUC(damage by signed margin): `0.998729`
- AUC(damage by abs margin): `0.488342`

## By column

| column | n | act n | dmg n | act median margin | dmg median margin | act |margin|<0.5 | dmg |margin|<0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 1497 | 761 | 731 | -0.2638195604085922 | 0.2729578986763954 | 0.7148488830486203 | 0.7482900136798906 |
| COMPS | 4336 | 2095 | 2232 | -0.11159297823905945 | 0.10455012321472168 | 0.9312649164677804 | 0.9377240143369175 |
| EWoK | 297 | 147 | 149 | -0.08870351314544678 | 0.13406705856323242 | 0.891156462585034 | 0.9060402684563759 |
| Entity | 140 | 73 | 66 | -0.27417861856520176 | 0.18403319269418716 | 0.6438356164383562 | 0.7878787878787878 |
| GlobalPIQA | 5 | 3 | 2 | -0.03741035461425746 | 0.08658036716037587 | 1.0 | 1.0 |
| Supplement | 88 | 37 | 51 | -0.2864379937527701 | 0.35655971709638834 | 0.7567567567567568 | 0.6078431372549019 |

JSON: `experiments/archive/frontier_consolidation/data/anchor_margin_alpha_census_full/anchor_margin_alpha_census.json`
