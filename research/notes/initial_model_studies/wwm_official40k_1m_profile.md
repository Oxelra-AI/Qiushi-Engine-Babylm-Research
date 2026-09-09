# wwm official40k 1m profile — WWM fixed base: official-corpus 40k tokenizer vs baseline 16k at 1M

Evidence JSON: `experiments/archive/initial_model_studies/data/wwm_official40k_1m_profile.json`

Only the tokenizer representation changes. The raw official-corpus examples/order/source mix, WWM objective, fixed length 256, model depth/width/heads, optimizer schedule, word exposure, and seeds are matched to each seed's existing WWM 16k baseline. Vocabulary size changes embedding capacity and tokenization/WWM density; those linked changes are recorded below.

| seed | tok | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR | loss_last | params | emb params | mask tok/word |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 40k | 55.93 | 46.40 | 49.55 | 17.97 | 49.85 | 1.44 | 0.18 | 6.76 | 16796480 | 10240000 | 0.2058 |
| 43 | 40k | 55.92 | 48.80 | 54.36 | 17.60 | 49.97 | 1.57 | 0.18 | 6.92 | 16796480 | 10240000 | 0.2052 |

## 40k minus matched 16k WWM baseline

| seed | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | -0.24 | -4.40 | 0.55 | -0.09 | -0.35 | -8.38 | -3.58 |
| 43 | -0.71 | -2.00 | 3.00 | -0.03 | -0.46 | -8.25 | -3.54 |

## Mean score delta

| BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---:|---:|---:|---:|---:|---:|
| -0.47 | -3.20 | 1.77 | -0.06 | -0.41 | -8.31 | -3.56 |

## Coupled tokenizer changes

| seed | 16k tokens/word | 40k tokens/word | % change | 16k expected WWM pred tokens | 40k expected WWM pred tokens | % change | 16k trunc frac | 40k trunc frac |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1.4939 | 1.3995 | -6.32% | 216981.1 | 206255.1 | -4.94% | 0.2965 | 0.1899 |
| 43 | 1.4904 | 1.3968 | -6.28% | 216651.8 | 206025.0 | -4.91% | 0.2920 | 0.1846 |
