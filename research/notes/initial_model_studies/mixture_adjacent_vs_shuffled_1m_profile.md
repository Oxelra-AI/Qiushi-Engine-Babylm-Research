# mixture adjacent vs shuffled 1m profile — 50/50 official+rewrite mixture: adjacent vs shuffled

Evidence JSON: `experiments/archive/initial_model_studies/data/mixture_adjacent_vs_shuffled_1m_profile.json`

Both arms per seed share identical 500k official examples, identical 500k rewrite source/target multisets, identical slot order, total words, kept tokens, WWM groups, and expected mask opportunity. Only rewrite target adjacency differs. Base: BERT-WWM, baseline 16k tokenizer, fixed 256, AdamW, paired seeds 42/43.

| seed | arm | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | mixture_adjacent | 54.51 | 45.60 | 49.18 | 18.64 | 50.36 | 7.00 | 2.77 |
| 42 | mixture_shuffled | 54.63 | 48.80 | 51.00 | 18.41 | 49.80 | 6.94 | 2.88 |
| 43 | mixture_adjacent | 53.84 | 46.40 | 49.09 | 17.39 | 50.15 | 7.60 | 2.31 |
| 43 | mixture_shuffled | 54.53 | 46.40 | 50.55 | 16.89 | 50.37 | 7.75 | 2.36 |

## mixture-adjacent minus mixture-shuffled

| seed | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | -0.12 | -3.20 | -1.82 | 0.23 | 0.56 | 0.06 | -0.11 |
| 43 | -0.69 | 0.00 | -1.46 | 0.50 | -0.22 | -0.15 | -0.05 |

## mean delta

| BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---:|---:|---:|---:|---:|---:|
| -0.41 | -1.60 | -1.64 | 0.36 | 0.17 | -0.04 | -0.08 |
