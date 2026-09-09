# debertav2 wwm 1m profile — WWM fixed base: parameter-matched DeBERTa-v2 vs BERT at 1M

Evidence JSON: `experiments/archive/initial_model_studies/data/debertav2_wwm_1m_profile.json`

Only the architecture changes. The raw official-corpus examples/order/source mix, baseline 16k tokenizer, WWM objective, fixed length 256, optimizer schedule, word exposure, and seeds are matched to each seed's existing BERT-WWM baseline. The DeBERTa-v2 configuration is hidden 240, 8 layers, 6 heads, intermediate 960, relative attention p2c/c2p, total parameters 10,733,104 (+0.055% vs BERT).

| seed | arch | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR | loss_last | params | emb params | non-emb params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | DeBERTaV2 | 56.01 | 52.00 | 50.45 | 18.02 | 49.96 | 9.89 | 3.81 | 6.66 | 10733104 | 3932160 | 6800944 |
| 43 | DeBERTaV2 | 56.22 | 50.40 | 48.73 | 17.82 | 50.27 | 9.66 | 3.48 | 6.53 | 10733104 | 3932160 | 6800944 |

## DeBERTaV2 minus matched BERT-WWM baseline

| seed | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | -0.16 | 1.20 | 1.45 | -0.04 | -0.24 | 0.07 | 0.05 |
| 43 | -0.41 | -0.40 | -2.63 | 0.19 | -0.16 | -0.16 | -0.24 |

## Mean score delta

| BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---:|---:|---:|---:|---:|---:|
| -0.28 | 0.40 | -0.59 | 0.07 | -0.20 | -0.04 | -0.10 |
