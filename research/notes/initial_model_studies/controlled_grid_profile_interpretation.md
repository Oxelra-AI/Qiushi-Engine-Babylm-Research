# controlled grid profile interpretation — Controlled pool/seed memory profile

Evidence JSON: `experiments/archive/initial_model_studies/data/controlled_grid_profile.json`

All runs use 1M word exposure, `lr_total_steps=98`, paired shared GPT-2 core initialization, identical consumed example order within each pool/seed dense-memory pair, official tokenizer/corpus, and `chck_1M` evaluation.

## Scores

| pool | seed | model | BLiMP | Supp | EWoK | Entity | COMPS | Reading eye | Reading SPR |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| pool1M | 42 | dense | 53.84 | 49.20 | 48.18 | 16.62 | 50.32 | 9.03 | 2.42 |
| pool1M | 42 | memory | 53.60 | 46.80 | 46.45 | 16.94 | 50.24 | 9.04 | 2.64 |
| pool1M | 43 | dense | 54.45 | 51.60 | 49.91 | 16.42 | 49.32 | 8.54 | 2.63 |
| pool1M | 43 | memory | 54.57 | 51.20 | 49.55 | 15.78 | 49.01 | 8.23 | 2.50 |
| pool10M | 42 | dense | 53.37 | 48.80 | 46.73 | 18.53 | 50.46 | 10.24 | 3.07 |
| pool10M | 42 | memory | 53.48 | 49.20 | 48.27 | 17.88 | 50.26 | 9.85 | 3.32 |
| pool10M | 43 | dense | 53.79 | 51.20 | 51.45 | 18.05 | 50.07 | 9.70 | 3.07 |
| pool10M | 43 | memory | 53.80 | 51.20 | 49.00 | 18.45 | 49.88 | 9.29 | 3.05 |

## Memory minus dense deltas

| pool | seed | BLiMP | Supp | EWoK | Entity | COMPS | Reading eye | Reading SPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pool1M | 42 | -0.24 | -2.40 | -1.73 | 0.32 | -0.08 | 0.01 | 0.22 |
| pool1M | 43 | 0.12 | -0.40 | -0.36 | -0.64 | -0.31 | -0.31 | -0.13 |
| pool10M | 42 | 0.11 | 0.40 | 1.54 | -0.65 | -0.20 | -0.39 | 0.25 |
| pool10M | 43 | 0.01 | 0.00 | -2.45 | 0.40 | -0.19 | -0.41 | -0.02 |

## Mean deltas and pool interaction

| metric | mean delta pool1M | mean delta pool10M | pool10M minus pool1M |
|---|---:|---:|---:|
| blimp_fast | -0.06 | 0.06 | 0.12 |
| supplement_fast | -1.40 | 0.20 | 1.60 |
| ewok_fast | -1.04 | -0.46 | 0.59 |
| entity_tracking_fast | -0.16 | -0.12 | 0.04 |
| comps | -0.20 | -0.20 | 0.00 |
| reading_eye_tracking | -0.15 | -0.40 | -0.25 |
| reading_self_paced | 0.04 | 0.12 | 0.07 |

## Immediate scientific reading

Entity memory effect: pool1M mean -0.16, pool10M mean -0.12, interaction 0.04.

EWoK memory effect: pool1M mean -1.04, pool10M mean -0.46, interaction 0.59.

Reading-eye memory effect: pool1M mean -0.15, pool10M mean -0.40, interaction -0.25.

Next work: train/profile `memory_nopersist_causal` in the cell whose memory advantage is scientifically most informative, so the surviving effect can be separated from extra per-token capacity.

