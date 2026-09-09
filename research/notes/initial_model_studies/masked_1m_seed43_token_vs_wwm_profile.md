# masked 1m seed43 token vs wwm profile — Seed-43 1M masked-LM token masking vs whole-word masking

Evidence JSON: `experiments/archive/initial_model_studies/data/masked_1m_seed43_profile.json`

This reconstructs the completed seed-43 profiles from report files after the original runner failed only during final aggregation. The two runs use identical selected examples/order/source mix, baseline tokenizer, 8-layer/256-hidden BERT MLM, `max_seq_length=256`, `max_position_embeddings=512`, exact 1M word exposure, and official `mlm` backend.

| mode | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR | loss_last |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| token | 54.16 | 52.00 | 46.45 | 17.46 | 50.55 | 9.85 | 3.69 | 6.71 |
| wwm | 56.63 | 50.80 | 51.36 | 17.63 | 50.43 | 9.82 | 3.72 | 6.76 |

## WWM minus token

| BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---:|---:|---:|---:|---:|---:|
| 2.47 | -1.20 | 4.91 | 0.17 | -0.12 | -0.03 | 0.03 |
