# masked 1m pos512 token vs wwm profile — 1M masked-LM token masking vs whole-word masking, position-capacity repaired

Evidence JSON: `experiments/archive/initial_model_studies/data/masked_1m_grid_pos512_profile.json`

Both runs use the same official-corpus selected examples/order/source mix, same baseline tokenizer, same BERT-like masked model size, same optimizer schedule, same seeds, 1M whitespace-word exposure, `max_seq_length=256`, `max_position_embeddings=512`, and official `mlm` backend at `chck_1M`.

| mode | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR | loss_last |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| token | 54.34 | 50.40 | 48.91 | 17.21 | 50.33 | 9.91 | 3.99 | 6.76 |
| wwm | 56.17 | 50.80 | 49.00 | 18.06 | 50.20 | 9.82 | 3.76 | 6.66 |

## WWM minus token

| BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---:|---:|---:|---:|---:|---:|
| 1.83 | 0.40 | 0.09 | 0.85 | -0.13 | -0.09 | -0.23 |
