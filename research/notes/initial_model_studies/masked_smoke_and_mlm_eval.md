# masked smoke and mlm eval — Masked-LM smoke and official `mlm` evaluator check

Evidence JSON: `experiments/archive/initial_model_studies/data/masked_smoke_summary.json`

Two 10k-word masked-LM smokes were trained with the new standalone `babylm_masked_train.py`: token masking and whole-word masking. Both load with `AutoModelForMaskedLM` at root and `chck_1M`; both were evaluated on official fast BLiMP using backend `mlm`.

| mask mode | run id | params | loss first | loss last | BLiMP fast (`mlm`) |
|---|---|---:|---:|---:|---:|
| token | `babylm_masked_token_smoke10k` | 2544768 | 9.6864 | 9.0755 | 56.24 |
| wwm | `babylm_masked_wwm_smoke10k` | 2544768 | 9.7196 | 9.2664 | 54.81 |

Same first 12 example IDs: `True`. Same source-word mix: `True`.
