# official40k accum training validation — official40k DeBERTa-v2 accumulation-run validation

Evidence JSON: `experiments/archive/initial_model_studies/data/official40k_accum_training_validation.json`

Run: `experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_official40k_wwm_seed42_100M_b128_acc2`

Validated as a complete 100M official-corpus run repaired with gradient accumulation.

| property | value |
|---|---:|
| parameters | 45826720 |
| embedding parameters | 19200000 |
| non-embedding parameters | 26626720 |
| vocab size | 40000 |
| exposure words | 100000000 |
| optimizer steps | 2442 |
| microbatch / accumulation | 128 / 2 |
| loss first | 10.7027 |
| loss last | 2.6364 |
| masked tokens/word | 0.205964 |
| checkpoints | 100 |
| kept tokens/word | 1.372627 |
| truncated example fraction | 0.182256 |
