# masked 1m pos512 token vs wwm profile — Masked max-position repair probe

Evidence JSON: `experiments/archive/initial_model_studies/data/masked_position_probe.json`

A 10k token-masked BertForMaskedLM checkpoint was trained with `max_position_embeddings=512`, loaded from root and `chck_1M`, passed a 200-token forward test, and completed official fast Entity Tracking with backend `mlm`.

Entity Tracking fast (`mlm`): 15.64.
