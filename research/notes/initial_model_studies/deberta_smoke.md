# deberta smoke — DeBERTa-v2 WWM smoke

Evidence JSON: `experiments/archive/initial_model_studies/data/deberta_smoke_summary.json`

A primary parameter-matched DeBERTa-v2 WWM checkpoint trained for 10k official words, saved root/`chck_1M`, loaded with `AutoModelForMaskedLM`, passed 180-token forward tests, and completed official fast BLiMP and Entity with backend `mlm`.

BLiMP fast: 54.47; Entity fast: 16.61.
Params: 10733104; embedding params: 3932160; non-embedding params: 6800944.
