# eval code coordinate audit evaluation-coordinate readiness summary

Ready after patch: `True`

## What is identical without repair

- BLiMP: `True`
- Supplement: `True`
- Entity: `True`
- COMPS: `True`
- Reading: `True`
- SuperGLUE: `True`

## What is intentionally repaired or supplied

- EWoK: strictsmall tokenizer visibility audit current official 7,618-row re-evaluation (`True`).
- AoA: strictsmall tokenizer visibility audit min_context=0, 8,005 contexts/checkpoint (`True`).
- GlobalPIQA: changed block overlap ancestry current official current official `dl.py` generated files are byte-identical to inherited files (`True`).
- SuperGLUE: wrapper subprocess root patched to the pristine current checkout because INITIAL_MODEL_STUDIES `finetune/classifier_model.py` differs from the official file.

## Wrapper patch status

- seed43022: pristine STRICT override `True`, changed block overlap ancestry current official GlobalPIQA paths `True` — `experiments/archive/representation_and_objectives/training/scripts/full_eval_strictsmalltok_seed43022.py`
- seed43122: pristine STRICT override `True`, changed block overlap ancestry current official GlobalPIQA paths `True` — `experiments/archive/representation_and_objectives/training/scripts/full_eval_strictsmalltok_seed43122.py`

## Evaluation policy

Both completed corrected-tokenizer retrains should receive the same full official coordinate. The seven-column surface can order work but not replace SuperGLUE/AoA/pristine collation or remove the second seed.
