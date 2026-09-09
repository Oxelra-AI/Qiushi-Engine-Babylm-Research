# earlier analysis init-matched trainer CPU smoke

Status: `INITMATCHED_TRAINER_SMOKE_OK` returncode `0`.
Tiny run rows/words: `8` / `1190`.

## Init-matched build event

Target/ref vocab: `19609` / `16384`.
Copied same-shape tensors: `28`; skipped vocab-shaped tensors: `4`.
Same-shape exact numel fraction after copy: `1.0`.
Random-like exact tensors after copy: `11/11`.

## Tiny metrics

Word exposure `1190`, steps `2`, loss_first `9.93168830871582`, loss_last `9.954780578613281`, params `1386329`.

This is only a software smoke, not model-quality evidence.

Full JSON: `experiments/archive/frontier_consolidation/data/minfreq50_initmatched_smoke/initmatched_trainer_smoke.json`
