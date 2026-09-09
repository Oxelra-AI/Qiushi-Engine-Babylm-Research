# earlier analysis AoA calibration extraction and fitted-direction readout

This is a 30M same-seed calibration screen using the corrected aoa calibration design adapter-faithful arms. It measures arm-minus-control fitted model AoA against corpus-internal CHILDES-minus-whole enrichment z; exact credited-exposure replay is intentionally not required for this first direction readout.

## Arm integrity

| arm | params | words | steps | loss first | loss last | architecture | missing ckpts |
|---|---:|---:|---:|---:|---:|---|---|
| control | 35463008 | 30000000 | 759 | 9.813929557800293 | 3.3367316722869873 | AdapterDebertaV2ForMaskedLM | [] |

## Raw AoA summaries

| arm | raw r | p | fitted words | clipped score |
|---|---:|---:|---:|---:|
| control | nan | nan | 0 | nan |

## Arm-minus-control fitted AoA versus enrichment z

| arm | n | mean delta | r(delta,z) | p | slope delta~z | direction |
|---|---:|---:|---:|---:|---:|---|

Full JSON: `experiments/archive/relation_learning/data/aoa_calibration_smoke/summary.json`
