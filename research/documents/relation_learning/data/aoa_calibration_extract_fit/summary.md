# earlier analysis AoA calibration extraction and fitted-direction readout

This is a 30M same-seed calibration screen using the corrected aoa calibration design adapter-faithful arms. It measures arm-minus-control fitted model AoA against corpus-internal CHILDES-minus-whole enrichment z; exact credited-exposure replay is intentionally not required for this first direction readout.

## Arm integrity

| arm | params | words | steps | loss first | loss last | architecture | missing ckpts |
|---|---:|---:|---:|---:|---:|---|---|
| control | 35463008 | 30000000 | 759 | 9.813929557800293 | 3.3367316722869873 | AdapterDebertaV2ForMaskedLM | [] |
| schedule | 35463008 | 30000051 | 752 | 9.904044151306152 | 4.736428737640381 | AdapterDebertaV2ForMaskedLM | [] |
| enrichment | 35463008 | 30000000 | 759 | 9.814611434936523 | 2.972158670425415 | AdapterDebertaV2ForMaskedLM | [] |

## Raw AoA summaries

| arm | raw r | p | fitted words | clipped score |
|---|---:|---:|---:|---:|
| control | -0.04955432929298245 | 0.4071202904718168 | 282 | 0.0 |
| schedule | 0.20713466049583562 | 0.0092427515084342 | 157 | 0.20713466049583562 |
| enrichment | 0.023898778800867326 | 0.6953087440541987 | 271 | 0.0 |

## Arm-minus-control fitted AoA versus enrichment z

| arm | n | mean delta | r(delta,z) | p | slope delta~z | direction |
|---|---:|---:|---:|---:|---:|---|
| schedule | 144 | -0.07543951621867026 | -0.4199468726952622 | 1.6079090539725377e-07 | -0.4476011239058884 | useful |
| enrichment | 238 | -0.12569499011104746 | -0.2728063894557542 | 1.9747818432949825e-05 | -0.17642382456373282 | useful |

Full JSON: `experiments/archive/relation_learning/data/aoa_calibration_extract_fit/summary.json`
