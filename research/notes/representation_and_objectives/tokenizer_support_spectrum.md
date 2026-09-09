# tokenizer support spectrum tokenizer support spectrum

This CPU-only evidence asset was built while the legal-40k 100M trainings were running. It does not train or evaluate a language model and does not use official evaluation text to train tokenizers. Support-floored tokenizers are learned only from the exact allowed 10M compact_view_reinvest pool.

## Pool support summary

| tokenizer | vocab | tok/word | used non-special | median count | p10 count | vocab<50 | mass<50 | vocab<100 | mass<100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legal_a01_16k | 16384 | 1.4669 | 16266 | 140.0 | 64.0 | 0.080 | 0.0020 | 0.341 | 0.0249 |
| legal_byte_bpe_24k | 24576 | 1.4281 | 24366 | 73.0 | 34.0 | 0.324 | 0.0183 | 0.609 | 0.0523 |
| legal_byte_bpe_32k | 32768 | 1.4066 | 32350 | 46.0 | 20.0 | 0.531 | 0.0336 | 0.725 | 0.0646 |
| legal_byte_bpe_40k | 40000 | 1.3943 | 39320 | 33.0 | 14.0 | 0.632 | 0.0413 | 0.785 | 0.0713 |
| legal_byte_bpe_40k_minfreq10 | 40000 | 1.3943 | 39320 | 33.0 | 14.0 | 0.632 | 0.0413 | 0.785 | 0.0713 |
| legal_byte_bpe_40k_minfreq25 | 29529 | 1.4139 | 29185 | 54.0 | 25.0 | 0.466 | 0.0286 | 0.688 | 0.0605 |
| legal_byte_bpe_40k_minfreq50 | 19609 | 1.4484 | 19454 | 105.0 | 49.0 | 0.100 | 0.0031 | 0.477 | 0.0383 |

## New-token support relative to legal16k

| tokenizer | new vocab | new used | new mass | median new count | p10 new count | new vocab<50 | new mass<50 | new vocab<100 | new mass<100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legal_byte_bpe_24k | 8192 | 8144 | 0.0253 | 44.0 | 34.0 | 0.656 | 0.0140 | 1.000 | 0.0253 |
| legal_byte_bpe_32k | 16384 | 16180 | 0.0375 | 30.0 | 20.0 | 0.855 | 0.0280 | 1.000 | 0.0375 |
| legal_byte_bpe_40k | 23616 | 23182 | 0.0441 | 23.0 | 13.1 | 0.907 | 0.0352 | 1.000 | 0.0441 |
| legal_byte_bpe_40k_minfreq10 | 23616 | 23182 | 0.0441 | 23.0 | 13.1 | 0.907 | 0.0352 | 1.000 | 0.0441 |
| legal_byte_bpe_40k_minfreq25 | 13145 | 13000 | 0.0335 | 35.0 | 25.0 | 0.809 | 0.0234 | 1.000 | 0.0335 |
| legal_byte_bpe_40k_minfreq50 | 3225 | 3210 | 0.0124 | 57.0 | 50.0 | 0.031 | 0.0000 | 1.000 | 0.0124 |

## Evaluation-family exposure to low-support training tokens

| tokenizer | family | eval tokens | mean pool count | p10 | p50 | frac<50 | frac<100 | frac<200 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| legal_a01_16k | Supplement | 172904 | 83476.7 | 224.0 | 5356.0 | 0.0073 | 0.0322 | 0.0911 |
| legal_a01_16k | EWoK | 286116 | 135683.8 | 133.0 | 5858.0 | 0.0012 | 0.0418 | 0.1250 |
| legal_a01_16k | Entity | 9281512 | 152129.9 | 70.0 | 31217.0 | 0.0000 | 0.1135 | 0.1606 |
| legal_a01_16k | COMPS | 3294276 | 95229.6 | 125.0 | 3201.0 | 0.0017 | 0.0681 | 0.1516 |
| legal_a01_16k | GlobalPIQA | 18453 | 63852.8 | 157.0 | 4621.0 | 0.0212 | 0.0507 | 0.1268 |
| legal_a01_16k | SuperGLUE | 15834278 | 83972.7 | 166.0 | 5421.0 | 0.0126 | 0.0501 | 0.1194 |
| legal_a01_16k | BLiMP | 1228503 | 90214.4 | 133.0 | 3406.0 | 0.0034 | 0.0560 | 0.1564 |
| legal_a01_16k | Reading | 22223 | 122301.8 | 348.0 | 18894.0 | 0.0004 | 0.0126 | 0.0504 |
| legal_byte_bpe_24k | Supplement | 167559 | 85702.1 | 158.0 | 5759.0 | 0.0267 | 0.0658 | 0.1165 |
| legal_byte_bpe_24k | EWoK | 270342 | 142997.4 | 91.0 | 5552.0 | 0.0234 | 0.1090 | 0.1649 |
| legal_byte_bpe_24k | Entity | 9267436 | 152185.8 | 70.0 | 31217.0 | 0.0015 | 0.1151 | 0.1637 |
| legal_byte_bpe_24k | COMPS | 3158439 | 98492.2 | 82.0 | 2154.0 | 0.0579 | 0.1218 | 0.1953 |
| legal_byte_bpe_24k | GlobalPIQA | 18016 | 64949.3 | 124.0 | 3618.0 | 0.0354 | 0.0772 | 0.1521 |
| legal_byte_bpe_24k | SuperGLUE | 15179172 | 86927.8 | 104.0 | 4815.0 | 0.0408 | 0.0968 | 0.1601 |
| legal_byte_bpe_24k | BLiMP | 1152518 | 95489.5 | 79.0 | 3265.0 | 0.0417 | 0.1222 | 0.2033 |
| legal_byte_bpe_24k | Reading | 21943 | 123614.8 | 329.0 | 18902.0 | 0.0067 | 0.0252 | 0.0584 |
| legal_byte_bpe_32k | Supplement | 164905 | 86965.7 | 140.0 | 6045.0 | 0.0425 | 0.0769 | 0.1247 |
| legal_byte_bpe_32k | EWoK | 262242 | 147099.0 | 67.0 | 5075.0 | 0.0773 | 0.1445 | 0.2193 |
| legal_byte_bpe_32k | Entity | 9267436 | 152102.2 | 70.0 | 31217.0 | 0.0015 | 0.1151 | 0.1652 |
| legal_byte_bpe_32k | COMPS | 3058190 | 101174.5 | 53.0 | 1328.0 | 0.0921 | 0.1516 | 0.2138 |
| legal_byte_bpe_32k | GlobalPIQA | 17241 | 64659.8 | 74.0 | 2896.0 | 0.0823 | 0.1447 | 0.2014 |
| legal_byte_bpe_32k | SuperGLUE | 14821127 | 88749.8 | 77.0 | 4974.0 | 0.0676 | 0.1193 | 0.1800 |
| legal_byte_bpe_32k | BLiMP | 1119658 | 98068.5 | 64.0 | 3209.0 | 0.0748 | 0.1489 | 0.2225 |
| legal_byte_bpe_32k | Reading | 21824 | 124178.8 | 324.0 | 19203.0 | 0.0122 | 0.0290 | 0.0612 |
| legal_byte_bpe_40k | Supplement | 163350 | 87697.7 | 131.0 | 6052.0 | 0.0495 | 0.0822 | 0.1284 |
| legal_byte_bpe_40k | EWoK | 260780 | 147828.8 | 56.0 | 5075.0 | 0.0878 | 0.1495 | 0.2230 |
| legal_byte_bpe_40k | Entity | 9267436 | 152057.8 | 70.0 | 31217.0 | 0.0031 | 0.1151 | 0.1652 |
| legal_byte_bpe_40k | COMPS | 2987213 | 103416.4 | 37.0 | 1410.0 | 0.1292 | 0.1798 | 0.2283 |
| legal_byte_bpe_40k | GlobalPIQA | 17043 | 65313.7 | 54.0 | 2514.0 | 0.0975 | 0.1609 | 0.2092 |
| legal_byte_bpe_40k | SuperGLUE | 14620608 | 89819.7 | 66.0 | 5265.0 | 0.0816 | 0.1314 | 0.1896 |
| legal_byte_bpe_40k | BLiMP | 1103684 | 99404.6 | 59.0 | 3354.0 | 0.0851 | 0.1574 | 0.2278 |
| legal_byte_bpe_40k | Reading | 21817 | 124175.8 | 318.0 | 19203.0 | 0.0126 | 0.0294 | 0.0612 |
| legal_byte_bpe_40k_minfreq10 | Supplement | 163350 | 87697.7 | 131.0 | 6052.0 | 0.0495 | 0.0822 | 0.1284 |
| legal_byte_bpe_40k_minfreq10 | EWoK | 260780 | 147828.8 | 56.0 | 5075.0 | 0.0878 | 0.1495 | 0.2230 |
| legal_byte_bpe_40k_minfreq10 | Entity | 9267436 | 152057.8 | 70.0 | 31217.0 | 0.0031 | 0.1151 | 0.1652 |
| legal_byte_bpe_40k_minfreq10 | COMPS | 2987213 | 103416.4 | 37.0 | 1410.0 | 0.1292 | 0.1798 | 0.2283 |
| legal_byte_bpe_40k_minfreq10 | GlobalPIQA | 17043 | 65313.7 | 54.0 | 2514.0 | 0.0975 | 0.1609 | 0.2092 |
| legal_byte_bpe_40k_minfreq10 | SuperGLUE | 14620608 | 89819.7 | 66.0 | 5265.0 | 0.0816 | 0.1314 | 0.1896 |
| legal_byte_bpe_40k_minfreq10 | BLiMP | 1103684 | 99404.6 | 59.0 | 3354.0 | 0.0851 | 0.1574 | 0.2278 |
| legal_byte_bpe_40k_minfreq10 | Reading | 21817 | 124175.8 | 318.0 | 19203.0 | 0.0126 | 0.0294 | 0.0612 |
| legal_byte_bpe_40k_minfreq25 | Supplement | 165773 | 86545.4 | 145.0 | 6015.0 | 0.0382 | 0.0740 | 0.1219 |
| legal_byte_bpe_40k_minfreq25 | EWoK | 263296 | 146553.4 | 67.0 | 4818.0 | 0.0745 | 0.1418 | 0.2144 |
| legal_byte_bpe_40k_minfreq25 | Entity | 9267436 | 152126.6 | 70.0 | 31217.0 | 0.0015 | 0.1151 | 0.1652 |
| legal_byte_bpe_40k_minfreq25 | COMPS | 3082202 | 100590.0 | 62.0 | 1458.0 | 0.0832 | 0.1388 | 0.2084 |
| legal_byte_bpe_40k_minfreq25 | GlobalPIQA | 17749 | 62984.1 | 86.0 | 2873.0 | 0.0516 | 0.1132 | 0.1685 |
| legal_byte_bpe_40k_minfreq25 | SuperGLUE | 14943607 | 88097.6 | 85.0 | 4818.0 | 0.0584 | 0.1114 | 0.1730 |
| legal_byte_bpe_40k_minfreq25 | BLiMP | 1127645 | 97423.4 | 69.0 | 3265.0 | 0.0654 | 0.1404 | 0.2168 |
| legal_byte_bpe_40k_minfreq25 | Reading | 21841 | 124134.3 | 324.0 | 19227.0 | 0.0119 | 0.0289 | 0.0612 |
| legal_byte_bpe_40k_minfreq50 | Supplement | 170437 | 84525.9 | 188.0 | 5356.0 | 0.0091 | 0.0469 | 0.1045 |
| legal_byte_bpe_40k_minfreq50 | EWoK | 276368 | 140140.9 | 121.0 | 6984.0 | 0.0016 | 0.0799 | 0.1441 |
| legal_byte_bpe_40k_minfreq50 | Entity | 9281512 | 152042.8 | 70.0 | 31217.0 | 0.0000 | 0.1135 | 0.1619 |
| legal_byte_bpe_40k_minfreq50 | COMPS | 3238112 | 96441.0 | 113.0 | 2609.0 | 0.0210 | 0.0898 | 0.1722 |
| legal_byte_bpe_40k_minfreq50 | GlobalPIQA | 18234 | 64407.8 | 145.0 | 4239.0 | 0.0225 | 0.0632 | 0.1381 |
| legal_byte_bpe_40k_minfreq50 | SuperGLUE | 15525166 | 85360.1 | 134.0 | 4976.0 | 0.0151 | 0.0724 | 0.1387 |
| legal_byte_bpe_40k_minfreq50 | BLiMP | 1192442 | 92607.3 | 107.0 | 3103.0 | 0.0061 | 0.0875 | 0.1819 |
| legal_byte_bpe_40k_minfreq50 | Reading | 22081 | 122968.4 | 346.0 | 18902.0 | 0.0004 | 0.0192 | 0.0538 |

## Scientific use

- This file does not settle the 40k route. The running legal-40k models still need full pristine official evaluation.
- If legal 40k improves Supplement/EWoK but loses GlobalPIQA/Entity/COMPS, these support spectra can test whether rare, low-support new units are a plausible failure mode.
- If legal 40k fails broadly, the spectra help decide whether a support-floored tokenizer is scientifically distinct enough to justify a later experiment, or whether the next route should move to masking/sequence curriculum or depth-over-width architecture.
- Evaluation-family support columns are interpretation-only; tokenizer learning used only the allowed 10M pool.

JSON: `experiments/archive/representation_and_objectives/data/tokenizer_support_spectrum/tokenizer_support_spectrum.json`
CSV tables: `pool_support_summary.csv`, `new_vs_legal16_support_summary.csv`, `eval_family_low_support.csv`.
