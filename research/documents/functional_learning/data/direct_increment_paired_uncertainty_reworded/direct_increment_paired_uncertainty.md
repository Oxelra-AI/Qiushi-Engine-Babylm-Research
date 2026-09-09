# earlier analysis paired uncertainty: clean preservation versus exact `(M,S)`

This analysis uses completed official prediction files. It keeps the official component definitions and resamples paired validation items within the fixed official subtasks/tasks. It does not estimate private-training randomness or downstream fine-tuning seed variation.

## clean64_vs_exact_MS

Compared `clean_pres_seed62064_MSplusKL` minus `ms_acquisition_seed62064_MS`.

- zero/Reading payload sum delta: `0.235000` = `0.026111` Overall units
- SuperGLUE primary delta: `0.159869` = `0.017763` Overall units
- direct Overall delta from payload components: `0.043874`
- paired item/bootstrap direct Overall interval: median `0.044709`, 2.5--97.5% `-0.034541` to `0.129073`, fraction > 0 `0.8650`

### Component contributions

| component/task group | exact/payload delta | n units | 2.5% | 50% | 97.5% | fraction > 0 |
|---|---:|---:|---:|---:|---:|---:|
| BLiMP | 0.140000 | 67 | 0.063386 | 0.142862 | 0.218776 | 1.0000 |
| Supplement | 0.200000 | 5 | -0.126382 | 0.179147 | 0.541539 | 0.8767 |
| EWoK | -0.130000 | 11 | -0.591994 | -0.132191 | 0.328185 | 0.2900 |
| Entity | 0.010000 | 18 | -0.270676 | 0.008550 | 0.282137 | 0.5217 |
| COMPS | 0.010000 | 4 | -0.119412 | 0.013416 | 0.142244 | 0.5750 |
| GlobalPIQA | 0.000000 | 2 | 0.000000 | 0.000000 | 0.000000 | 0.0000 |
| Reading | 0.005000 | 1726 | -0.040361 | 0.006692 | 0.055219 | 0.6250 |

### SuperGLUE task contributions

| task | metric | exact clean-minus-reference | n | 2.5% | 50% | 97.5% | fraction > 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| boolq | accuracy | 0.000000 | 1635 | -0.795107 | 0.000000 | 0.795107 | 0.4717 |
| multirc | accuracy | 0.247525 | 2424 | -0.742574 | 0.247525 | 1.238655 | 0.6717 |
| rte | accuracy | 0.000000 | 139 | 0.000000 | 0.000000 | 0.000000 | 0.0000 |
| wsc | accuracy | 0.000000 | 52 | 0.000000 | 0.000000 | 0.000000 | 0.0000 |
| mrpc | f1 | 0.386589 | 204 | 0.000000 | 0.383794 | 1.241857 | 0.6567 |
| qqp | f1 | 0.158974 | 20215 | -0.272572 | 0.152285 | 0.561496 | 0.7617 |
| mnli | accuracy | 0.325998 | 4908 | -0.713631 | 0.325998 | 1.406377 | 0.7083 |

## clean65_vs_exact_MS_training_seed_replication_context

Compared `clean_pres_seed62065_MSplusKL` minus `ms_acquisition_seed62064_MS`.

- zero/Reading payload sum delta: `0.130000` = `0.014444` Overall units
- SuperGLUE primary delta: `0.132739` = `0.014749` Overall units
- direct Overall delta from payload components: `0.029193`
- paired item/bootstrap direct Overall interval: median `0.032571`, 2.5--97.5% `-0.055681` to `0.122465`, fraction > 0 `0.7500`

### Component contributions

| component/task group | exact/payload delta | n units | 2.5% | 50% | 97.5% | fraction > 0 |
|---|---:|---:|---:|---:|---:|---:|
| BLiMP | 0.100000 | 67 | 0.023519 | 0.103909 | 0.177905 | 0.9917 |
| Supplement | 0.210000 | 5 | -0.089221 | 0.202326 | 0.615747 | 0.8933 |
| EWoK | -0.230000 | 11 | -0.704447 | -0.229003 | 0.293120 | 0.1950 |
| Entity | 0.060000 | 18 | -0.221882 | 0.062953 | 0.355635 | 0.6600 |
| COMPS | -0.010000 | 4 | -0.143705 | -0.010328 | 0.109947 | 0.4317 |
| GlobalPIQA | 0.000000 | 2 | 0.000000 | 0.000000 | 0.000000 | 0.0000 |
| Reading | 0.000000 | 1726 | -0.048911 | 0.003142 | 0.054259 | 0.5300 |

### SuperGLUE task contributions

| task | metric | exact clean-minus-reference | n | 2.5% | 50% | 97.5% | fraction > 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| boolq | accuracy | -0.183486 | 1635 | -0.917431 | -0.183486 | 0.611621 | 0.3050 |
| multirc | accuracy | 0.288779 | 2424 | -0.661097 | 0.288779 | 1.362417 | 0.7183 |
| rte | accuracy | 0.000000 | 139 | -2.877698 | 0.000000 | 2.877698 | 0.4300 |
| wsc | accuracy | 0.000000 | 52 | 0.000000 | 0.000000 | 0.000000 | 0.0000 |
| mrpc | f1 | 0.386589 | 204 | 0.000000 | 0.381170 | 1.254924 | 0.6667 |
| qqp | f1 | 0.131665 | 20215 | -0.265819 | 0.134165 | 0.554866 | 0.7500 |
| mnli | accuracy | 0.305623 | 4908 | -0.753871 | 0.285249 | 1.263753 | 0.6800 |

## clean64_vs_coherent86_policy_gain

Compared `clean_pres_seed62064_MSplusKL` minus `coherent86`.

- zero/Reading payload sum delta: `1.900000` = `0.211111` Overall units
- SuperGLUE primary delta: `0.101999` = `0.011333` Overall units
- direct Overall delta from payload components: `0.222444`
- paired item/bootstrap direct Overall interval: median `0.214315`, 2.5--97.5% `-0.047047` to `0.537030`, fraction > 0 `0.9250`

### Component contributions

| component/task group | exact/payload delta | n units | 2.5% | 50% | 97.5% | fraction > 0 |
|---|---:|---:|---:|---:|---:|---:|
| BLiMP | -0.250000 | 67 | -0.364760 | -0.252564 | -0.145085 | 0.0000 |
| Supplement | -0.360000 | 5 | -1.290879 | -0.316044 | 0.335278 | 0.2050 |
| EWoK | -0.200000 | 11 | -0.829530 | -0.176229 | 0.402446 | 0.2817 |
| Entity | 1.080000 | 18 | 0.525421 | 1.075679 | 1.640519 | 1.0000 |
| COMPS | 0.110000 | 4 | -0.075726 | 0.116401 | 0.310191 | 0.8783 |
| GlobalPIQA | 1.485000 | 2 | -0.941748 | 1.485437 | 3.942112 | 0.9050 |
| Reading | 0.035000 | 1726 | -0.109100 | 0.030267 | 0.161438 | 0.6700 |

### SuperGLUE task contributions

| task | metric | exact clean-minus-reference | n | 2.5% | 50% | 97.5% | fraction > 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| boolq | accuracy | 0.672783 | 1635 | -1.406728 | 0.672783 | 2.996942 | 0.6983 |
| multirc | accuracy | -0.123762 | 2424 | -1.362417 | -0.165017 | 1.196370 | 0.4083 |
| rte | accuracy | -0.719424 | 139 | -3.597122 | -0.719424 | 2.158273 | 0.2483 |
| wsc | accuracy | 0.000000 | 52 | -5.769231 | 0.000000 | 5.769231 | 0.3217 |
| mrpc | f1 | 0.689655 | 204 | 0.000000 | 0.673408 | 1.718995 | 0.8617 |
| qqp | f1 | 0.031743 | 20215 | -0.182271 | 0.038871 | 0.226040 | 0.6183 |
| mnli | accuracy | 0.162999 | 4908 | -0.285249 | 0.162999 | 0.570497 | 0.7450 |

## Scientific reading

The completed files establish a higher fixed-coordinate score for clean seed62064 over coherent86, but this paired resampling does not establish broad superiority beyond the fixed validation pools: the clean64-vs-coherent86 interval also crosses zero under this conditional item perturbation. The preservation-specific increment over exact `(M,S)` is small. The direct clean64-minus-`(M,S)` score is carried by a narrow positive zero/Reading sum plus a SuperGLUE difference of about 0.160 component points. The paired item resampling asks whether those finite validation files make the sign fragile; it should be read together with the still-needed same-downstream-seed SuperGLUE comparison, because item resampling cannot stand in for fine-tuning-seed variation. Endpoint selection remains separate: seed62064 is the higher complete fixed-coordinate endpoint, while seed62065 is the training-seed replication.
