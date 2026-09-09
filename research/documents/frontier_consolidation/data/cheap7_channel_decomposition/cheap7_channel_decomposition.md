# earlier analysis cheap7 channel decomposition and low-count bootstrap

Status: **COMPLETE**  ·  anchor `chck82_anchor`  ·  n_boot 2000

Payload-only (no model inference). Discrete columns reconstructed via the
validated official_score; Reading carried from payload.

## cheap7 vs cheap7-without-GlobalPIQA

| arm | cheap7 | Δcheap7 vs anchor | cheap6(no GP) | Δcheap6(no GP) vs anchor |
|---|---:|---:|---:|---:|
| coherent86_alpha0p75 | 44.181208 | +0.221758 | 45.117558 | +0.094478 |
| coherent86_alpha0p5 | 44.177065 | +0.217615 | 45.031818 | +0.008739 |
| coherent86_alpha1 | 44.106866 | +0.147417 | 45.114160 | +0.091080 |
| shuffled86_private | 44.014248 | +0.054798 | 44.756105 | -0.266975 |
| chck82_anchor | 43.959450 | +0.000000 | 45.023080 | +0.000000 |
| ordinary86_backbone | 43.771493 | -0.187957 | 45.044088 | +0.021008 |
| spanbreak86 | 43.124432 | -0.835018 | 44.051320 | -0.971760 |

Ordering by cheap7: ['coherent86_alpha0p75', 'coherent86_alpha0p5', 'coherent86_alpha1', 'shuffled86_private', 'chck82_anchor', 'ordinary86_backbone', 'spanbreak86']
Ordering by cheap6 (no GlobalPIQA): ['coherent86_alpha0p75', 'coherent86_alpha1', 'ordinary86_backbone', 'coherent86_alpha0p5', 'chck82_anchor', 'shuffled86_private', 'spanbreak86']

## cheap7-delta decomposition vs anchor (Δcol / 7)

### ordinary86_backbone  (Δcheap7 -0.187957)

| column | Δcol | contrib to cheap7 |
|---|---:|---:|
| BLiMP | -0.0178 | -0.002538 |
| Supplement | -0.2621 | -0.037448 |
| EWoK | +0.0849 | +0.012126 |
| Entity | +0.2685 | +0.038357 |
| COMPS | +0.1013 | +0.014468 |
| GlobalPIQA | -1.4417 | -0.205964 |
| Reading | -0.0487 | -0.006959 |

Positive-contribution sum +0.064951; top positive column `Entity` = 0.591 of positive; GlobalPIQA share of positive = 0.0.

### shuffled86_private  (Δcheap7 +0.054798)

| column | Δcol | contrib to cheap7 |
|---|---:|---:|
| BLiMP | +0.7451 | +0.106440 |
| Supplement | -3.1264 | -0.446634 |
| EWoK | +1.3848 | +0.197831 |
| Entity | -1.5424 | -0.220340 |
| COMPS | +0.4608 | +0.065827 |
| GlobalPIQA | +1.9854 | +0.283634 |
| Reading | +0.4763 | +0.068041 |

Positive-contribution sum +0.721772; top positive column `GlobalPIQA` = 0.393 of positive; GlobalPIQA share of positive = 0.3929685045377405.

### coherent86_alpha1  (Δcheap7 +0.147417)

| column | Δcol | contrib to cheap7 |
|---|---:|---:|
| BLiMP | +0.0380 | +0.005429 |
| Supplement | +0.7081 | +0.101153 |
| EWoK | -0.1494 | -0.021347 |
| Entity | +0.1259 | +0.017986 |
| COMPS | -0.1974 | -0.028195 |
| GlobalPIQA | +0.4854 | +0.069348 |
| Reading | +0.0213 | +0.003041 |

Positive-contribution sum +0.196958; top positive column `Supplement` = 0.514 of positive; GlobalPIQA share of positive = 0.35209600873108143.

### spanbreak86  (Δcheap7 -0.835018)

| column | Δcol | contrib to cheap7 |
|---|---:|---:|
| BLiMP | -0.3012 | -0.043025 |
| Supplement | -0.5731 | -0.081872 |
| EWoK | -0.6833 | -0.097612 |
| Entity | -4.0730 | -0.581861 |
| COMPS | -0.1463 | -0.020893 |
| GlobalPIQA | -0.0146 | -0.002080 |
| Reading | -0.0537 | -0.007673 |

Positive-contribution sum +0.000000; no positive column contribution.

### coherent86_alpha0p5  (Δcheap7 +0.217615)

| column | Δcol | contrib to cheap7 |
|---|---:|---:|
| BLiMP | +0.0534 | +0.007632 |
| Supplement | +0.2081 | +0.029729 |
| EWoK | -0.0593 | -0.008477 |
| Entity | -0.0838 | -0.011978 |
| COMPS | -0.0822 | -0.011743 |
| GlobalPIQA | +1.4709 | +0.210125 |
| Reading | +0.0163 | +0.002327 |

Positive-contribution sum +0.249813; top positive column `GlobalPIQA` = 0.841 of positive; GlobalPIQA share of positive = 0.8411295967217212.

### coherent86_alpha0p75  (Δcheap7 +0.221758)

| column | Δcol | contrib to cheap7 |
|---|---:|---:|
| BLiMP | +0.0247 | +0.003527 |
| Supplement | +0.6976 | +0.099651 |
| EWoK | -0.0358 | -0.005120 |
| Entity | +0.0082 | +0.001168 |
| COMPS | -0.1440 | -0.020571 |
| GlobalPIQA | +0.9854 | +0.140777 |
| Reading | +0.0163 | +0.002327 |

Positive-contribution sum +0.247449; top positive column `GlobalPIQA` = 0.569 of positive; GlobalPIQA share of positive = 0.5689116491568781.

## Low-count column bootstrap (candidate − anchor)

### ordinary86_backbone

| column | point Δ | boot mean | boot std | CI95 | P(Δ<=0) |
|---|---:|---:|---:|---|---:|
| GlobalPIQA | -1.4417 | -1.4908 | 1.8867 | [-5.3252, +2.0291] | 0.7690 |
| EWoK | +0.0849 | +0.0878 | 0.2882 | [-0.4392, +0.6627] | 0.3915 |
| Entity | +0.2685 | +0.2695 | 0.2850 | [-0.2864, +0.8256] | 0.1755 |
| Supplement | -0.2621 | -0.2541 | 0.2281 | [-0.7273, +0.1034] | 0.8680 |

### shuffled86_private

| column | point Δ | boot mean | boot std | CI95 | P(Δ<=0) |
|---|---:|---:|---:|---|---:|
| GlobalPIQA | +1.9854 | +2.0764 | 2.3869 | [-2.8689, +6.8981] | 0.1890 |
| EWoK | +1.3848 | +1.3826 | 0.9595 | [-0.1978, +3.5134] | 0.0480 |
| Entity | -1.5424 | -1.5430 | 0.5655 | [-2.7263, -0.5572] | 1.0000 |
| Supplement | -3.1264 | -3.1658 | 1.7514 | [-7.0344, -0.2122] | 0.9920 |

### coherent86_alpha1

| column | point Δ | boot mean | boot std | CI95 | P(Δ<=0) |
|---|---:|---:|---:|---|---:|
| GlobalPIQA | +0.4854 | +0.4991 | 1.0813 | [-1.5000, +2.9272] | 0.3760 |
| EWoK | -0.1494 | -0.1510 | 0.2590 | [-0.6636, +0.3646] | 0.7145 |
| Entity | +0.1259 | +0.1151 | 0.1967 | [-0.2601, +0.5045] | 0.2830 |
| Supplement | +0.7081 | +0.7066 | 0.5940 | [-0.3187, +2.0295] | 0.1315 |

### spanbreak86

| column | point Δ | boot mean | boot std | CI95 | P(Δ<=0) |
|---|---:|---:|---:|---|---:|
| GlobalPIQA | -0.0146 | +0.0346 | 1.5581 | [-2.9709, +2.9709] | 0.5375 |
| EWoK | -0.6833 | -0.7044 | 0.7048 | [-2.2918, +0.4523] | 0.8325 |
| Entity | -4.0730 | -4.1110 | 2.0569 | [-8.0964, -0.0056] | 0.9760 |
| Supplement | -0.5731 | -0.5784 | 0.6290 | [-1.7348, +0.6582] | 0.8175 |

### coherent86_alpha0p5

| column | point Δ | boot mean | boot std | CI95 | P(Δ<=0) |
|---|---:|---:|---:|---|---:|
| GlobalPIQA | +1.4709 | +1.4500 | 0.8464 | [+0.0000, +3.4272] | 0.0490 |
| EWoK | -0.0593 | -0.0518 | 0.1749 | [-0.3823, +0.2925] | 0.6165 |
| Entity | -0.0838 | -0.0878 | 0.1136 | [-0.3128, +0.1367] | 0.7790 |
| Supplement | +0.2081 | +0.2082 | 0.3107 | [-0.2182, +0.8900] | 0.3285 |

### coherent86_alpha0p75

| column | point Δ | boot mean | boot std | CI95 | P(Δ<=0) |
|---|---:|---:|---:|---|---:|
| GlobalPIQA | +0.9854 | +0.9806 | 0.9636 | [-0.9709, +2.9563] | 0.1595 |
| EWoK | -0.0358 | -0.0451 | 0.1959 | [-0.4153, +0.3345] | 0.5915 |
| Entity | +0.0082 | +0.0077 | 0.1614 | [-0.3138, +0.3225] | 0.4770 |
| Supplement | +0.6976 | +0.6863 | 0.5559 | [-0.1211, +1.9238] | 0.0915 |

JSON: `experiments/archive/frontier_consolidation/data/cheap7_channel_decomposition/cheap7_channel_decomposition.json`
