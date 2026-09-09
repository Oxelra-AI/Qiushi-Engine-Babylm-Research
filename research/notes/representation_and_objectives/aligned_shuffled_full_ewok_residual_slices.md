# aligned shuffled full ewok residual slices aligned-vs-shuffled full-EWoK residual slices

Status: **PASS**

Existing full ewok coupled turnover CSVs only; no model inference/training.

## domain

| group | n | acc aligned-shuffled count | stable aligned-shuffled | aligned-only | shuffled-only | interaction mean | interaction median |
|---|---:|---:|---:|---:|---:|---:|---:|
| material-dynamics | 770 | -67 | 123 | 102 | 169 | -0.1594 | -0.1592 |
| physical-interactions | 556 | 17 | -31 | 142 | 125 | 0.0085 | 0.0086 |
| agent-properties | 2210 | 11 | -15 | 502 | 491 | 0.0137 | -0.0008 |
| physical-dynamics | 120 | 9 | -25 | 30 | 21 | -0.0072 | 0.1471 |
| spatial-relations | 490 | 9 | -13 | 112 | 103 | 0.0331 | 0.0059 |
| quantitative-properties | 314 | -8 | 0 | 72 | 80 | 0.0387 | -0.0151 |
| social-properties | 328 | 7 | 25 | 72 | 65 | -0.0630 | -0.1725 |
| social-relations | 1548 | -5 | 40 | 345 | 350 | -0.0126 | 0.0157 |
| social-interactions | 294 | 5 | 6 | 85 | 80 | 0.1909 | -0.0151 |
| physical-relations | 818 | -5 | -1 | 155 | 160 | -0.0309 | 0.0106 |
| material-properties | 170 | -1 | 20 | 32 | 33 | -0.2002 | -0.1708 |

## ContextDiff

| group | n | acc aligned-shuffled count | stable aligned-shuffled | aligned-only | shuffled-only | interaction mean | interaction median |
|---|---:|---:|---:|---:|---:|---:|---:|
| material | 840 | -63 | 119 | 123 | 186 | -0.1534 | -0.1409 |
| other | 490 | 20 | -22 | 103 | 83 | 0.1404 | -0.0033 |
| variable swap | 2564 | 15 | 13 | 662 | 647 | 0.0011 | 0.0032 |
| negation | 190 | 8 | -19 | 42 | 34 | 0.0239 | 0.0209 |
| antonym | 3374 | -7 | 41 | 673 | 680 | -0.0217 | -0.0021 |
| number | 80 | -4 | 3 | 17 | 21 | 0.0030 | -0.0065 |
| game | 20 | 3 | -5 | 8 | 5 | 0.7552 | 0.1728 |
| active-passive | 30 | -1 | -4 | 9 | 10 | 0.2717 | 0.2092 |
| variable_swap | 30 | 1 | 3 | 12 | 11 | -0.0249 | -0.0381 |

## TargetDiff

| group | n | acc aligned-shuffled count | stable aligned-shuffled | aligned-only | shuffled-only | interaction mean | interaction median |
|---|---:|---:|---:|---:|---:|---:|---:|
| concept swap | 4746 | -44 | 142 | 977 | 1021 | -0.0270 | -0.0139 |
| variable swap | 2872 | 16 | -13 | 672 | 656 | 0.0086 | 0.0052 |

JSON: `experiments/archive/representation_and_objectives/data/aligned_shuffled_full_ewok_residual/aligned_shuffled_full_ewok_residual_slices.json`
