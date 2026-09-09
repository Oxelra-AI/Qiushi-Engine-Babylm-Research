# context geometry and v5 design special-token geometry KL bins

Rows per form: 48.

| endpoint | scale | coherent+special KL | short+special KL | coherent no-special KL | ratio | max short token KL |
|---|---:|---:|---:|---:|---:|---:|
| chck82_slow_scale1p75 | -1.0 | -0.00000000 | 0.00000000 | -0.00000000 | -1.952 | 0.0000 |
| coherent86_alpha075 | 0.75 | 0.00079373 | 0.00190361 | 0.00074675 | 2.398 | 0.0486 |
| dense64_u0080 | 0.75 | 0.00377517 | 0.01520096 | 0.00294882 | 4.027 | 2.2495 |
| dense65_u0080 | 0.75 | 0.00382750 | 0.01556958 | 0.00298302 | 4.068 | 2.2662 |
| clean_pres64_u0080 | 0.75 | 0.00223559 | 0.00917638 | 0.00177841 | 4.105 | 1.5310 |
| clean_pres65_u0080 | 0.75 | 0.00218533 | 0.00904220 | 0.00174927 | 4.138 | 1.5210 |

## Largest KL tokens

| endpoint | form | token | kind | dist | row tokens | KL |
|---|---|---|---|---:|---:|---:|
| dense65_u0080 | short_add_special | `ĠFull` | content | 2 | 10 | 2.26615 |
| dense64_u0080 | short_add_special | `ĠFull` | content | 2 | 10 | 2.24948 |
| clean_pres64_u0080 | short_add_special | `ĠFull` | content | 2 | 10 | 1.53100 |
| clean_pres65_u0080 | short_add_special | `ĠFull` | content | 2 | 10 | 1.52097 |
| dense65_u0080 | short_add_special | `Ġdead` | content | 17 | 36 | 0.90689 |
| dense65_u0080 | short_add_special | `ĠW` | content | 22 | 56 | 0.89021 |
| dense64_u0080 | short_add_special | `ĠW` | content | 22 | 56 | 0.85275 |
| dense64_u0080 | short_add_special | `Ġdead` | content | 17 | 36 | 0.82704 |
| dense65_u0080 | coherent_add_special | `ĠBas` | content | 87 | 185 | 0.65267 |
| dense64_u0080 | coherent_add_special | `ĠBas` | content | 87 | 185 | 0.63589 |
| dense65_u0080 | coherent_add_special | `ĠKid` | content | 65 | 218 | 0.63569 |
| dense64_u0080 | coherent_add_special | `ĠKid` | content | 65 | 218 | 0.60833 |
| clean_pres65_u0080 | short_add_special | `ĠW` | content | 22 | 56 | 0.58162 |
| clean_pres64_u0080 | short_add_special | `ĠW` | content | 22 | 56 | 0.57549 |
| dense64_u0080 | coherent_add_special | `</s>` | special | 0 | 189 | 0.54711 |
| dense65_u0080 | coherent_add_special | `</s>` | special | 0 | 189 | 0.54600 |
| dense65_u0080 | short_add_special | `ĠD` | content | 7 | 36 | 0.49045 |
| dense64_u0080 | short_add_special | `ĠD` | content | 7 | 36 | 0.46438 |
| dense65_u0080 | coherent_no_special | `Ġan` | content | -1 | 166 | 0.42788 |
| dense64_u0080 | coherent_no_special | `Ġan` | content | -1 | 166 | 0.42179 |
| dense65_u0080 | short_add_special | `<s>` | special | 0 | 20 | 0.39628 |
| dense65_u0080 | short_add_special | `</s>` | special | 0 | 10 | 0.38976 |
| dense65_u0080 | coherent_add_special | `Ġeverybody` | content | 44 | 192 | 0.38579 |
| clean_pres65_u0080 | coherent_add_special | `ĠKid` | content | 65 | 218 | 0.38533 |
| dense64_u0080 | short_add_special | `</s>` | special | 0 | 10 | 0.38395 |
| dense64_u0080 | short_add_special | `<s>` | special | 0 | 20 | 0.37997 |
| clean_pres64_u0080 | coherent_add_special | `ĠKid` | content | 65 | 218 | 0.37642 |
| dense64_u0080 | coherent_add_special | `Ġeverybody` | content | 44 | 192 | 0.37304 |
| dense65_u0080 | short_add_special | `Ġpred` | content | 3 | 67 | 0.34948 |
| clean_pres65_u0080 | short_add_special | `Ġdead` | content | 17 | 36 | 0.33985 |

Distance table: `experiments/archive/relation_learning/data/candidate_private_slow_kl_geometry/kl_by_distance.csv`
Row-length table: `experiments/archive/relation_learning/data/candidate_private_slow_kl_geometry/kl_by_row_length.csv`
JSON: `experiments/archive/relation_learning/data/candidate_private_slow_kl_geometry/special_geometry_kl_bins.json`
