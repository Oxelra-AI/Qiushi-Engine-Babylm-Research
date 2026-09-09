# context geometry and v5 design special-token geometry KL bins

Rows per form: 48.

| endpoint | scale | coherent+special KL | short+special KL | coherent no-special KL | ratio | max short token KL |
|---|---:|---:|---:|---:|---:|---:|
| dense64_private_scale_0p55 | 0.55 | 0.00182221 | 0.00696212 | 0.00146925 | 3.821 | 1.1446 |
| dense64_private_scale_0p57 | 0.57 | 0.00197721 | 0.00761107 | 0.00158776 | 3.849 | 1.2528 |
| dense64_private_scale_0p58 | 0.58 | 0.00205784 | 0.00794947 | 0.00164920 | 3.863 | 1.3078 |
| dense64_private_scale_0p60 | 0.6 | 0.00222554 | 0.00865466 | 0.00177659 | 3.889 | 1.4191 |

## Largest KL tokens

| endpoint | form | token | kind | dist | row tokens | KL |
|---|---|---|---|---:|---:|---:|
| dense64_private_scale_0p60 | short_add_special | `ĠFull` | content | 2 | 10 | 1.41912 |
| dense64_private_scale_0p58 | short_add_special | `ĠFull` | content | 2 | 10 | 1.30777 |
| dense64_private_scale_0p57 | short_add_special | `ĠFull` | content | 2 | 10 | 1.25278 |
| dense64_private_scale_0p55 | short_add_special | `ĠFull` | content | 2 | 10 | 1.14459 |
| dense64_private_scale_0p60 | short_add_special | `ĠW` | content | 22 | 56 | 0.50964 |
| dense64_private_scale_0p58 | short_add_special | `ĠW` | content | 22 | 56 | 0.46876 |
| dense64_private_scale_0p57 | short_add_special | `ĠW` | content | 22 | 56 | 0.44889 |
| dense64_private_scale_0p55 | short_add_special | `ĠW` | content | 22 | 56 | 0.41037 |
| dense64_private_scale_0p60 | coherent_add_special | `</s>` | special | 0 | 189 | 0.35292 |
| dense64_private_scale_0p60 | coherent_add_special | `ĠKid` | content | 65 | 218 | 0.33632 |
| dense64_private_scale_0p58 | coherent_add_special | `</s>` | special | 0 | 189 | 0.32942 |
| dense64_private_scale_0p57 | coherent_add_special | `</s>` | special | 0 | 189 | 0.31793 |
| dense64_private_scale_0p60 | short_add_special | `Ġdead` | content | 17 | 36 | 0.31126 |
| dense64_private_scale_0p58 | coherent_add_special | `ĠKid` | content | 65 | 218 | 0.30582 |
| dense64_private_scale_0p55 | coherent_add_special | `</s>` | special | 0 | 189 | 0.29545 |
| dense64_private_scale_0p57 | coherent_add_special | `ĠKid` | content | 65 | 218 | 0.29121 |
| dense64_private_scale_0p60 | short_add_special | `ĠD` | content | 7 | 36 | 0.28855 |
| dense64_private_scale_0p58 | short_add_special | `Ġdead` | content | 17 | 36 | 0.26686 |
| dense64_private_scale_0p58 | short_add_special | `ĠD` | content | 7 | 36 | 0.26674 |
| dense64_private_scale_0p55 | coherent_add_special | `ĠKid` | content | 65 | 218 | 0.26329 |
| dense64_private_scale_0p60 | coherent_no_special | `Ġan` | content | -1 | 166 | 0.25982 |
| dense64_private_scale_0p57 | short_add_special | `ĠD` | content | 7 | 36 | 0.25607 |
| dense64_private_scale_0p57 | short_add_special | `Ġdead` | content | 17 | 36 | 0.24662 |
| dense64_private_scale_0p58 | coherent_no_special | `Ġan` | content | -1 | 166 | 0.24066 |
| dense64_private_scale_0p55 | short_add_special | `ĠD` | content | 7 | 36 | 0.23522 |
| dense64_private_scale_0p60 | short_add_special | `<s>` | special | 0 | 20 | 0.23376 |
| dense64_private_scale_0p57 | coherent_no_special | `Ġan` | content | -1 | 166 | 0.23133 |
| dense64_private_scale_0p58 | short_add_special | `<s>` | special | 0 | 20 | 0.21679 |
| dense64_private_scale_0p55 | coherent_no_special | `Ġan` | content | -1 | 166 | 0.21320 |
| dense64_private_scale_0p60 | short_add_special | `</s>` | special | 0 | 10 | 0.21002 |

Distance table: `experiments/archive/relation_learning/data/shrinkage_scale_kl_screen/kl_by_distance.csv`
Row-length table: `experiments/archive/relation_learning/data/shrinkage_scale_kl_screen/kl_by_row_length.csv`
JSON: `experiments/archive/relation_learning/data/shrinkage_scale_kl_screen/special_geometry_kl_bins.json`
