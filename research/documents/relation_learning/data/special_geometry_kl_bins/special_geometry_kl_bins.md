# context geometry and v5 design special-token geometry KL bins

Rows per form: 32.

| endpoint | scale | coherent+special KL | short+special KL | coherent no-special KL | ratio | max short token KL |
|---|---:|---:|---:|---:|---:|---:|
| coherent86_alpha075 | 0.75 | 0.00082465 | 0.00183627 | 0.00077791 | 2.227 | 0.0486 |
| coherent_special_98097_alpha075 | 0.75 | 0.00193546 | 0.00974691 | 0.00099256 | 5.036 | 1.1635 |
| coherent_special_98098_alpha075 | 0.75 | 0.00240540 | 0.01199229 | 0.00108493 | 4.986 | 1.1311 |

## Largest KL tokens

| endpoint | form | token | kind | dist | row tokens | KL |
|---|---|---|---|---:|---:|---:|
| coherent_special_98098_alpha075 | coherent_add_special | `ĠInstead` | content | 1 | 232 | 2.87616 |
| coherent_special_98097_alpha075 | coherent_add_special | `ĠInstead` | content | 1 | 232 | 1.90890 |
| coherent_special_98098_alpha075 | coherent_add_special | `ĠAP` | content | 1 | 205 | 1.55897 |
| coherent_special_98097_alpha075 | coherent_add_special | `ĠAP` | content | 1 | 205 | 1.34824 |
| coherent_special_98098_alpha075 | coherent_add_special | `ĠWOMAN` | content | 1 | 242 | 1.32498 |
| coherent_special_98097_alpha075 | short_add_special | `ĠApril` | content | 1 | 24 | 1.16346 |
| coherent_special_98098_alpha075 | short_add_special | `ĠApril` | content | 1 | 24 | 1.13112 |
| coherent_special_98098_alpha075 | short_add_special | `</s>` | special | 0 | 24 | 1.06664 |
| coherent_special_98098_alpha075 | short_add_special | `<s>` | special | 0 | 24 | 0.91521 |
| coherent_special_98097_alpha075 | coherent_add_special | `ĠWOMAN` | content | 1 | 242 | 0.65137 |
| coherent_special_98097_alpha075 | short_add_special | `<s>` | special | 0 | 24 | 0.60925 |
| coherent_special_98098_alpha075 | coherent_add_special | `Ġre` | content | 1 | 243 | 0.60168 |
| coherent_special_98097_alpha075 | short_add_special | `</s>` | special | 0 | 24 | 0.56516 |
| coherent_special_98097_alpha075 | coherent_add_special | `Ġdisc` | content | 1 | 246 | 0.47698 |
| coherent_special_98097_alpha075 | short_add_special | `<s>` | special | 0 | 36 | 0.46410 |
| coherent_special_98097_alpha075 | short_add_special | `ĠHow` | content | 1 | 61 | 0.45777 |
| coherent_special_98098_alpha075 | short_add_special | `<s>` | special | 0 | 36 | 0.41091 |
| coherent_special_98098_alpha075 | short_add_special | `ĠHow` | content | 1 | 61 | 0.39580 |
| coherent_special_98098_alpha075 | short_add_special | `ĠWhy` | content | 1 | 23 | 0.38759 |
| coherent_special_98097_alpha075 | coherent_add_special | `Ġre` | content | 1 | 243 | 0.35846 |
| coherent_special_98097_alpha075 | short_add_special | `ĠWhy` | content | 1 | 23 | 0.33338 |
| coherent_special_98098_alpha075 | coherent_add_special | `Ġdisc` | content | 1 | 246 | 0.30882 |
| coherent_special_98098_alpha075 | short_add_special | `ĠWhether` | content | 1 | 67 | 0.27817 |
| coherent_special_98097_alpha075 | short_add_special | `ĠWhether` | content | 1 | 67 | 0.25090 |
| coherent_special_98098_alpha075 | coherent_add_special | `<s>` | special | 0 | 242 | 0.24997 |
| coherent_special_98097_alpha075 | short_add_special | `ĠNice` | content | 1 | 10 | 0.24895 |
| coherent_special_98098_alpha075 | coherent_add_special | `</s>` | special | 0 | 256 | 0.24736 |
| coherent_special_98098_alpha075 | coherent_add_special | `Ġmulti` | content | 1 | 247 | 0.24049 |
| coherent_special_98098_alpha075 | short_add_special | `ĠNice` | content | 1 | 10 | 0.22003 |
| coherent_special_98098_alpha075 | short_add_special | `</s>` | special | 0 | 8 | 0.21499 |

Distance table: `experiments/archive/relation_learning/data/special_geometry_kl_bins/kl_by_distance.csv`
Row-length table: `experiments/archive/relation_learning/data/special_geometry_kl_bins/kl_by_row_length.csv`
JSON: `experiments/archive/relation_learning/data/special_geometry_kl_bins/special_geometry_kl_bins.json`
