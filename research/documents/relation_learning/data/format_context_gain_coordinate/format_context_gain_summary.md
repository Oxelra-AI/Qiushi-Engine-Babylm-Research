# context geometry and v5 design format endpoint context-gain coordinate

context_gain = isolation_loss - row_context_loss. Negative endpoint-minus-anchor delta_context_gain means the endpoint relies less on adjacent row context than the anchor on the same Strict-complement spans.

Scored 651 Strict-complement sentence spans.

## Endpoint summaries

| endpoint | n | row loss | isolated loss | context gain | se | removed |
|---|---:|---:|---:|---:|---:|---:|
| chck82_slow_scale1p75 | 623 | 3.0055 | 4.0296 | 1.0241 | 0.0748 | 28 |
| coherent86_alpha075 | 623 | 2.9921 | 4.0300 | 1.0378 | 0.0750 | 28 |
| coherent_special_98097_alpha075 | 623 | 2.9873 | 3.8229 | 0.8356 | 0.0711 | 28 |
| coherent_special_98098_alpha075 | 623 | 2.9869 | 3.8232 | 0.8363 | 0.0707 | 28 |

## Paired contrasts

| endpoint | anchor | n | d_row | d_iso | d_context_gain | se | frac higher |
|---|---|---:|---:|---:|---:|---:|---:|
| coherent86_alpha075 | chck82_slow_scale1p75 | 623 | -0.0134 | +0.0004 | +0.0138 | 0.0044 | 0.501 |
| coherent_special_98097_alpha075 | chck82_slow_scale1p75 | 623 | -0.0182 | -0.2067 | -0.1885 | 0.0276 | 0.425 |
| coherent_special_98098_alpha075 | chck82_slow_scale1p75 | 623 | -0.0186 | -0.2063 | -0.1878 | 0.0274 | 0.432 |
| chck82_slow_scale1p75 | coherent86_alpha075 | 623 | +0.0134 | -0.0004 | -0.0138 | 0.0044 | 0.499 |
| coherent_special_98097_alpha075 | coherent86_alpha075 | 623 | -0.0048 | -0.2071 | -0.2022 | 0.0285 | 0.437 |
| coherent_special_98098_alpha075 | coherent86_alpha075 | 623 | -0.0052 | -0.2067 | -0.2015 | 0.0282 | 0.432 |

JSON: `experiments/archive/relation_learning/data/format_context_gain_coordinate/format_context_gain_summary.json`
