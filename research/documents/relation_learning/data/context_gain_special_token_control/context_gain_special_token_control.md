# earlier analysis context/isolation coordinate with and without special tokens

A with-special contrast against a no-special-trained anchor can mix context reliance with learned boundary-token mismatch. A no-special contrast removes that boundary-token channel on the same text coordinate.

Scored 651 Strict-complement sentence spans.

## Endpoint summaries

| endpoint | form | n | row loss | isolated loss | context gain | se | removed |
|---|---|---:|---:|---:|---:|---:|---:|
| chck82_slow_scale1p75 | no_special | 624 | 2.9913 | 3.8894 | 0.8981 | 0.0707 | 27 |
| chck82_slow_scale1p75 | with_special | 623 | 3.0055 | 4.0296 | 1.0241 | 0.0748 | 28 |
| coherent86_alpha075 | no_special | 624 | 2.9772 | 3.8852 | 0.9080 | 0.0707 | 27 |
| coherent86_alpha075 | with_special | 623 | 2.9921 | 4.0300 | 1.0378 | 0.0750 | 28 |
| coherent_special_98097_alpha075 | no_special | 624 | 2.9775 | 3.9083 | 0.9308 | 0.0710 | 27 |
| coherent_special_98097_alpha075 | with_special | 623 | 2.9873 | 3.8229 | 0.8356 | 0.0711 | 28 |
| coherent_special_98098_alpha075 | no_special | 624 | 2.9768 | 3.9163 | 0.9395 | 0.0709 | 27 |
| coherent_special_98098_alpha075 | with_special | 623 | 2.9869 | 3.8232 | 0.8363 | 0.0707 | 28 |

## Paired contrasts

| endpoint | anchor | form | n | d_row | d_iso | d_context_gain | se | frac higher |
|---|---|---|---:|---:|---:|---:|---:|---:|
| coherent86_alpha075 | chck82_slow_scale1p75 | no_special | 624 | -0.0141 | -0.0042 | +0.0099 | 0.0044 | 0.526 |
| coherent_special_98097_alpha075 | chck82_slow_scale1p75 | no_special | 624 | -0.0138 | +0.0189 | +0.0327 | 0.0098 | 0.538 |
| coherent_special_98098_alpha075 | chck82_slow_scale1p75 | no_special | 624 | -0.0145 | +0.0269 | +0.0414 | 0.0109 | 0.558 |
| chck82_slow_scale1p75 | coherent86_alpha075 | no_special | 624 | +0.0141 | +0.0042 | -0.0099 | 0.0044 | 0.474 |
| coherent_special_98097_alpha075 | coherent86_alpha075 | no_special | 624 | +0.0003 | +0.0231 | +0.0228 | 0.0102 | 0.546 |
| coherent_special_98098_alpha075 | coherent86_alpha075 | no_special | 624 | -0.0004 | +0.0311 | +0.0315 | 0.0108 | 0.553 |
| coherent86_alpha075 | chck82_slow_scale1p75 | with_special | 623 | -0.0134 | +0.0004 | +0.0138 | 0.0044 | 0.501 |
| coherent_special_98097_alpha075 | chck82_slow_scale1p75 | with_special | 623 | -0.0182 | -0.2067 | -0.1885 | 0.0276 | 0.425 |
| coherent_special_98098_alpha075 | chck82_slow_scale1p75 | with_special | 623 | -0.0186 | -0.2063 | -0.1878 | 0.0274 | 0.432 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | 623 | +0.0134 | -0.0004 | -0.0138 | 0.0044 | 0.499 |
| coherent_special_98097_alpha075 | coherent86_alpha075 | with_special | 623 | -0.0048 | -0.2071 | -0.2022 | 0.0285 | 0.437 |
| coherent_special_98098_alpha075 | coherent86_alpha075 | with_special | 623 | -0.0052 | -0.2067 | -0.2015 | 0.0282 | 0.432 |
| chck82_slow_scale1p75 | chck82_slow_scale1p75 | with_special_minus_no_special | 623 | +0.0124 | +0.1379 | +0.1255 | 0.0284 | 0.530 |
| coherent86_alpha075 | coherent86_alpha075 | with_special_minus_no_special | 623 | +0.0131 | +0.1426 | +0.1295 | 0.0293 | 0.522 |
| coherent_special_98097_alpha075 | coherent_special_98097_alpha075 | with_special_minus_no_special | 623 | +0.0079 | -0.0878 | -0.0956 | 0.0182 | 0.419 |
| coherent_special_98098_alpha075 | coherent_special_98098_alpha075 | with_special_minus_no_special | 623 | +0.0083 | -0.0951 | -0.1035 | 0.0181 | 0.403 |

## Isolation token-loss deltas by target edge distance

Shown for endpoint-minus-anchor on masked tokens in isolated sentences.

| endpoint | anchor | form | bin field | bin | n tok | d token loss | se | frac improved |
|---|---|---|---|---|---:|---:|---:|---:|
| chck82_slow_scale1p75 | coherent86_alpha075 | no_special | target_edge_bin | 0_edge_token | 203 | +0.0285 | 0.0156 | 0.463 |
| chck82_slow_scale1p75 | coherent86_alpha075 | no_special | target_edge_bin | 1 | 188 | +0.0031 | 0.0117 | 0.500 |
| chck82_slow_scale1p75 | coherent86_alpha075 | no_special | target_edge_bin | 16+ | 231 | +0.0076 | 0.0086 | 0.481 |
| chck82_slow_scale1p75 | coherent86_alpha075 | no_special | target_edge_bin | 2-3 | 380 | -0.0003 | 0.0071 | 0.526 |
| chck82_slow_scale1p75 | coherent86_alpha075 | no_special | target_edge_bin | 4-7 | 677 | +0.0143 | 0.0054 | 0.471 |
| chck82_slow_scale1p75 | coherent86_alpha075 | no_special | target_edge_bin | 8-15 | 557 | +0.0076 | 0.0057 | 0.478 |
| coherent_special_98097_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 0_edge_token | 203 | +0.0710 | 0.0327 | 0.419 |
| coherent_special_98097_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 1 | 188 | +0.1800 | 0.0458 | 0.426 |
| coherent_special_98097_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 16+ | 231 | +0.0034 | 0.0090 | 0.489 |
| coherent_special_98097_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 2-3 | 380 | +0.0558 | 0.0208 | 0.495 |
| coherent_special_98097_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 4-7 | 677 | +0.0057 | 0.0066 | 0.517 |
| coherent_special_98097_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 8-15 | 557 | -0.0012 | 0.0071 | 0.519 |
| coherent_special_98098_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 0_edge_token | 203 | +0.1079 | 0.0350 | 0.365 |
| coherent_special_98098_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 1 | 188 | +0.1914 | 0.0522 | 0.415 |
| coherent_special_98098_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 16+ | 231 | -0.0099 | 0.0099 | 0.532 |
| coherent_special_98098_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 2-3 | 380 | +0.0637 | 0.0228 | 0.484 |
| coherent_special_98098_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 4-7 | 677 | +0.0079 | 0.0072 | 0.505 |
| coherent_special_98098_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 8-15 | 557 | -0.0063 | 0.0078 | 0.521 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | target_edge_bin | 0_edge_token | 203 | -0.0289 | 0.0118 | 0.581 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | target_edge_bin | 1 | 188 | +0.0048 | 0.0116 | 0.505 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | target_edge_bin | 16+ | 231 | +0.0073 | 0.0083 | 0.489 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | target_edge_bin | 2-3 | 380 | -0.0034 | 0.0076 | 0.537 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | target_edge_bin | 4-7 | 677 | +0.0143 | 0.0055 | 0.474 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | target_edge_bin | 8-15 | 557 | +0.0108 | 0.0058 | 0.492 |
| coherent_special_98097_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 0_edge_token | 203 | -1.5256 | 0.1366 | 0.828 |
| coherent_special_98097_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 1 | 188 | +0.0685 | 0.0366 | 0.457 |
| coherent_special_98097_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 16+ | 231 | +0.0055 | 0.0106 | 0.472 |
| coherent_special_98097_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 2-3 | 380 | -0.0198 | 0.0261 | 0.511 |
| coherent_special_98097_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 4-7 | 677 | +0.0074 | 0.0092 | 0.502 |
| coherent_special_98097_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 8-15 | 557 | -0.0082 | 0.0099 | 0.555 |
| coherent_special_98098_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 0_edge_token | 203 | -1.4406 | 0.1328 | 0.833 |
| coherent_special_98098_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 1 | 188 | +0.0509 | 0.0369 | 0.441 |
| coherent_special_98098_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 16+ | 231 | -0.0186 | 0.0112 | 0.532 |
| coherent_special_98098_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 2-3 | 380 | -0.0125 | 0.0185 | 0.503 |
| coherent_special_98098_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 4-7 | 677 | +0.0064 | 0.0083 | 0.490 |
| coherent_special_98098_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 8-15 | 557 | -0.0140 | 0.0097 | 0.585 |

JSON: `experiments/archive/relation_learning/data/context_gain_special_token_control/context_gain_special_token_control.json`
