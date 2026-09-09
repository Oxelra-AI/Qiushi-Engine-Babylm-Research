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
| half_98097_alpha075 | no_special | 624 | 2.9860 | 3.8404 | 0.8544 | 0.0670 | 27 |
| half_98097_alpha075 | with_special | 623 | 2.9967 | 3.7604 | 0.7637 | 0.0700 | 28 |
| half_98098_alpha075 | no_special | 624 | 2.9974 | 3.8424 | 0.8450 | 0.0660 | 27 |
| half_98098_alpha075 | with_special | 623 | 3.0086 | 3.7381 | 0.7294 | 0.0680 | 28 |

## Paired contrasts

| endpoint | anchor | form | n | d_row | d_iso | d_context_gain | se | frac higher |
|---|---|---|---:|---:|---:|---:|---:|---:|
| coherent86_alpha075 | chck82_slow_scale1p75 | no_special | 624 | -0.0141 | -0.0042 | +0.0099 | 0.0044 | 0.526 |
| half_98097_alpha075 | chck82_slow_scale1p75 | no_special | 624 | -0.0053 | -0.0490 | -0.0437 | 0.0165 | 0.449 |
| half_98098_alpha075 | chck82_slow_scale1p75 | no_special | 624 | +0.0061 | -0.0470 | -0.0531 | 0.0175 | 0.423 |
| chck82_slow_scale1p75 | coherent86_alpha075 | no_special | 624 | +0.0141 | +0.0042 | -0.0099 | 0.0044 | 0.474 |
| half_98097_alpha075 | coherent86_alpha075 | no_special | 624 | +0.0088 | -0.0448 | -0.0536 | 0.0168 | 0.444 |
| half_98098_alpha075 | coherent86_alpha075 | no_special | 624 | +0.0202 | -0.0428 | -0.0630 | 0.0178 | 0.436 |
| coherent86_alpha075 | chck82_slow_scale1p75 | with_special | 623 | -0.0134 | +0.0004 | +0.0138 | 0.0044 | 0.501 |
| half_98097_alpha075 | chck82_slow_scale1p75 | with_special | 623 | -0.0088 | -0.2692 | -0.2603 | 0.0446 | 0.327 |
| half_98098_alpha075 | chck82_slow_scale1p75 | with_special | 623 | +0.0031 | -0.2915 | -0.2947 | 0.0426 | 0.347 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | 623 | +0.0134 | -0.0004 | -0.0138 | 0.0044 | 0.499 |
| half_98097_alpha075 | coherent86_alpha075 | with_special | 623 | +0.0046 | -0.2695 | -0.2741 | 0.0450 | 0.335 |
| half_98098_alpha075 | coherent86_alpha075 | with_special | 623 | +0.0165 | -0.2919 | -0.3084 | 0.0432 | 0.347 |
| chck82_slow_scale1p75 | chck82_slow_scale1p75 | with_special_minus_no_special | 623 | +0.0124 | +0.1379 | +0.1255 | 0.0284 | 0.530 |
| coherent86_alpha075 | coherent86_alpha075 | with_special_minus_no_special | 623 | +0.0131 | +0.1426 | +0.1295 | 0.0293 | 0.522 |
| half_98097_alpha075 | half_98097_alpha075 | with_special_minus_no_special | 623 | +0.0089 | -0.0827 | -0.0916 | 0.0356 | 0.408 |
| half_98098_alpha075 | half_98098_alpha075 | with_special_minus_no_special | 623 | +0.0094 | -0.1071 | -0.1164 | 0.0340 | 0.411 |

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
| half_98097_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 0_edge_token | 203 | +0.2407 | 0.0608 | 0.374 |
| half_98097_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 1 | 188 | +0.1020 | 0.0797 | 0.489 |
| half_98097_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 16+ | 231 | -0.0348 | 0.0228 | 0.550 |
| half_98097_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 2-3 | 380 | -0.0552 | 0.0326 | 0.584 |
| half_98097_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 4-7 | 677 | -0.1087 | 0.0169 | 0.589 |
| half_98097_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 8-15 | 557 | -0.0716 | 0.0159 | 0.551 |
| half_98098_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 0_edge_token | 203 | +0.2761 | 0.0673 | 0.360 |
| half_98098_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 1 | 188 | +0.1421 | 0.0874 | 0.463 |
| half_98098_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 16+ | 231 | -0.0428 | 0.0250 | 0.528 |
| half_98098_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 2-3 | 380 | -0.0584 | 0.0347 | 0.553 |
| half_98098_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 4-7 | 677 | -0.1185 | 0.0195 | 0.576 |
| half_98098_alpha075 | coherent86_alpha075 | no_special | target_edge_bin | 8-15 | 557 | -0.0681 | 0.0183 | 0.553 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | target_edge_bin | 0_edge_token | 203 | -0.0289 | 0.0118 | 0.581 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | target_edge_bin | 1 | 188 | +0.0048 | 0.0116 | 0.505 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | target_edge_bin | 16+ | 231 | +0.0073 | 0.0083 | 0.489 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | target_edge_bin | 2-3 | 380 | -0.0034 | 0.0076 | 0.537 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | target_edge_bin | 4-7 | 677 | +0.0143 | 0.0055 | 0.474 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | target_edge_bin | 8-15 | 557 | +0.0108 | 0.0058 | 0.492 |
| half_98097_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 0_edge_token | 203 | -1.6266 | 0.2339 | 0.631 |
| half_98097_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 1 | 188 | -0.1413 | 0.0688 | 0.596 |
| half_98097_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 16+ | 231 | -0.0496 | 0.0268 | 0.515 |
| half_98097_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 2-3 | 380 | -0.1880 | 0.0396 | 0.589 |
| half_98097_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 4-7 | 677 | -0.1365 | 0.0198 | 0.586 |
| half_98097_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 8-15 | 557 | -0.0934 | 0.0200 | 0.564 |
| half_98098_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 0_edge_token | 203 | -1.7201 | 0.2231 | 0.650 |
| half_98098_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 1 | 188 | -0.1433 | 0.0709 | 0.612 |
| half_98098_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 16+ | 231 | -0.0610 | 0.0302 | 0.532 |
| half_98098_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 2-3 | 380 | -0.1948 | 0.0377 | 0.608 |
| half_98098_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 4-7 | 677 | -0.1411 | 0.0214 | 0.598 |
| half_98098_alpha075 | coherent86_alpha075 | with_special | target_edge_bin | 8-15 | 557 | -0.0866 | 0.0218 | 0.583 |

JSON: `experiments/archive/relation_learning/data/half_context_gain_special_token_control/context_gain_special_token_control.json`
