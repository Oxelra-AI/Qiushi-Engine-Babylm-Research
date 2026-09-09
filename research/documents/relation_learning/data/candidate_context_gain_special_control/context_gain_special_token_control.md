# earlier analysis context/isolation coordinate with and without special tokens

A with-special contrast against a no-special-trained anchor can mix context reliance with learned boundary-token mismatch. A no-special contrast removes that boundary-token channel on the same text coordinate.

Scored 651 Strict-complement sentence spans.

## Endpoint summaries

| endpoint | form | n | row loss | isolated loss | context gain | se | removed |
|---|---|---:|---:|---:|---:|---:|---:|
| chck82_slow_scale1p75 | no_special | 624 | 2.9913 | 3.8894 | 0.8981 | 0.0707 | 27 |
| chck82_slow_scale1p75 | with_special | 623 | 3.0055 | 4.0296 | 1.0241 | 0.0748 | 28 |
| clean_pres64_u0080 | no_special | 624 | 3.0063 | 3.9388 | 0.9325 | 0.0726 | 27 |
| clean_pres64_u0080 | with_special | 623 | 3.0366 | 4.0829 | 1.0463 | 0.0769 | 28 |
| clean_pres65_u0080 | no_special | 624 | 3.0066 | 3.9387 | 0.9321 | 0.0726 | 27 |
| clean_pres65_u0080 | with_special | 623 | 3.0366 | 4.0828 | 1.0462 | 0.0769 | 28 |
| coherent86_alpha075 | no_special | 624 | 2.9772 | 3.8852 | 0.9080 | 0.0707 | 27 |
| coherent86_alpha075 | with_special | 623 | 2.9921 | 4.0300 | 1.0378 | 0.0750 | 28 |
| dense64_u0080 | no_special | 624 | 3.0271 | 3.9760 | 0.9488 | 0.0735 | 27 |
| dense64_u0080 | with_special | 623 | 3.0643 | 4.1175 | 1.0532 | 0.0778 | 28 |
| dense65_u0080 | no_special | 624 | 3.0299 | 3.9782 | 0.9483 | 0.0736 | 27 |
| dense65_u0080 | with_special | 623 | 3.0674 | 4.1200 | 1.0527 | 0.0779 | 28 |

## Paired contrasts

| endpoint | anchor | form | n | d_row | d_iso | d_context_gain | se | frac higher |
|---|---|---|---:|---:|---:|---:|---:|---:|
| clean_pres64_u0080 | chck82_slow_scale1p75 | no_special | 624 | +0.0150 | +0.0494 | +0.0344 | 0.0106 | 0.562 |
| clean_pres65_u0080 | chck82_slow_scale1p75 | no_special | 624 | +0.0153 | +0.0493 | +0.0340 | 0.0103 | 0.559 |
| coherent86_alpha075 | chck82_slow_scale1p75 | no_special | 624 | -0.0141 | -0.0042 | +0.0099 | 0.0044 | 0.526 |
| dense64_u0080 | chck82_slow_scale1p75 | no_special | 624 | +0.0358 | +0.0865 | +0.0507 | 0.0136 | 0.577 |
| dense65_u0080 | chck82_slow_scale1p75 | no_special | 624 | +0.0386 | +0.0888 | +0.0501 | 0.0136 | 0.577 |
| chck82_slow_scale1p75 | coherent86_alpha075 | no_special | 624 | +0.0141 | +0.0042 | -0.0099 | 0.0044 | 0.474 |
| clean_pres64_u0080 | coherent86_alpha075 | no_special | 624 | +0.0291 | +0.0535 | +0.0244 | 0.0089 | 0.559 |
| clean_pres65_u0080 | coherent86_alpha075 | no_special | 624 | +0.0294 | +0.0535 | +0.0241 | 0.0087 | 0.556 |
| dense64_u0080 | coherent86_alpha075 | no_special | 624 | +0.0499 | +0.0907 | +0.0408 | 0.0122 | 0.577 |
| dense65_u0080 | coherent86_alpha075 | no_special | 624 | +0.0527 | +0.0930 | +0.0402 | 0.0123 | 0.574 |
| clean_pres64_u0080 | chck82_slow_scale1p75 | with_special | 623 | +0.0311 | +0.0533 | +0.0222 | 0.0105 | 0.546 |
| clean_pres65_u0080 | chck82_slow_scale1p75 | with_special | 623 | +0.0311 | +0.0533 | +0.0221 | 0.0103 | 0.551 |
| coherent86_alpha075 | chck82_slow_scale1p75 | with_special | 623 | -0.0134 | +0.0004 | +0.0138 | 0.0044 | 0.501 |
| dense64_u0080 | chck82_slow_scale1p75 | with_special | 623 | +0.0588 | +0.0879 | +0.0291 | 0.0136 | 0.555 |
| dense65_u0080 | chck82_slow_scale1p75 | with_special | 623 | +0.0619 | +0.0904 | +0.0286 | 0.0137 | 0.551 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | 623 | +0.0134 | -0.0004 | -0.0138 | 0.0044 | 0.499 |
| clean_pres64_u0080 | coherent86_alpha075 | with_special | 623 | +0.0445 | +0.0529 | +0.0084 | 0.0089 | 0.559 |
| clean_pres65_u0080 | coherent86_alpha075 | with_special | 623 | +0.0445 | +0.0529 | +0.0084 | 0.0087 | 0.555 |
| dense64_u0080 | coherent86_alpha075 | with_special | 623 | +0.0722 | +0.0875 | +0.0153 | 0.0121 | 0.559 |
| dense65_u0080 | coherent86_alpha075 | with_special | 623 | +0.0752 | +0.0901 | +0.0148 | 0.0123 | 0.562 |
| chck82_slow_scale1p75 | chck82_slow_scale1p75 | with_special_minus_no_special | 623 | +0.0124 | +0.1379 | +0.1255 | 0.0284 | 0.530 |
| clean_pres64_u0080 | clean_pres64_u0080 | with_special_minus_no_special | 623 | +0.0283 | +0.1420 | +0.1136 | 0.0330 | 0.502 |
| clean_pres65_u0080 | clean_pres65_u0080 | with_special_minus_no_special | 623 | +0.0281 | +0.1420 | +0.1139 | 0.0329 | 0.501 |
| coherent86_alpha075 | coherent86_alpha075 | with_special_minus_no_special | 623 | +0.0131 | +0.1426 | +0.1295 | 0.0293 | 0.522 |
| dense64_u0080 | dense64_u0080 | with_special_minus_no_special | 623 | +0.0352 | +0.1394 | +0.1042 | 0.0345 | 0.478 |
| dense65_u0080 | dense65_u0080 | with_special_minus_no_special | 623 | +0.0354 | +0.1397 | +0.1043 | 0.0346 | 0.478 |

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
| clean_pres64_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 0_edge_token | 203 | +0.0055 | 0.0368 | 0.517 |
| clean_pres64_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 1 | 188 | +0.0625 | 0.0195 | 0.426 |
| clean_pres64_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 16+ | 231 | +0.0360 | 0.0204 | 0.472 |
| clean_pres64_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 2-3 | 380 | +0.0594 | 0.0132 | 0.392 |
| clean_pres64_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 4-7 | 677 | +0.0505 | 0.0098 | 0.403 |
| clean_pres64_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 8-15 | 557 | +0.0459 | 0.0099 | 0.422 |
| clean_pres65_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 0_edge_token | 203 | +0.0042 | 0.0360 | 0.532 |
| clean_pres65_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 1 | 188 | +0.0635 | 0.0193 | 0.420 |
| clean_pres65_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 16+ | 231 | +0.0363 | 0.0202 | 0.468 |
| clean_pres65_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 2-3 | 380 | +0.0584 | 0.0130 | 0.400 |
| clean_pres65_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 4-7 | 677 | +0.0512 | 0.0097 | 0.394 |
| clean_pres65_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 8-15 | 557 | +0.0459 | 0.0098 | 0.418 |
| dense64_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 0_edge_token | 203 | +0.0441 | 0.0469 | 0.502 |
| dense64_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 1 | 188 | +0.0945 | 0.0268 | 0.415 |
| dense64_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 16+ | 231 | +0.0584 | 0.0284 | 0.472 |
| dense64_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 2-3 | 380 | +0.0967 | 0.0186 | 0.376 |
| dense64_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 4-7 | 677 | +0.0840 | 0.0139 | 0.380 |
| dense64_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 8-15 | 557 | +0.0834 | 0.0143 | 0.395 |
| dense65_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 0_edge_token | 203 | +0.0423 | 0.0480 | 0.502 |
| dense65_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 1 | 188 | +0.0974 | 0.0272 | 0.410 |
| dense65_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 16+ | 231 | +0.0602 | 0.0287 | 0.468 |
| dense65_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 2-3 | 380 | +0.0981 | 0.0188 | 0.387 |
| dense65_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 4-7 | 677 | +0.0863 | 0.0140 | 0.375 |
| dense65_u0080 | coherent86_alpha075 | no_special | target_edge_bin | 8-15 | 557 | +0.0858 | 0.0145 | 0.400 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | target_edge_bin | 0_edge_token | 203 | -0.0289 | 0.0118 | 0.581 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | target_edge_bin | 1 | 188 | +0.0048 | 0.0116 | 0.505 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | target_edge_bin | 16+ | 231 | +0.0073 | 0.0083 | 0.489 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | target_edge_bin | 2-3 | 380 | -0.0034 | 0.0076 | 0.537 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | target_edge_bin | 4-7 | 677 | +0.0143 | 0.0055 | 0.474 |
| chck82_slow_scale1p75 | coherent86_alpha075 | with_special | target_edge_bin | 8-15 | 557 | +0.0108 | 0.0058 | 0.492 |
| clean_pres64_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 0_edge_token | 203 | +0.1422 | 0.0212 | 0.325 |
| clean_pres64_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 1 | 188 | +0.0284 | 0.0165 | 0.452 |
| clean_pres64_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 16+ | 231 | +0.0336 | 0.0175 | 0.455 |
| clean_pres64_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 2-3 | 380 | +0.0521 | 0.0114 | 0.434 |
| clean_pres64_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 4-7 | 677 | +0.0393 | 0.0085 | 0.411 |
| clean_pres64_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 8-15 | 557 | +0.0456 | 0.0091 | 0.422 |
| clean_pres65_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 0_edge_token | 203 | +0.1394 | 0.0210 | 0.330 |
| clean_pres65_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 1 | 188 | +0.0299 | 0.0163 | 0.447 |
| clean_pres65_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 16+ | 231 | +0.0345 | 0.0174 | 0.446 |
| clean_pres65_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 2-3 | 380 | +0.0515 | 0.0112 | 0.432 |
| clean_pres65_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 4-7 | 677 | +0.0403 | 0.0085 | 0.408 |
| clean_pres65_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 8-15 | 557 | +0.0454 | 0.0090 | 0.420 |
| dense64_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 0_edge_token | 203 | +0.2244 | 0.0299 | 0.300 |
| dense64_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 1 | 188 | +0.0461 | 0.0221 | 0.447 |
| dense64_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 16+ | 231 | +0.0560 | 0.0244 | 0.446 |
| dense64_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 2-3 | 380 | +0.0862 | 0.0161 | 0.400 |
| dense64_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 4-7 | 677 | +0.0654 | 0.0124 | 0.378 |
| dense64_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 8-15 | 557 | +0.0816 | 0.0133 | 0.381 |
| dense65_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 0_edge_token | 203 | +0.2276 | 0.0303 | 0.296 |
| dense65_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 1 | 188 | +0.0486 | 0.0224 | 0.441 |
| dense65_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 16+ | 231 | +0.0578 | 0.0246 | 0.450 |
| dense65_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 2-3 | 380 | +0.0878 | 0.0162 | 0.387 |
| dense65_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 4-7 | 677 | +0.0677 | 0.0125 | 0.375 |
| dense65_u0080 | coherent86_alpha075 | with_special | target_edge_bin | 8-15 | 557 | +0.0840 | 0.0135 | 0.381 |

JSON: `experiments/archive/relation_learning/data/candidate_context_gain_special_control/context_gain_special_token_control.json`
