# cluster continuation noaoa result cluster held-out diagnostic

Created UTC: 2026-08-27T14:23:12Z
Held-out sentences: 1276; seen sentences: 1276.

## Best held-out loss per arm

| arm | checkpoint | heldout loss | seen loss | generalization gap |
|---|---:|---:|---:|---:|
| E1_true_cluster | chck_95M | 4.3199 | 4.3112 | +0.0088 |
| E2_anchor_shuffle | chck_99M | 4.3064 | 4.3051 | +0.0013 |
| E3_anchor_repeat | chck_99M | 4.3328 | 4.3152 | +0.0176 |
| E4_untouched_tail | chck_95M | 4.3264 | 4.3197 | +0.0067 |

## E1 contrasts by checkpoint

### chck_85M
- `E1_minus_E2_anchor_shuffle`: heldout -0.0124; seen -0.0103; gap -0.0022
- `E1_minus_E3_anchor_repeat`: heldout -0.0412; seen -0.0414; gap +0.0002
- `E1_minus_E4_untouched_tail`: heldout -0.0509; seen -0.0492; gap -0.0017
### chck_90M
- `E1_minus_E2_anchor_shuffle`: heldout +0.0116; seen +0.0006; gap +0.0109
- `E1_minus_E3_anchor_repeat`: heldout +0.0067; seen +0.0092; gap -0.0025
- `E1_minus_E4_untouched_tail`: heldout -0.0011; seen -0.0020; gap +0.0009
### chck_95M
- `E1_minus_E2_anchor_shuffle`: heldout +0.0108; seen +0.0009; gap +0.0098
- `E1_minus_E3_anchor_repeat`: heldout -0.0158; seen -0.0112; gap -0.0045
- `E1_minus_E4_untouched_tail`: heldout -0.0065; seen -0.0085; gap +0.0020
### chck_99M
- `E1_minus_E2_anchor_shuffle`: heldout +0.0210; seen +0.0087; gap +0.0123
- `E1_minus_E3_anchor_repeat`: heldout -0.0055; seen -0.0014; gap -0.0041
- `E1_minus_E4_untouched_tail`: heldout -0.0075; seen -0.0143; gap +0.0068

E1 best across arm bests: `False`.

