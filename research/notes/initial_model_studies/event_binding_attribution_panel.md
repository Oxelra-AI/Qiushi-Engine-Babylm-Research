# event binding attribution panel event-binding attribution panel

Decision: **LEADER_SIGNAL_NOT_REPRODUCED_BY_TESTED_NONDATA_FACTORS**

No new model was trained. All rows use the exact Step 302b diagnostic.

## Checkpoints

| checkpoint | effect mean | 95% CI | fraction > 0 |
|---|---:|---:|---:|
| protected8x480_10M | +0.000065 | [-0.000170, +0.000296] | 0.505 |
| protected8x480_100M | +0.087790 | [+0.062525, +0.113914] | 0.614 |
| wwm43_100M | +0.133062 | [+0.095456, +0.171452] | 0.640 |
| s1_shape12x384_10M | -0.000037 | [-0.000142, +0.000071] | 0.462 |
| s1_shape12x384_100M | +0.017059 | [-0.003112, +0.037744] | 0.529 |
| s2_curriculum12x384_100M | -0.001929 | [-0.005907, +0.002026] | 0.469 |
| official40k8x480_100M | +0.040612 | [+0.020118, +0.061643] | 0.555 |
| s3_shape40k_10M | +0.000176 | [-0.000064, +0.000426] | 0.514 |
| lamb_s1_10M | +0.000002 | [-0.000029, +0.000033] | 0.483 |
| public_leader | +0.301854 | [+0.224782, +0.379233] | 0.638 |

## Matched contrasts

| contrast | delta mean | 95% CI |
|---|---:|---:|
| seed43_minus_seed42_at_100M | +0.045272 | [+0.019642, +0.071285] |
| shape12x384_minus_8x480_at_10M | -0.000102 | [-0.000356, +0.000159] |
| shape12x384_minus_8x480_at_100M | -0.070731 | [-0.087712, -0.053913] |
| curriculum_minus_flat_at_100M | -0.018989 | [-0.038272, -0.000385] |
| 40k_minus_16k_under_8x480_at_100M | -0.047178 | [-0.062713, -0.031909] |
| 40k_minus_16k_under_12x384_at_10M | +0.000214 | [-0.000042, +0.000481] |
| lamb_minus_adamw_under_12x384_at_10M | +0.000039 | [-0.000075, +0.000150] |
| public_leader_minus_protected | +0.214064 | [+0.121326, +0.305326] |

The significant seed43-minus-seed42 contrast measures seed sensitivity; a random seed is not an actionable factor and cannot be promoted as a method.

Next action: Treat leader-grade experience structure as the main unresolved factor and build a coverage-preserving, event-binding semantic compression arm with an ordinary-simplification control.

Evidence JSON: `experiments/archive/initial_model_studies/data/event_binding_attribution_panel.json`
