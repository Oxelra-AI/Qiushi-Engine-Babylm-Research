# event binding timescale event-binding learning timescale

Decision: **TEN_MILLION_H2_SCREEN_PREMATURE_BUT_PROBE_NOT_DECISIVE_FOR_CAPABILITY**

| exposure | seed 42 effect (95% CI) | seed 43 effect (95% CI) |
|---:|---:|---:|
| 5M | +0.000037 [-0.000017, +0.000093] | -0.000077 [-0.000161, +0.000007] |
| 10M | +0.000065 [-0.000173, +0.000302] | +0.000026 [-0.000363, +0.000418] |
| 20M | -0.000488 [-0.001981, +0.001007] | -0.001074 [-0.004213, +0.002006] |
| 40M | +0.004705 [-0.006334, +0.015651] | -0.001284 [-0.012936, +0.010618] |
| 60M | +0.026086 [+0.007368, +0.044719] | +0.032670 [+0.001814, +0.063500] |
| 80M | +0.060696 [+0.037974, +0.084241] | +0.107567 [+0.071672, +0.144424] |
| 100M | +0.087790 [+0.062785, +0.113955] | +0.133062 [+0.096002, +0.171575] |

First positive-CI milestone by seed: {'wwm_seed42': 60, 'wwm_seed43': 60}

Within-seed Spearman correlations (effect levels / first differences):

- wwm_seed42 Entity: +0.964 / +0.429; available7: +0.857 / -0.543
- wwm_seed43 Entity: +0.607 / -0.086; available7: +0.643 / -0.543

Next action: Retract 10M-based mechanism closure, but do not optimize this probe directly; design future data screens around transfer-per-word and verify at the observed onset.

Evidence JSON: `experiments/archive/initial_model_studies/data/event_binding_timescale.json`
