# masking curriculum 100M eval — 100M masking-curriculum trajectory mechanism note

Trajectory summary JSON: `experiments/archive/compact_experience/data/curriculum_100M_eval/curriculum_trajectory_eval_summary.json`
Final-checkpoint summary JSON: `experiments/archive/compact_experience/data/curriculum_100M_eval/curriculum_100M_eval_summary.json`
Compact CSV: `experiments/archive/compact_experience/data/curriculum_100M_eval/trajectory_compact_table.csv`

## Absolute trajectory scores

| arm | checkpoint | mode | eff mask | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | proxy |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wwm_fixed | chck_60M | wwm | 0.150 | 59.860 | 60.400 | 48.910 | 17.190 | 49.720 | 34.195 | 7.350 | 29.877 |
| wwm_fixed | chck_70M | wwm | 0.150 | 61.930 | 62.400 | 49.360 | 17.090 | 50.930 | 34.710 | 7.605 | 30.567 |
| wwm_fixed | chck_80M | wwm | 0.150 | 63.190 | 64.000 | 48.730 | 19.050 | 50.750 | 33.725 | 7.685 | 30.901 |
| wwm_fixed | chck_100M | wwm | 0.149 | 63.490 | 66.000 | 49.450 | 20.560 | 50.730 | 32.740 | 7.445 | 31.249 |
| wwm_to_token | chck_60M | wwm | 0.150 | 59.840 | 60.400 | 48.910 | 17.190 | 49.720 | 34.195 | 7.350 | 29.875 |
| wwm_to_token | chck_70M | wwm | 0.150 | 61.930 | 62.400 | 49.360 | 17.090 | 50.930 | 34.710 | 7.605 | 30.567 |
| wwm_to_token | chck_80M | token | 0.150 | 63.340 | 59.200 | 48.550 | 16.590 | 50.920 | 35.150 | 7.165 | 30.226 |
| wwm_to_token | chck_100M | token | 0.150 | 64.430 | 60.800 | 50.550 | 18.660 | 50.690 | 36.150 | 6.985 | 31.010 |
| amlm_hard_switch | chck_60M | wwm | 0.239 | 62.900 | 63.600 | 48.820 | 19.370 | 50.930 | 34.165 | 7.640 | 30.932 |
| amlm_hard_switch | chck_70M | wwm | 0.212 | 63.370 | 61.600 | 47.820 | 18.930 | 50.940 | 39.020 | 7.155 | 31.074 |
| amlm_hard_switch | chck_80M | token | 0.185 | 65.010 | 58.400 | 47.910 | 18.460 | 51.200 | 36.580 | 6.950 | 30.607 |
| amlm_hard_switch | chck_100M | token | 0.147 | 65.970 | 60.400 | 49.000 | 20.350 | 51.170 | 33.635 | 6.720 | 30.896 |

## Contrasts versus fixed WWM by checkpoint

| checkpoint | wwm→token proxy Δ | wwm→token Supp Δ | wwm→token Entity Δ | wwm→token GPIQA Δ | AMLM+switch proxy Δ | AMLM+switch Supp Δ | AMLM+switch Entity Δ | AMLM+switch GPIQA Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chck_60M | -0.002 | 0.000 | 0.000 | 0.000 | 1.055 | 3.200 | 2.180 | -0.030 |
| chck_70M | 0.000 | 0.000 | 0.000 | 0.000 | 0.507 | -0.800 | 1.840 | 4.310 |
| chck_80M | -0.675 | -4.800 | -2.460 | 1.425 | -0.294 | -5.600 | -0.590 | 2.855 |
| chck_100M | -0.239 | -5.200 | -1.900 | 3.410 | -0.353 | -5.600 | -0.210 | 0.895 |

## Final 100M contrast context

- `wwm_to_token_minus_wwm_fixed`: weighted fast proxy Δ -0.239
- `amlm_hard_minus_wwm_fixed`: weighted fast proxy Δ -0.124
- `amlm_hard_switch_minus_wwm_fixed`: weighted fast proxy Δ -0.353
- `amlm_hard_switch_minus_amlm_hard`: weighted fast proxy Δ -0.228
- `amlm_hard_minus_wwm_to_token`: weighted fast proxy Δ 0.114
- `amlm_hard_switch_minus_wwm_to_token`: weighted fast proxy Δ -0.114

## Mechanistic reading

- `chck_60M`: WWM→token proxy Δ -0.002, Supplement Δ 0.000, Entity Δ 0.000, GPIQA Δ 0.000; AMLM+switch proxy Δ 1.055, Supplement Δ 3.200, Entity Δ 2.180, GPIQA Δ -0.030.
- `chck_70M`: WWM→token proxy Δ 0.000, Supplement Δ 0.000, Entity Δ 0.000, GPIQA Δ 0.000; AMLM+switch proxy Δ 0.507, Supplement Δ -0.800, Entity Δ 1.840, GPIQA Δ 4.310.
- `chck_80M`: WWM→token proxy Δ -0.675, Supplement Δ -4.800, Entity Δ -2.460, GPIQA Δ 1.425; AMLM+switch proxy Δ -0.294, Supplement Δ -5.600, Entity Δ -0.590, GPIQA Δ 2.855.
- `chck_100M`: WWM→token proxy Δ -0.239, Supplement Δ -5.200, Entity Δ -1.900, GPIQA Δ 3.410; AMLM+switch proxy Δ -0.353, Supplement Δ -5.600, Entity Δ -0.210, GPIQA Δ 0.895.

The WWM→token arm is exactly the same training process as fixed WWM through `chck_70M` under this script: same seed, data order, sequence schedule, masking mode, and masking probability. Its pre-switch scores match fixed WWM essentially exactly, so the causal post-switch comparison begins at `chck_80M` and `chck_100M`. After switching to token-level masking, WWM→token gains BLiMP and GlobalPIQA but loses Supplement, Entity, and Reading enough that the weighted fast proxy stays below fixed WWM. AMLM+switch shows the same broad pattern, with difficulty weighting partly protecting Entity at 100M but not repairing the Supplement/Reading penalty.

This supports a trade-off mechanism rather than a uniform curriculum improvement: token-level prediction and hard-token emphasis strengthen some local lexical/syntactic or plausibility signals, but they appear to reduce the whole-word/phrase-level structural pressure that benefits Supplement and reading-style measures under the current baseline16k AdamW DeBERTa recipe. The next useful construction should not blindly copy the leaderboard switch. It should either replicate fixed versus switch with additional seeds to quantify the effect, or build a gentler/conditional granularity transition that preserves WWM pressure on structure-sensitive examples while adding token-level pressure where BLiMP/GPIQA benefit.
