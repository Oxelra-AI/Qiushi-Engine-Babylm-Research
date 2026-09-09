# compact core full eval projection thresholds trainer-exact token and WWM-group exposure measurement

Summary JSON: `experiments/archive/frontier_consolidation/data/trainer_exact_token_exposure_measurement/trainer_exact_token_exposure_summary.json`
Tokenizer: `experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model`; seq length 256; WWM probability 0.15; ten-pass exposure is exact 10x.

## Pool totals

| arm | words | rows | visible candidate tokens | visible WWM groups | over-seq256 rows |
|---|---:|---:|---:|---:|---:|
| near_repeat | 10000000 | 64833 | 14367015 | 9787644 | 15840 |
| near_view | 10000000 | 64833 | 14365915 | 9787699 | 15834 |
| compact_repeat_core | 10000000 | 64683 | 14371158 | 9785594 | 15990 |
| compact_view_core | 10000000 | 64683 | 14391511 | 9785327 | 16026 |
| compact_view_reinvest | 10000000 | 64740 | 14391548 | 9787279 | 15883 |
| compact_repeat_reinvest | 10000000 | 64740 | 14366888 | 9787586 | 15841 |
| lengthmatched_compact_core | 10000000 | 64683 | 14401336 | 9784046 | 16147 |

## Key contrasts

- near_view_minus_near_repeat: candidate tokens -1100 (-0.008%), WWM groups 55 (+0.001%), over-seq256 rows -6.
- compact_view_core_minus_compact_repeat_core: candidate tokens 20353 (+0.142%), WWM groups -267 (-0.003%), over-seq256 rows 36.
- compact_view_reinvest_minus_compact_view_core: candidate tokens 37 (+0.000%), WWM groups 1952 (+0.020%), over-seq256 rows -143.
- compact_repeat_reinvest_minus_compact_repeat_core: candidate tokens -4270 (-0.030%), WWM groups 1992 (+0.020%), over-seq256 rows -149.
- compact_view_core_minus_lengthmatched_compact_core: candidate tokens -9825 (-0.068%), WWM groups 1281 (+0.013%), over-seq256 rows -121.

## Interpretation

Use this measurement to separate semantic/view effects from raw token or WWM-group exposure differences when reading the density eval repair compact-core task gains.
