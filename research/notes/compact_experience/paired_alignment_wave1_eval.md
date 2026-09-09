# paired alignment accounting audit — Paired-alignment Wave 1 fast evaluation

Summary JSON: `experiments/archive/compact_experience/data/paired_alignment_eval/paired_alignment_wave1_eval_summary.json`

This evaluates the critical Wave-1 contrast: ALIGNED and MISMATCHED have the same sentence inventory, same alternating packed structure, same word exposure, and differ in whether adjacent orig/simp text is meaning-matched. This is a fast mechanism screen, not official Overall: SuperGLUE and AoA are absent, most zero-shot columns use fast subsets, and COMPS uses `full_eval/comps`.

## chck_100M scores

| target | BLiMP | Supplement | EWoK | Entity | COMPS | GPIQA mean | Reading | equal-7 mean | weighted fast proxy | loss_last |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned__chck_100M | 64.120 | 60.800 | 51.180 | 27.740 | 51.420 | 37.520 | 7.160 | 42.849 | 32.264 | 1.946 |
| mismatched__chck_100M | 65.630 | 58.000 | 50.450 | 17.100 | 50.770 | 41.050 | 7.575 | 41.511 | 31.268 | 3.227 |
| initial_model_baseline__chck_100M | 67.340 | 65.200 | 49.640 | 21.240 | 53.110 | 36.120 | 7.330 | 42.854 | 32.272 | 2.609 |

## Contrasts

| contrast | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | equal-7 | weighted fast proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned_minus_mismatched | -1.510 | 2.800 | 0.730 | 10.640 | 0.650 | -3.530 | -0.415 | 1.338 | 0.996 |
| aligned_minus_initial_model_studies_baseline | -3.220 | -4.400 | 1.540 | 6.500 | -1.690 | 1.400 | -0.170 | -0.006 | -0.007 |
| mismatched_minus_initial_model_studies_baseline | -1.710 | -7.200 | 0.810 | -4.140 | -2.340 | 4.930 | 0.245 | -1.344 | -1.003 |

Weighted fast proxy = (3/28) * (BLiMP + Supplement + EWoK + Entity + COMPS + GlobalPIQA_mean) + (1/8) * Reading. It is a screening statistic only.

## Training exposure notes

- `aligned`: word_exposure=100000000, loss_first=9.793, loss_last=1.946, source_words_consumed={'epoch1::aligned_pair': 9999840, 'epoch2::aligned_pair': 9999840, 'epoch3::aligned_pair': 9999840, 'epoch4::aligned_pair': 9999840, 'epoch5::aligned_pair': 9999840, 'epoch6::aligned_pair': 9999840, 'epoch7::aligned_pair': 9999840, 'epoch8::aligned_pair': 9999840, 'epoch9::aligned_pair': 9999840, 'epoch10::aligned_pair': 9999840, 'epoch11::aligned_pair': 1600}
- `mismatched`: word_exposure=100000000, loss_first=9.790, loss_last=3.227, source_words_consumed={'epoch1::mismatched_pair': 9999840, 'epoch2::mismatched_pair': 9999840, 'epoch3::mismatched_pair': 9999840, 'epoch4::mismatched_pair': 9999840, 'epoch5::mismatched_pair': 9999840, 'epoch6::mismatched_pair': 9999840, 'epoch7::mismatched_pair': 9999840, 'epoch8::mismatched_pair': 9999840, 'epoch9::mismatched_pair': 9999840, 'epoch10::mismatched_pair': 9999840, 'epoch11::mismatched_pair': 1600}
- The ALIGNED and MISMATCHED base pools each contain 9,999,840 words; exact 100,000,000 exposure equals 10 full passes plus 1,600 words. This is acceptable for the matched mechanism contrast but should be rerun at exactly 99,998,400 exposure before treating it as a strict official candidate.
