# expensive work score thresholds support-floor substrate alignment

CPU-only synthesis of existing tokenizer/corpus measurements and existing UID score deltas; no training or model evaluation.

Does the minfreq50 support-floored legal tokenizer mechanically align with the UID families where the spatial repair route status legal endpoint lost relative to the old non-submittable tokenizer reference?

## Global support-floor movement
- minfreq50 global raw tokens/word delta vs spatial repair route status: -0.018092; visible tokens/word delta -0.014199.
- over-256 rows delta -1021; truncated tokens delta -38930.
- expected target-token ratio 0.990067; expected selected-group ratio 1.002026.
- same-shape initialization match after copy: 1.0000; ordinary same-seed match 0.0000.

## Correlation with spatial repair route status legal loss
Negative legal delta means the legal spatial repair route status endpoint lost against the old non-submittable reference. Positive deficit magnitude is the lost amount clipped at zero.
| subset | n | metric | mean | Pearson(legal delta, metric) | Pearson(deficit magnitude, metric) |
|---|---:|---|---:|---:|---:|
| ALL | 83 | `minfreq50_minus_step35_tpw` | -0.036351 | 0.0179 | 0.0119 |
| ALL | 83 | `minfreq50_token_reduction_pct_vs_step35` | 0.025692 | 0.0109 | -0.0274 |
| ALL | 83 | `minfreq50_closes_step35_old_tpw_gap_frac` | 8.261615 | -0.0696 | 0.0674 |
| ALL | 83 | `minfreq50_minus_step35_support_log_gain` | -0.039362 | 0.0974 | -0.0983 |
| ALL | 83 | `minfreq50_minus_step35_frac_lt50_delta` | 0.004827 | -0.1149 | 0.1128 |
| ALL | 83 | `minfreq50_minus_step35_frac_lt100_delta` | 0.030022 | -0.1285 | 0.1174 |
| ALL | 83 | `minfreq50_minus_step35_vs_old_token_multiset_jaccard` | -0.013737 | -0.1031 | 0.1874 |
| ALL | 83 | `minfreq50_minus_step35_vs_old_b_not_in_a_frac` | -0.003243 | 0.0097 | -0.0520 |
| ALL | 83 | `minfreq50_minus_step35_eval_tokens` | -501.614458 | -0.1334 | 0.1933 |
| ALL | 83 | `minfreq50_eval_token_reduction_pct` | 0.022988 | 0.0816 | -0.0929 |
| BLiMP | 67 | `minfreq50_minus_step35_tpw` | -0.041183 | 0.0859 | -0.0176 |
| BLiMP | 67 | `minfreq50_token_reduction_pct_vs_step35` | 0.028939 | -0.0247 | -0.0189 |
| BLiMP | 67 | `minfreq50_closes_step35_old_tpw_gap_frac` | 10.181822 | -0.0845 | 0.0889 |
| BLiMP | 67 | `minfreq50_minus_step35_support_log_gain` | -0.030862 | 0.0024 | 0.0306 |
| BLiMP | 67 | `minfreq50_minus_step35_frac_lt50_delta` | 0.002696 | -0.1821 | 0.1008 |
| BLiMP | 67 | `minfreq50_minus_step35_frac_lt100_delta` | 0.031071 | -0.0909 | 0.0507 |
| BLiMP | 67 | `minfreq50_minus_step35_vs_old_token_multiset_jaccard` | -0.013018 | -0.0955 | 0.1948 |
| BLiMP | 67 | `minfreq50_minus_step35_vs_old_b_not_in_a_frac` | -0.006376 | 0.1047 | -0.1861 |
| BLiMP | 67 | `minfreq50_minus_step35_eval_tokens` | -536.283582 | -0.0853 | 0.1041 |
| BLiMP | 67 | `minfreq50_eval_token_reduction_pct` | 0.028939 | -0.0247 | -0.0189 |
| Supplement | 5 | `minfreq50_minus_step35_tpw` | -0.008738 | 0.0122 | 0.0569 |
| Supplement | 5 | `minfreq50_token_reduction_pct_vs_step35` | 0.006586 | 0.1070 | -0.1647 |
| Supplement | 5 | `minfreq50_closes_step35_old_tpw_gap_frac` | -2.966778 | -0.4060 | 0.4425 |
| Supplement | 5 | `minfreq50_minus_step35_support_log_gain` | -0.190683 | 0.6796 | -0.6843 |
| Supplement | 5 | `minfreq50_minus_step35_frac_lt50_delta` | 0.043038 | -0.6794 | 0.6797 |
| Supplement | 5 | `minfreq50_minus_step35_frac_lt100_delta` | 0.047659 | -0.7103 | 0.7054 |
| Supplement | 5 | `minfreq50_minus_step35_vs_old_token_multiset_jaccard` | -0.037840 | 0.6772 | -0.6779 |
| Supplement | 5 | `minfreq50_minus_step35_vs_old_b_not_in_a_frac` | 0.038226 | -0.6591 | 0.6631 |
| Supplement | 5 | `minfreq50_minus_step35_eval_tokens` | -270.600000 | -0.2770 | 0.3089 |
| Supplement | 5 | `minfreq50_eval_token_reduction_pct` | -0.038300 | 0.6385 | -0.6460 |
| EWoK | 11 | `minfreq50_minus_step35_tpw` | -0.019472 | 0.3987 | -0.3451 |
| EWoK | 11 | `minfreq50_token_reduction_pct_vs_step35` | 0.014603 | -0.3857 | 0.3232 |
| EWoK | 11 | `minfreq50_closes_step35_old_tpw_gap_frac` | 1.010419 | -0.0709 | -0.0284 |
| EWoK | 11 | `minfreq50_minus_step35_support_log_gain` | -0.022356 | -0.1882 | 0.2056 |
| EWoK | 11 | `minfreq50_minus_step35_frac_lt50_delta` | 0.000440 | -0.1068 | 0.1262 |
| EWoK | 11 | `minfreq50_minus_step35_frac_lt100_delta` | 0.015617 | -0.3396 | 0.2918 |
| EWoK | 11 | `minfreq50_minus_step35_vs_old_token_multiset_jaccard` | -0.007155 | -0.4504 | 0.5072 |
| EWoK | 11 | `minfreq50_minus_step35_vs_old_b_not_in_a_frac` | -0.003010 | 0.5481 | -0.5849 |
| EWoK | 11 | `minfreq50_minus_step35_eval_tokens` | -395.454545 | -0.1956 | 0.3066 |
| EWoK | 11 | `minfreq50_eval_token_reduction_pct` | 0.014603 | -0.3857 | 0.3232 |
| Worst20LegalLosses | 20 | `minfreq50_minus_step35_tpw` | -0.037156 | -0.3163 | 0.3163 |
| Worst20LegalLosses | 20 | `minfreq50_token_reduction_pct_vs_step35` | 0.026052 | 0.3448 | -0.3448 |
| Worst20LegalLosses | 20 | `minfreq50_closes_step35_old_tpw_gap_frac` | 7.765266 | -0.0596 | 0.0596 |
| Worst20LegalLosses | 20 | `minfreq50_minus_step35_support_log_gain` | -0.043533 | 0.0764 | -0.0764 |
| Worst20LegalLosses | 20 | `minfreq50_minus_step35_frac_lt50_delta` | 0.005941 | -0.1309 | 0.1309 |
| Worst20LegalLosses | 20 | `minfreq50_minus_step35_frac_lt100_delta` | 0.032468 | 0.0810 | -0.0810 |
| Worst20LegalLosses | 20 | `minfreq50_minus_step35_vs_old_token_multiset_jaccard` | -0.009184 | -0.2587 | 0.2587 |
| Worst20LegalLosses | 20 | `minfreq50_minus_step35_vs_old_b_not_in_a_frac` | -0.005272 | 0.0306 | -0.0306 |
| Worst20LegalLosses | 20 | `minfreq50_minus_step35_eval_tokens` | -427.450000 | -0.4293 | 0.4293 |
| Worst20LegalLosses | 20 | `minfreq50_eval_token_reduction_pct` | 0.021913 | 0.2167 | -0.2167 |

## Group means
| group | n | mean legal delta | mean minfreq50-spatial repair route status tpw | mean support log gain | mean frac<50 delta | mean Jaccard-vs-old gain |
|---|---:|---:|---:|---:|---:|---:|
| worst20 | 20 | -10.0830 | -0.037156 | -0.043533 | 0.005941 | -0.009184 |
| other_losses | 30 | -2.3763 | -0.038437 | -0.042743 | 0.005578 | -0.016876 |
| nonloss | 33 | 4.8342 | -0.033967 | -0.033762 | 0.003469 | -0.013642 |
| ewok_relation | 7 | -3.1586 | -0.018665 | -0.025062 | 0.000325 | -0.013200 |
| ewok_property | 4 | -3.4950 | -0.020884 | -0.017620 | 0.000640 | 0.003424 |

## Twenty largest spatial repair route status legal losses
| column | uid | legal delta | minfreq50-spatial repair route status tpw | support log gain | frac<50 delta | Jaccard-vs-old gain | new-vs-old frac delta |
|---|---|---:|---:|---:|---:|---:|---:|
| BLiMP | principle_A_reconstruction | -20.89 | -0.031650 | -0.012828 | 0.002504 | -0.000727 | -0.009674 |
| EWoK | physical-dynamics | -19.50 | -0.024648 | -0.015898 | 0.000000 | 0.044678 | -0.032983 |
| BLiMP | wh_questions_object_gap | -17.34 | -0.037796 | -0.032148 | 0.006413 | -0.014948 | -0.004545 |
| EWoK | social-properties | -13.75 | -0.030628 | -0.024918 | 0.002174 | -0.008769 | -0.005449 |
| Supplement | qa_congruence_easy | -12.50 | -0.008349 | -0.343602 | 0.077108 | -0.063736 | 0.070617 |
| BLiMP | animate_subject_trans | -11.91 | -0.042591 | -0.032941 | 0.003386 | -0.005207 | -0.010707 |
| BLiMP | regular_plural_subject_verb_agreement_1 | -11.13 | -0.054621 | -0.045188 | 0.002181 | -0.008998 | -0.011667 |
| BLiMP | tough_vs_raising_1 | -10.97 | -0.031099 | -0.012422 | 0.000656 | -0.004094 | -0.008644 |
| BLiMP | anaphor_gender_agreement | -10.71 | -0.050066 | -0.038867 | 0.003362 | -0.005361 | -0.011887 |
| BLiMP | matrix_question_npi_licensor_present | -9.47 | -0.038462 | -0.035058 | 0.002479 | 0.009867 | -0.019002 |
| EWoK | material-dynamics | -8.83 | -0.011149 | -0.012997 | 0.000007 | -0.020798 | 0.007996 |
| BLiMP | anaphor_number_agreement | -7.41 | -0.036546 | -0.029145 | 0.002089 | -0.009477 | -0.006614 |
| BLiMP | npi_present_1 | -7.15 | -0.036354 | -0.026653 | 0.002001 | -0.012421 | -0.006172 |
| BLiMP | existential_there_quantifiers_2 | -6.58 | -0.058480 | -0.043458 | 0.001352 | -0.018335 | -0.008877 |
| BLiMP | wh_vs_that_no_gap_long_distance | -6.51 | -0.041195 | -0.028536 | 0.002101 | -0.024212 | -0.001169 |
| BLiMP | sentential_subject_island | -5.83 | -0.063648 | -0.051747 | 0.001822 | -0.033534 | -0.000769 |
| EWoK | quantitative-properties | -5.80 | -0.026345 | -0.001498 | 0.000014 | 0.024441 | -0.023005 |
| BLiMP | principle_A_domain_1 | -5.47 | -0.037950 | -0.024022 | 0.002823 | -0.001782 | -0.012235 |
| BLiMP | wh_island | -5.10 | -0.039714 | -0.025466 | 0.002618 | -0.032443 | 0.004498 |
| BLiMP | principle_A_domain_2 | -4.81 | -0.041824 | -0.033276 | 0.003730 | 0.002174 | -0.015148 |

## Scientific reading
- The support-floor route remains a broad legal representation experiment, not a targeted repair derived from the worst UID labels. Its strongest prior value is small but clean: fewer fragmented tokens, lower truncation, lower target-token burden, and much less unsupported tail than large 40k while preserving compact-view reinvestment.
- If its future 70M/80M score surface helps, the result should be read as a representation/support/segmentation package. If it hurts Entity or GlobalPIQA like the larger 40k route, the support-floor route should stop rather than be expanded by vocabulary size alone.
- Existing eval-string correlations here should not be used to tune the tokenizer; they only explain what a fixed, already-trained legal tokenizer changes.

JSON: `experiments/archive/frontier_consolidation/data/supportfloor_substrate_alignment/supportfloor_substrate_alignment.json`
CSV: `experiments/archive/frontier_consolidation/data/supportfloor_substrate_alignment/supportfloor_substrate_alignment_per_uid.csv`
