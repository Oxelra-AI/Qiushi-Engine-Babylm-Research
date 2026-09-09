# route synthesis for independent_review held-out tokenizer seam/phase analysis

CPU-only measurement of old vs legal spatial repair route status tokenizers on official evaluation strings. It tests word-aligned segmentation mechanisms not measured by legal deficit representation vs candidate alignment aggregate tokens-per-word/support analyses. No model is trained and no eval text is used to construct training data.

## Tokenizers
- `old` len=16384: `experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M`
- `legal` len=16384: `experiments/archive/frontier_consolidation/data/compliant_tokenizer`

## Correlations with legal-spatial repair route status minus old score delta
Negative delta means the legal tokenizer endpoint lost. For bad metrics (lost seams, larger local token shifts), a representation explanation predicts negative correlation.

### BLiMP (n=67)
| metric | mean | Pearson | Spearman |
|---|---:|---:|---:|
| `mean_legal_minus_old_token_count_per_word` | 0.006211 | -0.0486 | -0.1642 |
| `mean_abs_token_count_delta_per_word` | 0.054227 | -0.0454 | -0.1103 |
| `token_count_changed_word_frac` | 0.049405 | -0.0339 | -0.1169 |
| `boundary_changed_word_frac` | 0.058240 | -0.0712 | -0.1325 |
| `mean_boundary_jaccard` | 0.948342 | 0.1062 | 0.1902 |
| `legal_minus_old_whole_word_frac` | -0.003064 | -0.0025 | 0.1565 |
| `mean_abs_token_start_shift` | 0.160194 | 0.1646 | 0.1301 |
| `p90_abs_token_start_shift` | 0.805970 | 0.1426 | 0.1190 |
| `p99_abs_token_start_shift` | 1.812985 | 0.0398 | -0.0177 |
| `std_token_start_shift` | 0.435431 | 0.1477 | 0.1043 |
| `suffix_legal_minus_old_preserve_rate` | 0.007313 | 0.0629 | 0.0299 |
| `suffix_legal_lost_old_seam_rate_over_candidates` | 0.009936 | -0.0487 | -0.0158 |
| `suffix_legal_lost_old_seam_rate_given_old` | 0.112372 | -0.1710 | -0.1603 |
| `suffix_legal_minus_old_whole_candidate_rate` | 0.010719 | -0.0628 | -0.0220 |
| `prefix_legal_minus_old_preserve_rate` | -0.001817 | -0.1648 | -0.1980 |
| `prefix_legal_lost_old_seam_rate_over_candidates` | 0.014318 | 0.2190 | 0.1646 |
| `reflexive_legal_minus_old_preserve_rate` | 0.000000 | — | — |
| `reflexive_legal_lost_old_seam_rate_over_candidates` | 0.000000 | — | — |
| `any_legal_minus_old_preserve_word_rate` | 0.008954 | 0.0501 | 0.0139 |
| `any_legal_lost_old_seam_word_rate` | 0.013083 | -0.0263 | -0.0242 |
| `any_legal_minus_old_whole_candidate_word_rate` | 0.007681 | -0.0220 | 0.0601 |

### Supplement (n=5)
| metric | mean | Pearson | Spearman |
|---|---:|---:|---:|
| `mean_legal_minus_old_token_count_per_word` | -0.002420 | -0.0756 | 0.3000 |
| `mean_abs_token_count_delta_per_word` | 0.012566 | 0.3110 | 0.0000 |
| `token_count_changed_word_frac` | 0.011940 | 0.3023 | 0.0000 |
| `boundary_changed_word_frac` | 0.014476 | 0.2604 | 0.0000 |
| `mean_boundary_jaccard` | 0.988096 | -0.3271 | 0.0000 |
| `legal_minus_old_whole_word_frac` | -0.000722 | 0.2627 | 0.6000 |
| `mean_abs_token_start_shift` | 0.278370 | -0.7434 | -0.6000 |
| `p90_abs_token_start_shift` | 0.800000 | -0.3027 | 0.0000 |
| `p99_abs_token_start_shift` | 1.200000 | 0.3134 | 0.3536 |
| `std_token_start_shift` | 0.450418 | -0.4497 | -0.5000 |
| `suffix_legal_minus_old_preserve_rate` | -0.000982 | -0.3134 | -0.3536 |
| `suffix_legal_lost_old_seam_rate_over_candidates` | 0.002387 | 0.3134 | 0.3536 |
| `suffix_legal_lost_old_seam_rate_given_old` | 0.072146 | 0.5179 | 0.8660 |
| `suffix_legal_minus_old_whole_candidate_rate` | -0.000721 | -0.0202 | 0.2236 |
| `prefix_legal_minus_old_preserve_rate` | -0.002501 | -0.5179 | -0.8660 |
| `prefix_legal_lost_old_seam_rate_over_candidates` | 0.005149 | 0.5179 | 0.8660 |
| `reflexive_legal_minus_old_preserve_rate` | 0.000000 | — | — |
| `reflexive_legal_lost_old_seam_rate_over_candidates` | 0.000000 | — | — |
| `any_legal_minus_old_preserve_word_rate` | -0.001351 | -0.3134 | -0.3536 |
| `any_legal_lost_old_seam_word_rate` | 0.003180 | 0.3134 | 0.3536 |
| `any_legal_minus_old_whole_candidate_word_rate` | -0.001631 | -0.0598 | 0.2236 |

### EWoK (n=11)
| metric | mean | Pearson | Spearman |
|---|---:|---:|---:|
| `mean_legal_minus_old_token_count_per_word` | 0.019981 | 0.2468 | 0.2636 |
| `mean_abs_token_count_delta_per_word` | 0.038406 | 0.2751 | 0.4273 |
| `token_count_changed_word_frac` | 0.034047 | 0.1938 | 0.3000 |
| `boundary_changed_word_frac` | 0.038213 | 0.0604 | -0.0455 |
| `mean_boundary_jaccard` | 0.966510 | -0.0378 | 0.0909 |
| `legal_minus_old_whole_word_frac` | -0.013913 | -0.1762 | -0.2727 |
| `mean_abs_token_start_shift` | 0.146499 | 0.3272 | 0.1091 |
| `p90_abs_token_start_shift` | 0.454545 | 0.3843 | 0.2887 |
| `p99_abs_token_start_shift` | 1.636364 | 0.1964 | 0.2072 |
| `std_token_start_shift` | 0.411770 | 0.2388 | 0.1091 |
| `suffix_legal_minus_old_preserve_rate` | 0.014215 | -0.5212 | 0.1526 |
| `suffix_legal_lost_old_seam_rate_over_candidates` | 0.004091 | -0.3121 | -0.2181 |
| `suffix_legal_lost_old_seam_rate_given_old` | 0.043038 | -0.7333 | -0.4091 |
| `suffix_legal_minus_old_whole_candidate_rate` | -0.033245 | 0.3067 | 0.1093 |
| `prefix_legal_minus_old_preserve_rate` | -0.009439 | -0.5149 | -0.3700 |
| `prefix_legal_lost_old_seam_rate_over_candidates` | 0.020803 | 0.1876 | 0.1348 |
| `reflexive_legal_minus_old_preserve_rate` | — | — | — |
| `reflexive_legal_lost_old_seam_rate_over_candidates` | — | — | — |
| `any_legal_minus_old_preserve_word_rate` | 0.014796 | -0.4845 | 0.2745 |
| `any_legal_lost_old_seam_word_rate` | 0.004425 | -0.3109 | -0.2527 |
| `any_legal_minus_old_whole_candidate_word_rate` | -0.024890 | 0.3262 | 0.1284 |

### ALL (n=83)
| metric | mean | Pearson | Spearman |
|---|---:|---:|---:|
| `mean_legal_minus_old_token_count_per_word` | 0.007516 | -0.0126 | -0.0846 |
| `mean_abs_token_count_delta_per_word` | 0.049621 | 0.0665 | -0.0163 |
| `token_count_changed_word_frac` | 0.045112 | 0.0657 | -0.0233 |
| `boundary_changed_word_frac` | 0.052950 | 0.0275 | -0.0410 |
| `mean_boundary_jaccard` | 0.953144 | -0.0065 | 0.0662 |
| `legal_minus_old_whole_word_frac` | -0.004360 | -0.0055 | 0.0680 |
| `mean_abs_token_start_shift` | 0.165498 | 0.1099 | 0.1330 |
| `p90_abs_token_start_shift` | 0.759036 | 0.1856 | 0.1364 |
| `p99_abs_token_start_shift` | 1.752651 | 0.1063 | 0.0834 |
| `std_token_start_shift` | 0.433198 | 0.1521 | 0.1341 |
| `suffix_legal_minus_old_preserve_rate` | 0.007728 | -0.1428 | 0.0679 |
| `suffix_legal_lost_old_seam_rate_over_candidates` | 0.008707 | -0.0388 | -0.0405 |
| `suffix_legal_lost_old_seam_rate_given_old` | 0.102945 | -0.1753 | -0.1854 |
| `suffix_legal_minus_old_whole_candidate_rate` | 0.004203 | 0.1239 | -0.0506 |
| `prefix_legal_minus_old_preserve_rate` | -0.002877 | -0.2395 | -0.2147 |
| `prefix_legal_lost_old_seam_rate_over_candidates` | 0.014859 | 0.1428 | 0.1406 |
| `reflexive_legal_minus_old_preserve_rate` | 0.000000 | — | — |
| `reflexive_legal_lost_old_seam_rate_over_candidates` | 0.000000 | — | — |
| `any_legal_minus_old_preserve_word_rate` | 0.009108 | -0.1306 | 0.0552 |
| `any_legal_lost_old_seam_word_rate` | 0.011339 | -0.0102 | -0.0454 |
| `any_legal_minus_old_whole_candidate_word_rate` | 0.002803 | 0.1350 | 0.0086 |

## Worst legal-tokenizer losses
| column | uid | Δ score | mean_abs_token_start_shift | p90_abs_token_start_shift | boundary_changed_word_frac | any_legal_lost_old_seam_word_rate | suffix_legal_lost_old_seam_rate_over_candidates | reflexive_legal_lost_old_seam_rate_over_candidates | any_legal_minus_old_whole_candidate_word_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | principle_A_reconstruction | -20.89 | 0.0369 | 0.0000 | 0.0466 | 0.0072 | 0.0053 | 0.0000 | 0.0041 |
| EWoK | physical-dynamics | -19.50 | 0.0282 | 0.0000 | 0.0387 | 0.0000 | 0.0000 | — | -0.1600 |
| BLiMP | wh_questions_object_gap | -17.34 | 0.2088 | 1.0000 | 0.0588 | 0.0213 | 0.0165 | — | 0.0279 |
| EWoK | social-properties | -13.75 | 0.0910 | 0.0000 | 0.0371 | 0.0111 | 0.0106 | — | -0.0055 |
| Supplement | qa_congruence_easy | -12.50 | 0.4304 | 1.0000 | 0.0083 | 0.0000 | 0.0000 | — | 0.0000 |
| BLiMP | animate_subject_trans | -11.91 | 0.1145 | 1.0000 | 0.0700 | 0.0127 | 0.0072 | — | 0.0082 |
| BLiMP | regular_plural_subject_verb_agreement_1 | -11.13 | 0.1285 | 1.0000 | 0.0744 | 0.0086 | 0.0054 | — | 0.0101 |
| BLiMP | tough_vs_raising_1 | -10.97 | 0.1630 | 1.0000 | 0.0547 | 0.0592 | 0.0603 | — | 0.0292 |
| BLiMP | anaphor_gender_agreement | -10.71 | 0.1852 | 1.0000 | 0.0751 | 0.0082 | 0.0068 | 0.0000 | 0.0071 |
| BLiMP | matrix_question_npi_licensor_present | -9.47 | 0.1631 | 1.0000 | 0.0594 | 0.0106 | 0.0050 | — | 0.0051 |
| EWoK | material-dynamics | -8.83 | 0.0508 | 0.0000 | 0.0455 | 0.0287 | 0.0248 | — | 0.0430 |
| BLiMP | anaphor_number_agreement | -7.41 | 0.1006 | 0.0000 | 0.0518 | 0.0111 | 0.0057 | 0.0000 | 0.0131 |
| BLiMP | npi_present_1 | -7.15 | 0.1007 | 0.0000 | 0.0459 | 0.0119 | 0.0083 | — | -0.0035 |
| BLiMP | existential_there_quantifiers_2 | -6.58 | 0.1007 | 0.0000 | 0.0596 | 0.0113 | 0.0092 | — | -0.0345 |
| BLiMP | wh_vs_that_no_gap_long_distance | -6.51 | 0.2406 | 1.0000 | 0.0525 | 0.0168 | 0.0126 | — | 0.0134 |
| BLiMP | sentential_subject_island | -5.83 | 0.0694 | 0.0000 | 0.0617 | 0.0102 | 0.0092 | — | -0.0126 |
| EWoK | quantitative-properties | -5.80 | 0.2853 | 1.0000 | 0.0500 | 0.0000 | 0.0000 | — | 0.0451 |
| BLiMP | principle_A_domain_1 | -5.47 | 0.2210 | 1.0000 | 0.0530 | 0.0090 | 0.0065 | 0.0000 | 0.0076 |
| BLiMP | wh_island | -5.10 | 0.1223 | 1.0000 | 0.0473 | 0.0315 | 0.0294 | — | 0.0263 |
| BLiMP | principle_A_domain_2 | -4.81 | 0.2002 | 1.0000 | 0.0608 | 0.0058 | 0.0039 | 0.0000 | 0.0027 |

## Largest legal-tokenizer gains
| column | uid | Δ score | mean_abs_token_start_shift | p90_abs_token_start_shift | boundary_changed_word_frac | any_legal_lost_old_seam_word_rate | suffix_legal_lost_old_seam_rate_over_candidates | reflexive_legal_lost_old_seam_rate_over_candidates | any_legal_minus_old_whole_candidate_word_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | only_npi_licensor_present | 24.26 | 0.1221 | 1.0000 | 0.0478 | 0.0093 | 0.0060 | — | -0.0037 |
| BLiMP | left_branch_island_echo_question | 19.96 | 0.2047 | 1.0000 | 0.0797 | 0.0102 | 0.0061 | — | -0.0015 |
| BLiMP | tough_vs_raising_2 | 12.39 | 0.2356 | 1.0000 | 0.0633 | 0.0421 | 0.0342 | — | 0.0292 |
| BLiMP | superlative_quantifiers_2 | 9.44 | 0.0959 | 0.0000 | 0.0335 | 0.0084 | 0.0049 | — | 0.0174 |
| BLiMP | ellipsis_n_bar_1 | 7.48 | 0.3110 | 1.0000 | 0.0494 | 0.0089 | 0.0063 | — | 0.0111 |
| BLiMP | adjunct_island | 7.22 | 0.1677 | 1.0000 | 0.0730 | 0.0295 | 0.0267 | — | 0.0107 |
| BLiMP | complex_NP_island | 6.97 | 0.1924 | 1.0000 | 0.0567 | 0.0100 | 0.0063 | — | 0.0220 |
| EWoK | social-interactions | 5.78 | 0.2362 | 1.0000 | 0.0617 | 0.0060 | 0.0058 | — | 0.0000 |
| BLiMP | existential_there_subject_raising | 5.30 | 0.1087 | 1.0000 | 0.0496 | 0.0223 | 0.0202 | — | -0.0155 |
| BLiMP | wh_vs_that_with_gap_long_distance | 5.16 | 0.2153 | 1.0000 | 0.0448 | 0.0187 | 0.0122 | — | 0.0130 |
| EWoK | material-properties | 4.97 | 0.0084 | 0.0000 | 0.0331 | 0.0000 | 0.0000 | — | 0.0000 |
| BLiMP | determiner_noun_agreement_irregular_2 | 4.52 | 0.1400 | 1.0000 | 0.0559 | 0.0112 | 0.0071 | — | 0.0162 |

## Direct reading
- ALL lost-seam word correlation: Pearson -0.0102, Spearman -0.0454.
- BLiMP suffix lost-seam correlation: Pearson -0.0487, Spearman -0.0158.
- EWoK local phase-shift correlation: Pearson 0.3272, Spearman 0.1091.

Interpretation must be made from the sign and specificity: a strong representation-seam mechanism needs lost seams/phase shifts to be larger in the losing UIDs than in gaining UIDs, not merely nonzero. If correlations are weak or wrong-signed, this particular tokenizer-fit mechanism is not the next expensive route.

JSON: `experiments/archive/frontier_consolidation/data/eval_tokenizer_seam_phase_analysis/eval_tokenizer_seam_phase_analysis.json`
CSV: `experiments/archive/frontier_consolidation/data/eval_tokenizer_seam_phase_analysis/per_uid_seam_phase_metrics.csv`
