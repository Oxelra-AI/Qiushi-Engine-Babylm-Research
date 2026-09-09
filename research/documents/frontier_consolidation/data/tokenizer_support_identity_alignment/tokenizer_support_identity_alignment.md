# legal deficit representation vs candidate alignment tokenizer support and identity alignment

CPU-only analysis over existing tokenizers and the allowed 10M pool. Evaluation text is used only to interpret already-trained tokenizers and already-scored endpoint deltas.

## Pool support summary

| tokenizer | vocab | tok/word | used non-special | median count | p10 count | frac mass <50 | frac vocab <50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_inherited_16k_nonlegal | 16384 | 1.4789 | 16122 | 145.0 | 42.0 | 0.0037 | 0.1275 |
| legal_step35_16k | 16384 | 1.4665 | 16337 | 140 | 63.0 | 0.0021 | 0.0803 |
| legal_bytealpha_16k | 16384 | 1.4669 | 16266 | 140.0 | 64.0 | 0.0020 | 0.0799 |
| legal_24k | 24576 | 1.4281 | 24366 | 73.0 | 34.0 | 0.0183 | 0.3236 |
| legal_32k | 32768 | 1.4066 | 32350 | 46.0 | 20.0 | 0.0336 | 0.5313 |
| legal_40k | 40000 | 1.3943 | 39320 | 33.0 | 14.0 | 0.0413 | 0.6321 |
| legal_40k_minfreq25 | 29529 | 1.4139 | 29185 | 54 | 25.0 | 0.0286 | 0.4658 |
| legal_40k_minfreq50 | 19609 | 1.4484 | 19454 | 105.0 | 49.0 | 0.0031 | 0.1003 |

## Selected correlations with legal spatial repair route status score delta

Negative score delta means legal spatial repair route status lost vs the old non-submittable reference. If low support explains losses, `frac_pool_count_lt_*` should correlate negatively and log-count should correlate positively with score delta. If token identity shift explains losses, legal-vs-old new-token fraction should correlate negatively.

| column | n | metric | mean | Pearson(delta,metric) | Spearman |
|---|---:|---|---:|---:|---:|
| BLiMP | 67 | `legal_step35_16k_mean_log10_pool_count_plus1` | 3.6539 | -0.1440 | 0.0525 |
| BLiMP | 67 | `legal_step35_16k_frac_pool_count_lt_50` | 0.0036 | 0.0473 | 0.0949 |
| BLiMP | 67 | `legal_step35_16k_frac_pool_count_lt_100` | 0.0563 | -0.0117 | -0.0143 |
| BLiMP | 67 | `legal_step35_16k_vs_old_b_not_in_a_frac` | 0.0653 | -0.0423 | -0.1409 |
| BLiMP | 67 | `legal_step35_16k_vs_old_token_multiset_jaccard` | 0.8815 | 0.0435 | 0.1265 |
| BLiMP | 67 | `legal_40k_frac_pool_count_lt_50` | 0.0862 | 0.0825 | 0.0456 |
| BLiMP | 67 | `legal_40k_vs_old_b_not_in_a_frac` | 0.0960 | 0.0401 | 0.0106 |
| BLiMP | 67 | `legal_40k_vs_step35_b_not_in_a_frac` | 0.0949 | 0.0434 | -0.0303 |
| BLiMP | 67 | `legal_40k_minfreq50_frac_pool_count_lt_50` | 0.0063 | -0.0132 | 0.0338 |
| Supplement | 5 | `legal_step35_16k_mean_log10_pool_count_plus1` | 4.2065 | -0.3345 | 0.0000 |
| Supplement | 5 | `legal_step35_16k_frac_pool_count_lt_50` | 0.0014 | -0.5259 | -0.4000 |
| Supplement | 5 | `legal_step35_16k_frac_pool_count_lt_100` | 0.0147 | 0.3014 | -0.1000 |
| Supplement | 5 | `legal_step35_16k_vs_old_b_not_in_a_frac` | 0.0622 | -0.7499 | -0.5000 |
| Supplement | 5 | `legal_step35_16k_vs_old_token_multiset_jaccard` | 0.8828 | 0.7451 | 0.5000 |
| Supplement | 5 | `legal_40k_frac_pool_count_lt_50` | 0.0660 | -0.7734 | -0.5000 |
| Supplement | 5 | `legal_40k_vs_old_b_not_in_a_frac` | 0.1123 | -0.7364 | -0.5000 |
| Supplement | 5 | `legal_40k_vs_step35_b_not_in_a_frac` | 0.0695 | -0.7811 | -0.8000 |
| Supplement | 5 | `legal_40k_minfreq50_frac_pool_count_lt_50` | 0.0445 | -0.6898 | -0.6000 |
| EWoK | 11 | `legal_step35_16k_mean_log10_pool_count_plus1` | 3.9191 | -0.0564 | -0.0182 |
| EWoK | 11 | `legal_step35_16k_frac_pool_count_lt_50` | 0.0017 | 0.3799 | 0.1388 |
| EWoK | 11 | `legal_step35_16k_frac_pool_count_lt_100` | 0.0654 | -0.0094 | -0.0091 |
| EWoK | 11 | `legal_step35_16k_vs_old_b_not_in_a_frac` | 0.0514 | 0.0995 | 0.1545 |
| EWoK | 11 | `legal_step35_16k_vs_old_token_multiset_jaccard` | 0.9156 | -0.0421 | 0.0091 |
| EWoK | 11 | `legal_40k_frac_pool_count_lt_50` | 0.0833 | 0.2522 | 0.2364 |
| EWoK | 11 | `legal_40k_vs_old_b_not_in_a_frac` | 0.0637 | 0.2074 | 0.0818 |
| EWoK | 11 | `legal_40k_vs_step35_b_not_in_a_frac` | 0.0732 | 0.2382 | 0.0182 |
| EWoK | 11 | `legal_40k_minfreq50_frac_pool_count_lt_50` | 0.0022 | 0.3281 | 0.0191 |
| ALL | 83 | `legal_step35_16k_mean_log10_pool_count_plus1` | 3.7223 | -0.1541 | 0.0077 |
| ALL | 83 | `legal_step35_16k_frac_pool_count_lt_50` | 0.0032 | 0.1042 | 0.1138 |
| ALL | 83 | `legal_step35_16k_frac_pool_count_lt_100` | 0.0550 | -0.0010 | -0.0094 |
| ALL | 83 | `legal_step35_16k_vs_old_b_not_in_a_frac` | 0.0632 | -0.0413 | -0.1319 |
| ALL | 83 | `legal_step35_16k_vs_old_token_multiset_jaccard` | 0.8861 | 0.0411 | 0.1275 |
| ALL | 83 | `legal_40k_frac_pool_count_lt_50` | 0.0846 | 0.0935 | -0.0133 |
| ALL | 83 | `legal_40k_vs_old_b_not_in_a_frac` | 0.0927 | 0.0214 | 0.0017 |
| ALL | 83 | `legal_40k_vs_step35_b_not_in_a_frac` | 0.0905 | 0.0859 | 0.0112 |
| ALL | 83 | `legal_40k_minfreq50_frac_pool_count_lt_50` | 0.0080 | -0.0862 | 0.0612 |

## Focus rows

| column | uid | Δ score | spatial repair route status frac<50 | spatial repair route status new-vs-old frac | spatial repair route status-old Jaccard | 40k frac<50 | 40k new-vs-old frac | 40k-vs-spatial repair route status new frac | minfreq50 frac<50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | principle_A_reconstruction | -20.89 | 0.0006 | 0.0510 | 0.9075 | 0.0554 | 0.0631 | 0.0626 | 0.0031 |
| BLiMP | wh_questions_object_gap | -17.34 | 0.0056 | 0.0670 | 0.8750 | 0.0917 | 0.1075 | 0.1010 | 0.0120 |
| BLiMP | animate_subject_trans | -11.91 | 0.0023 | 0.0759 | 0.8646 | 0.0863 | 0.1015 | 0.1005 | 0.0057 |
| BLiMP | regular_plural_subject_verb_agreement_1 | -11.13 | 0.0020 | 0.0794 | 0.8593 | 0.1006 | 0.1196 | 0.1175 | 0.0041 |
| BLiMP | tough_vs_raising_1 | -10.97 | 0.0025 | 0.0562 | 0.8877 | 0.0445 | 0.0697 | 0.0571 | 0.0031 |
| BLiMP | anaphor_gender_agreement | -10.71 | 0.0022 | 0.0808 | 0.8608 | 0.0888 | 0.0997 | 0.1014 | 0.0055 |
| BLiMP | matrix_question_npi_licensor_present | -9.47 | 0.0013 | 0.0746 | 0.8714 | 0.0814 | 0.0878 | 0.0938 | 0.0037 |
| EWoK | physical-dynamics | -19.50 | 0.0000 | 0.0557 | 0.9162 | 0.0353 | 0.0294 | 0.0559 | 0.0000 |
| EWoK | social-properties | -13.75 | 0.0000 | 0.0494 | 0.9154 | 0.1070 | 0.0693 | 0.0741 | 0.0022 |
| EWoK | material-dynamics | -8.83 | 0.0009 | 0.0420 | 0.9066 | 0.1092 | 0.0977 | 0.0746 | 0.0009 |
| EWoK | quantitative-properties | -5.80 | 0.0007 | 0.0684 | 0.8840 | 0.0817 | 0.0709 | 0.0771 | 0.0007 |
| EWoK | spatial-relations | 3.06 | 0.0000 | 0.0290 | 0.9605 | 0.0402 | 0.0281 | 0.0396 | 0.0000 |
| EWoK | material-properties | 4.97 | 0.0100 | 0.0539 | 0.9170 | 0.0474 | 0.0368 | 0.0406 | 0.0104 |
| EWoK | social-interactions | 5.78 | 0.0000 | 0.0744 | 0.8789 | 0.1737 | 0.1228 | 0.1560 | 0.0000 |
| Supplement | qa_congruence_easy | -12.50 | 0.0020 | 0.0923 | 0.8305 | 0.0876 | 0.1679 | 0.0906 | 0.0791 |
| Supplement | qa_congruence_tricky | -3.64 | 0.0019 | 0.1032 | 0.8105 | 0.0986 | 0.1884 | 0.1009 | 0.0856 |

## Largest spatial repair route status losses

| column | uid | Δ score | spatial repair route status frac<50 | spatial repair route status new-vs-old frac | 40k frac<50 | 40k-vs-spatial repair route status new frac |
|---|---|---:|---:|---:|---:|---:|
| BLiMP | principle_A_reconstruction | -20.89 | 0.0006 | 0.0510 | 0.0554 | 0.0626 |
| EWoK | physical-dynamics | -19.50 | 0.0000 | 0.0557 | 0.0353 | 0.0559 |
| BLiMP | wh_questions_object_gap | -17.34 | 0.0056 | 0.0670 | 0.0917 | 0.1010 |
| EWoK | social-properties | -13.75 | 0.0000 | 0.0494 | 0.1070 | 0.0741 |
| Supplement | qa_congruence_easy | -12.50 | 0.0020 | 0.0923 | 0.0876 | 0.0906 |
| BLiMP | animate_subject_trans | -11.91 | 0.0023 | 0.0759 | 0.0863 | 0.1005 |
| BLiMP | regular_plural_subject_verb_agreement_1 | -11.13 | 0.0020 | 0.0794 | 0.1006 | 0.1175 |
| BLiMP | tough_vs_raising_1 | -10.97 | 0.0025 | 0.0562 | 0.0445 | 0.0571 |
| BLiMP | anaphor_gender_agreement | -10.71 | 0.0022 | 0.0808 | 0.0888 | 0.1014 |
| BLiMP | matrix_question_npi_licensor_present | -9.47 | 0.0013 | 0.0746 | 0.0814 | 0.0938 |
| EWoK | material-dynamics | -8.83 | 0.0009 | 0.0420 | 0.1092 | 0.0746 |
| BLiMP | anaphor_number_agreement | -7.41 | 0.0020 | 0.0542 | 0.0668 | 0.0722 |
| BLiMP | npi_present_1 | -7.15 | 0.0023 | 0.0563 | 0.0781 | 0.0846 |
| BLiMP | existential_there_quantifiers_2 | -6.58 | 0.0016 | 0.0694 | 0.0894 | 0.1140 |
| BLiMP | wh_vs_that_no_gap_long_distance | -6.51 | 0.0033 | 0.0597 | 0.0811 | 0.0922 |
| BLiMP | sentential_subject_island | -5.83 | 0.0018 | 0.0662 | 0.1029 | 0.1183 |
| EWoK | quantitative-properties | -5.80 | 0.0007 | 0.0684 | 0.0817 | 0.0771 |
| BLiMP | principle_A_domain_1 | -5.47 | 0.0016 | 0.0652 | 0.0783 | 0.0861 |
| BLiMP | wh_island | -5.10 | 0.0022 | 0.0514 | 0.0545 | 0.0710 |
| BLiMP | principle_A_domain_2 | -4.81 | 0.0033 | 0.0732 | 0.0813 | 0.0913 |

## Scientific reading

- Use this file to distinguish simple low-support or token-identity stories from broader legal representation/optimization effects.

- A 40k run can recover syntax/QA only if its larger merge inventory helps more than its added low-support embeddings hurt; this table shows which subtasks would be exposed to each side of that trade.

- Do not turn these correlations into a new route without the pending mature clean-vs-reinvest comparison and full official 40k results.


JSON: `experiments/archive/frontier_consolidation/data/tokenizer_support_identity_alignment/tokenizer_support_identity_alignment.json`
CSV: `experiments/archive/frontier_consolidation/data/tokenizer_support_identity_alignment/per_uid_support_identity.csv`
