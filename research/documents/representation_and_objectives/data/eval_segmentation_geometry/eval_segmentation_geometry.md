# earlier analysis — held-out segmentation geometry vs legal-tokenizer deficit

CPU-only analysis over existing official evaluation text, existing tokenizers, and the earlier analysis old-vs-legal per-UID score table. It does not train, score model checkpoints, fit a tokenizer, or use pending SGCR/full-EWoK results.

## Tokenizers
- `old_inherited_16k_nonlegal` len=16384: `experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M`
- `legal_step35_16k` len=16384: `experiments/archive/frontier_consolidation/data/compliant_tokenizer`
- `legal_a01_16k` len=16384: `experiments/archive/representation_and_objectives/data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer`
- `legal_a01_40k` len=40000: `experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k`
- `legal_a01_40k_minfreq50` len=19609: `experiments/archive/representation_and_objectives/data/tokenizer_support_spectrum/tokenizers/legal_byte_bpe_40k_minfreq50`

## Correlations: aoa mincontext discrepancy audit score delta vs fine-grained held-out tokenization geometry
`score_delta = legal_step35 - old_ref`; negative means the legal tokenizer endpoint lost. If a seam mechanism is primary, losses should align with lower legal-vs-old seam preservation or stem-piece reuse, not merely with total tokens per word.

| column | n | metric | mean | Pearson(delta,metric) | Spearman |
|---|---:|---|---:|---:|---:|
| BLiMP | 67 | `minus_old_tokens_per_word` | 0.006513 | -0.0475 | -0.1665 |
| BLiMP | 67 | `minus_old_whole_morph_rate` | 0.008993 | -0.0181 | 0.0534 |
| BLiMP | 67 | `minus_old_morph_seam_preserved_rate` | 0.009324 | 0.0077 | 0.0148 |
| BLiMP | 67 | `minus_old_suffix_stem_piece_reuse_rate` | -0.014489 | -0.0816 | -0.1094 |
| BLiMP | 67 | `minus_old_whole_content_rate` | 0.008908 | 0.0084 | 0.0704 |
| BLiMP | 67 | `vs_old_mean_abs_cumulative_token_displacement` | 0.209388 | 0.1239 | 0.1026 |
| BLiMP | 67 | `vs_old_mean_max_abs_cumulative_token_displacement` | 0.363631 | 0.1156 | 0.0888 |
| BLiMP | 67 | `vs_old_mean_abs_word_token_delta` | 0.054227 | -0.0454 | -0.1103 |
| BLiMP | 67 | `legal40k_minus_step35_morph_seam_preserved_rate` | -0.080613 | -0.1360 | -0.1985 |
| BLiMP | 67 | `legal40k_minus_step35_whole_morph_rate` | 0.246180 | 0.0489 | 0.0190 |
| BLiMP | 67 | `legal40k_closes_step35_old_morph_seam_preserved_rate_frac` | 6.067875 | 0.0532 | 0.1381 |
| BLiMP | 67 | `legal40k_closes_step35_old_tokens_per_word_frac` | 33.800672 | -0.0864 | 0.0265 |
| Supplement | 5 | `minus_old_tokens_per_word` | -0.002501 | -0.0930 | 0.3000 |
| Supplement | 5 | `minus_old_whole_morph_rate` | -0.001430 | -0.0496 | 0.2236 |
| Supplement | 5 | `minus_old_morph_seam_preserved_rate` | -0.001940 | -0.3134 | -0.3536 |
| Supplement | 5 | `minus_old_suffix_stem_piece_reuse_rate` | 0.002722 | -0.2459 | -0.4104 |
| Supplement | 5 | `minus_old_whole_content_rate` | -0.002447 | 0.7641 | 0.9000 |
| Supplement | 5 | `vs_old_mean_abs_cumulative_token_displacement` | 0.067170 | 0.4387 | 0.0000 |
| Supplement | 5 | `vs_old_mean_max_abs_cumulative_token_displacement` | 0.151013 | 0.3907 | 0.0000 |
| Supplement | 5 | `vs_old_mean_abs_word_token_delta` | 0.012566 | 0.3110 | 0.0000 |
| Supplement | 5 | `legal40k_minus_step35_morph_seam_preserved_rate` | -0.034203 | -0.4517 | -0.1026 |
| Supplement | 5 | `legal40k_minus_step35_whole_morph_rate` | 0.095577 | 0.4758 | 0.1026 |
| Supplement | 5 | `legal40k_closes_step35_old_morph_seam_preserved_rate_frac` | -5.234783 | — | — |
| Supplement | 5 | `legal40k_closes_step35_old_tokens_per_word_frac` | 5.665509 | -0.9069 | -0.8000 |
| EWoK | 11 | `minus_old_tokens_per_word` | 0.019981 | 0.2468 | 0.2636 |
| EWoK | 11 | `minus_old_whole_morph_rate` | -0.023348 | 0.3339 | 0.1101 |
| EWoK | 11 | `minus_old_morph_seam_preserved_rate` | 0.006318 | -0.5626 | 0.1621 |
| EWoK | 11 | `minus_old_suffix_stem_piece_reuse_rate` | 0.031839 | -0.4093 | -0.1101 |
| EWoK | 11 | `minus_old_whole_content_rate` | -0.024860 | 0.3666 | 0.2506 |
| EWoK | 11 | `vs_old_mean_abs_cumulative_token_displacement` | 0.179875 | 0.3463 | 0.1273 |
| EWoK | 11 | `vs_old_mean_max_abs_cumulative_token_displacement` | 0.460217 | 0.2898 | 0.3818 |
| EWoK | 11 | `vs_old_mean_abs_word_token_delta` | 0.038406 | 0.2751 | 0.4273 |
| EWoK | 11 | `legal40k_minus_step35_morph_seam_preserved_rate` | -0.121064 | 0.1466 | 0.0729 |
| EWoK | 11 | `legal40k_minus_step35_whole_morph_rate` | 0.243882 | -0.0352 | 0.0182 |
| EWoK | 11 | `legal40k_closes_step35_old_morph_seam_preserved_rate_frac` | 2.817654 | 0.7154 | 0.6571 |
| EWoK | 11 | `legal40k_closes_step35_old_tokens_per_word_frac` | 3.770700 | 0.2312 | 0.1879 |
| ALL | 83 | `minus_old_tokens_per_word` | 0.007755 | -0.0114 | -0.0877 |
| ALL | 83 | `minus_old_whole_morph_rate` | 0.004079 | 0.1370 | 0.0167 |
| ALL | 83 | `minus_old_morph_seam_preserved_rate` | 0.008247 | -0.1617 | 0.0234 |
| ALL | 83 | `minus_old_suffix_stem_piece_reuse_rate` | -0.007312 | -0.2087 | -0.0669 |
| ALL | 83 | `minus_old_whole_content_rate` | 0.003749 | 0.1690 | 0.0917 |
| ALL | 83 | `vs_old_mean_abs_cumulative_token_displacement` | 0.196909 | 0.1899 | 0.1569 |
| ALL | 83 | `vs_old_mean_max_abs_cumulative_token_displacement` | 0.363623 | 0.1370 | 0.1662 |
| ALL | 83 | `vs_old_mean_abs_word_token_delta` | 0.049621 | 0.0665 | -0.0163 |
| ALL | 83 | `legal40k_minus_step35_morph_seam_preserved_rate` | -0.083179 | -0.0293 | -0.1094 |
| ALL | 83 | `legal40k_minus_step35_whole_morph_rate` | 0.236803 | 0.0708 | 0.0395 |
| ALL | 83 | `legal40k_closes_step35_old_morph_seam_preserved_rate_frac` | 5.645902 | 0.1044 | 0.2081 |
| ALL | 83 | `legal40k_closes_step35_old_tokens_per_word_frac` | 28.703877 | -0.0820 | -0.0332 |

## Worst-vs-best UID contrast
| metric | worst20 mean | best20 mean |
|---|---:|---:|
| score delta | -10.0830 | 7.1305 |
| `minus_old_tokens_per_word` | 0.008300 | 0.007432 |
| `minus_old_whole_morph_rate` | 0.002339 | 0.003331 |
| `minus_old_morph_seam_preserved_rate` | 0.012333 | 0.008840 |
| `minus_old_suffix_stem_piece_reuse_rate` | -0.000695 | -0.010883 |
| `minus_old_whole_content_rate` | -0.001478 | 0.004768 |
| `vs_old_mean_abs_cumulative_token_displacement` | 0.177365 | 0.203380 |
| `vs_old_mean_max_abs_cumulative_token_displacement` | 0.347956 | 0.392173 |
| `vs_old_mean_abs_word_token_delta` | 0.048872 | 0.045809 |
| `legal40k_minus_step35_morph_seam_preserved_rate` | -0.080933 | -0.090822 |
| `legal40k_minus_step35_whole_morph_rate` | 0.228066 | 0.244479 |
| `legal40k_closes_step35_old_morph_seam_preserved_rate_frac` | -2.225381 | 3.265921 |
| `legal40k_closes_step35_old_tokens_per_word_frac` | 34.619711 | 16.977976 |

## Focus rows
| column | uid | Δ score | aoa mincontext discrepancy audit-old tpw | aoa mincontext discrepancy audit-old seam | aoa mincontext discrepancy audit-old whole morph | aoa mincontext discrepancy audit-old stem reuse | phase mean abs | 40k-aoa mincontext discrepancy audit seam | 40k seam gap closed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | principle_A_reconstruction | -20.89 | 0.0081 | 0.0182 | 0.0073 | 0.0028 | 0.0794 | -0.0874 | 4.800 |
| BLiMP | wh_questions_object_gap | -17.34 | 0.0009 | -0.0092 | 0.0294 | -0.0376 | 0.2572 | -0.0665 | -7.263 |
| BLiMP | animate_subject_trans | -11.91 | 0.0105 | 0.0046 | 0.0083 | -0.0105 | 0.1847 | -0.0680 | 14.818 |
| BLiMP | regular_plural_subject_verb_agreement_1 | -11.13 | 0.0123 | 0.0203 | 0.0103 | -0.0119 | 0.2016 | -0.0960 | 4.736 |
| BLiMP | tough_vs_raising_1 | -10.97 | -0.0096 | -0.0234 | 0.0313 | -0.0601 | 0.2232 | -0.0851 | -3.638 |
| BLiMP | anaphor_gender_agreement | -10.71 | 0.0203 | 0.0219 | 0.0150 | 0.0164 | 0.2573 | -0.1039 | 4.737 |
| EWoK | physical-dynamics | -19.50 | 0.0317 | 0.1400 | -0.1600 | 0.2000 | 0.0614 | -0.1400 | 1.000 |
| EWoK | social-properties | -13.75 | 0.0156 | -0.0055 | -0.0055 | 0.0119 | 0.1129 | -0.1305 | -23.600 |
| EWoK | material-dynamics | -8.83 | -0.0214 | -0.0289 | 0.0433 | -0.0289 | 0.0892 | -0.1625 | -5.625 |
| EWoK | quantitative-properties | -5.80 | 0.0192 | 0.0000 | 0.0451 | -0.0403 | 0.3350 | 0.0000 | — |
| EWoK | physical-relations | -1.83 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.1981 | — |
| EWoK | spatial-relations | 3.06 | 0.0222 | 0.0000 | -0.0935 | 0.0935 | 0.2029 | -0.1679 | — |
| EWoK | material-properties | 4.97 | 0.0289 | 0.0000 | 0.0000 | 0.0000 | 0.0539 | 0.0000 | — |
| EWoK | social-interactions | 5.78 | 0.0309 | 0.0059 | 0.0000 | 0.0000 | 0.2867 | -0.1538 | 26.000 |
| Supplement | qa_congruence_easy | -12.50 | -0.0009 | 0.0000 | 0.0000 | 0.0000 | 0.0096 | 0.0000 | — |
| Supplement | qa_congruence_tricky | -3.64 | -0.0049 | 0.0000 | 0.0000 | 0.0394 | 0.0161 | -0.0394 | — |
| Supplement | hypernym | 1.30 | -0.0007 | 0.0000 | -0.0282 | 0.0036 | 0.1097 | -0.0808 | — |
| Supplement | subject_aux_inversion | 1.42 | -0.0060 | -0.0097 | 0.0210 | -0.0295 | 0.2005 | -0.0508 | -5.235 |
| Supplement | turn_taking | 2.85 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — |

## Twenty largest legal aoa mincontext discrepancy audit losses
| column | uid | Δ score | aoa mincontext discrepancy audit-old tpw | aoa mincontext discrepancy audit-old seam | aoa mincontext discrepancy audit-old stem reuse | phase mean abs |
|---|---|---:|---:|---:|---:|---:|
| BLiMP | principle_A_reconstruction | -20.89 | 0.0081 | 0.0182 | 0.0028 | 0.0794 |
| EWoK | physical-dynamics | -19.50 | 0.0317 | 0.1400 | 0.2000 | 0.0614 |
| BLiMP | wh_questions_object_gap | -17.34 | 0.0009 | -0.0092 | -0.0376 | 0.2572 |
| EWoK | social-properties | -13.75 | 0.0156 | -0.0055 | 0.0119 | 0.1129 |
| Supplement | qa_congruence_easy | -12.50 | -0.0009 | 0.0000 | 0.0000 | 0.0096 |
| BLiMP | animate_subject_trans | -11.91 | 0.0105 | 0.0046 | -0.0105 | 0.1847 |
| BLiMP | regular_plural_subject_verb_agreement_1 | -11.13 | 0.0123 | 0.0203 | -0.0119 | 0.2016 |
| BLiMP | tough_vs_raising_1 | -10.97 | -0.0096 | -0.0234 | -0.0601 | 0.2232 |
| BLiMP | anaphor_gender_agreement | -10.71 | 0.0203 | 0.0219 | 0.0164 | 0.2573 |
| BLiMP | matrix_question_npi_licensor_present | -9.47 | 0.0171 | 0.0089 | -0.0063 | 0.2094 |
| EWoK | material-dynamics | -8.83 | -0.0214 | -0.0289 | -0.0289 | 0.0892 |
| BLiMP | anaphor_number_agreement | -7.41 | 0.0002 | 0.0020 | -0.0069 | 0.1560 |
| BLiMP | npi_present_1 | -7.15 | 0.0084 | 0.0043 | -0.0098 | 0.1385 |
| BLiMP | existential_there_quantifiers_2 | -6.58 | 0.0151 | 0.0495 | 0.0368 | 0.1348 |
| BLiMP | wh_vs_that_no_gap_long_distance | -6.51 | -0.0011 | -0.0007 | -0.0197 | 0.2802 |
| BLiMP | sentential_subject_island | -5.83 | 0.0088 | 0.0449 | 0.0106 | 0.1123 |
| EWoK | quantitative-properties | -5.80 | 0.0192 | 0.0000 | -0.0403 | 0.3350 |
| BLiMP | principle_A_domain_1 | -5.47 | 0.0148 | 0.0052 | -0.0162 | 0.2822 |
| BLiMP | wh_island | -5.10 | -0.0011 | -0.0148 | -0.0341 | 0.1649 |
| BLiMP | principle_A_domain_2 | -4.81 | 0.0171 | 0.0093 | -0.0101 | 0.2572 |

## Reading
This sharper geometry test should be read together with corpus lineage multiplicity audit: simple held-out tokens-per-word already failed as an explanation. A route-relevant positive result would require structured alignment of losses with seam/stem/phase metrics on the affected UIDs. If the correlations remain weak or sign-mixed, a pure tokenizer-seam story is not sufficient; the next route should emphasize either a different representation mechanism, optimization interaction, or the distinct compact-view/conditional signal rather than a type-dedup tokenizer run.

Full JSON: `experiments/archive/representation_and_objectives/data/eval_segmentation_geometry/eval_segmentation_geometry.json`
Per-UID CSV: `experiments/archive/representation_and_objectives/data/eval_segmentation_geometry/eval_segmentation_geometry_per_uid.csv`
