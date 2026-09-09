# legal deficit representation vs candidate alignment tokenizer-fragmentation / legal-deficit alignment

CPU-only analysis of existing tokenizers and existing eval strings. It does not train, evaluate, or tune a tokenizer from evaluation data.

## Tokenizers read

- `old_inherited_16k_nonlegal` len=16384: `experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M`
- `legal_step35_16k` len=16384: `experiments/archive/frontier_consolidation/data/compliant_tokenizer`
- `legal_bytealpha_16k` len=16384: `experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet`
- `legal_24k` len=24576: `experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_24k`
- `legal_32k` len=32768: `experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_32k`
- `legal_40k` len=40000: `experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k`
- `legal_40k_minfreq25` len=29529: `experiments/archive/representation_and_objectives/data/tokenizer_support_spectrum/tokenizers/legal_byte_bpe_40k_minfreq25`
- `legal_40k_minfreq50` len=19609: `experiments/archive/representation_and_objectives/data/tokenizer_support_spectrum/tokenizers/legal_byte_bpe_40k_minfreq50`

## Correlations with score delta (legal_step35 minus old_ref)

Negative delta means the legal spatial repair route status endpoint lost. A useful fragmentation explanation would show worse losses where `minus_old_tpw` or overflow increased; a useful representation-candidate signal would show candidate movement specifically on losing subtasks.

| column | n | metric | mean metric | Pearson(delta,metric) | Spearman |
|---|---:|---|---:|---:|---:|
| BLiMP | 67 | `minus_old_tpw` | 0.006517 | -0.0474 | -0.1653 |
| BLiMP | 67 | `bytealpha_minus_step35_tpw` | 0.000158 | 0.2084 | 0.0955 |
| BLiMP | 67 | `legal40k_minus_step35_tpw` | -0.144947 | 0.0262 | 0.0428 |
| BLiMP | 67 | `legal40k_minfreq50_minus_step35_tpw` | -0.041183 | 0.0859 | 0.0959 |
| BLiMP | 67 | `legal40k_closes_step35_old_tpw_gap_frac` | 33.800672 | -0.0864 | 0.0265 |
| BLiMP | 67 | `legal_step35_16k_frac_strings_over_256` | 0.000000 | — | — |
| BLiMP | 67 | `old_inherited_16k_nonlegal_frac_strings_over_256` | 0.000000 | — | — |
| BLiMP | 67 | `legal_40k_frac_strings_over_256` | 0.000000 | — | — |
| Supplement | 5 | `minus_old_tpw` | 0.062811 | -0.6708 | -0.6000 |
| Supplement | 5 | `bytealpha_minus_step35_tpw` | 0.000380 | 0.4191 | 0.3354 |
| Supplement | 5 | `legal40k_minus_step35_tpw` | -0.037339 | -0.2913 | 0.0000 |
| Supplement | 5 | `legal40k_minfreq50_minus_step35_tpw` | -0.008738 | 0.0122 | 0.1000 |
| Supplement | 5 | `legal40k_closes_step35_old_tpw_gap_frac` | -20.956846 | -0.3534 | -0.5000 |
| Supplement | 5 | `legal_step35_16k_frac_strings_over_256` | 0.000000 | — | — |
| Supplement | 5 | `old_inherited_16k_nonlegal_frac_strings_over_256` | 0.000000 | — | — |
| Supplement | 5 | `legal_40k_frac_strings_over_256` | 0.000000 | — | — |
| EWoK | 11 | `minus_old_tpw` | 0.019960 | 0.2465 | 0.2636 |
| EWoK | 11 | `bytealpha_minus_step35_tpw` | 0.022303 | 0.0705 | 0.0818 |
| EWoK | 11 | `legal40k_minus_step35_tpw` | -0.102288 | -0.3094 | -0.1182 |
| EWoK | 11 | `legal40k_minfreq50_minus_step35_tpw` | -0.019472 | 0.3987 | 0.4364 |
| EWoK | 11 | `legal40k_closes_step35_old_tpw_gap_frac` | 3.770700 | 0.2312 | 0.1879 |
| EWoK | 11 | `legal_step35_16k_frac_strings_over_256` | 0.000000 | — | — |
| EWoK | 11 | `old_inherited_16k_nonlegal_frac_strings_over_256` | 0.000000 | — | — |
| EWoK | 11 | `legal_40k_frac_strings_over_256` | 0.000000 | — | — |
| ALL | 83 | `minus_old_tpw` | 0.011690 | -0.1050 | -0.1304 |
| ALL | 83 | `bytealpha_minus_step35_tpw` | 0.003107 | -0.0616 | 0.1109 |
| ALL | 83 | `legal40k_minus_step35_tpw` | -0.132811 | -0.0935 | -0.0509 |
| ALL | 83 | `legal40k_minfreq50_minus_step35_tpw` | -0.036351 | 0.0179 | 0.0739 |
| ALL | 83 | `legal40k_closes_step35_old_tpw_gap_frac` | 26.799607 | -0.0718 | 0.0097 |
| ALL | 83 | `legal_step35_16k_frac_strings_over_256` | 0.000000 | — | — |
| ALL | 83 | `old_inherited_16k_nonlegal_frac_strings_over_256` | 0.000000 | — | — |
| ALL | 83 | `legal_40k_frac_strings_over_256` | 0.000000 | — | — |

## Focus rows

| column | uid | Δ score | old tpw | spatial repair route status tpw | spatial repair route status-old | 40k tpw | 40k-spatial repair route status | 40k gap-closed | minfreq50-spatial repair route status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | principle_A_reconstruction | -20.89 | 1.5099 | 1.5180 | 0.0081 | 1.4111 | -0.1069 | 13.229 | -0.0316 |
| BLiMP | wh_questions_object_gap | -17.34 | 1.3686 | 1.3695 | 0.0009 | 1.2227 | -0.1468 | 163.714 | -0.0378 |
| BLiMP | animate_subject_trans | -11.91 | 1.4586 | 1.4691 | 0.0105 | 1.3159 | -0.1532 | 14.570 | -0.0426 |
| BLiMP | regular_plural_subject_verb_agreement_1 | -11.13 | 1.5176 | 1.5300 | 0.0123 | 1.3429 | -0.1870 | 15.168 | -0.0546 |
| BLiMP | tough_vs_raising_1 | -10.97 | 1.3816 | 1.3721 | -0.0096 | 1.2853 | -0.0868 | -9.076 | -0.0311 |
| BLiMP | anaphor_gender_agreement | -10.71 | 1.5515 | 1.5718 | 0.0203 | 1.4011 | -0.1708 | 8.416 | -0.0501 |
| EWoK | physical-dynamics | -19.50 | 1.2324 | 1.2641 | 0.0317 | 1.1972 | -0.0669 | 2.111 | -0.0246 |
| EWoK | social-properties | -13.75 | 1.4239 | 1.4395 | 0.0156 | 1.3345 | -0.1051 | 6.725 | -0.0306 |
| EWoK | material-dynamics | -8.83 | 1.4855 | 1.4640 | -0.0215 | 1.3623 | -0.1017 | -4.740 | -0.0111 |
| EWoK | quantitative-properties | -5.80 | 1.3070 | 1.3262 | 0.0192 | 1.2159 | -0.1103 | 5.744 | -0.0263 |
| EWoK | physical-relations | -1.83 | 1.2110 | 1.2110 | 0.0000 | 1.1580 | -0.0530 | — | -0.0039 |
| EWoK | spatial-relations | 3.06 | 1.2085 | 1.2307 | 0.0222 | 1.1711 | -0.0596 | 2.687 | -0.0179 |
| EWoK | material-properties | 4.97 | 1.2790 | 1.3080 | 0.0289 | 1.2301 | -0.0779 | 2.694 | -0.0019 |
| EWoK | social-interactions | 5.78 | 1.4214 | 1.4523 | 0.0309 | 1.2550 | -0.1973 | 6.390 | -0.0188 |
| Supplement | qa_congruence_easy | -12.50 | 1.4276 | 1.5455 | 0.1178 | 1.5250 | -0.0204 | 0.173 | -0.0083 |
| Supplement | qa_congruence_tricky | -3.64 | 1.4859 | 1.6159 | 0.1300 | 1.5881 | -0.0278 | 0.214 | -0.0078 |
| Supplement | hypernym | 1.30 | 1.3094 | 1.3087 | -0.0007 | 1.2420 | -0.0667 | -93.125 | -0.0084 |
| Supplement | subject_aux_inversion | 1.42 | 1.1821 | 1.1761 | -0.0060 | 1.1043 | -0.0718 | -12.046 | -0.0192 |
| Supplement | turn_taking | 2.85 | 1.4297 | 1.5027 | 0.0730 | 1.5027 | 0.0000 | 0.000 | 0.0000 |

## Twenty largest spatial repair route status losses

| column | uid | Δ score | spatial repair route status-old tpw | 40k-spatial repair route status tpw | 40k gap-closed |
|---|---|---:|---:|---:|---:|
| BLiMP | principle_A_reconstruction | -20.89 | 0.0081 | -0.1069 | 13.229 |
| EWoK | physical-dynamics | -19.50 | 0.0317 | -0.0669 | 2.111 |
| BLiMP | wh_questions_object_gap | -17.34 | 0.0009 | -0.1468 | 163.714 |
| EWoK | social-properties | -13.75 | 0.0156 | -0.1051 | 6.725 |
| Supplement | qa_congruence_easy | -12.50 | 0.1178 | -0.0204 | 0.173 |
| BLiMP | animate_subject_trans | -11.91 | 0.0105 | -0.1532 | 14.570 |
| BLiMP | regular_plural_subject_verb_agreement_1 | -11.13 | 0.0123 | -0.1870 | 15.168 |
| BLiMP | tough_vs_raising_1 | -10.97 | -0.0096 | -0.0868 | -9.076 |
| BLiMP | anaphor_gender_agreement | -10.71 | 0.0203 | -0.1708 | 8.416 |
| BLiMP | matrix_question_npi_licensor_present | -9.47 | 0.0171 | -0.1372 | 8.037 |
| EWoK | material-dynamics | -8.83 | -0.0215 | -0.1017 | -4.740 |
| BLiMP | anaphor_number_agreement | -7.41 | 0.0002 | -0.1253 | 511.000 |
| BLiMP | npi_present_1 | -7.15 | 0.0084 | -0.1231 | 14.700 |
| BLiMP | existential_there_quantifiers_2 | -6.58 | 0.0151 | -0.1719 | 11.409 |
| BLiMP | wh_vs_that_no_gap_long_distance | -6.51 | -0.0011 | -0.1290 | -122.417 |
| BLiMP | sentential_subject_island | -5.83 | 0.0088 | -0.1909 | 21.727 |
| EWoK | quantitative-properties | -5.80 | 0.0192 | -0.1103 | 5.744 |
| BLiMP | principle_A_domain_1 | -5.47 | 0.0148 | -0.1317 | 8.898 |
| BLiMP | wh_island | -5.10 | -0.0011 | -0.1041 | -91.125 |
| BLiMP | principle_A_domain_2 | -4.81 | 0.0171 | -0.1423 | 8.303 |

## Interpretation guard

- This file can falsify simple fragmentation stories if score losses do not align with spatial repair route status-vs-old token inflation or overflow.

- A legal 40k candidate that reduces tokens per word on losing subtasks is still only a representation hypothesis until full official 40k evaluations finish; low-support vocabulary and the accumulated-trainer path remain separate evidence.


Full JSON: `experiments/archive/frontier_consolidation/data/tokenizer_fragmentation_deficit_alignment/tokenizer_fragmentation_deficit_alignment.json`
