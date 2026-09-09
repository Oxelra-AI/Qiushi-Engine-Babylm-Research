# coupled control interpretation infrastructure dual-view broad-damage and hard-surface anatomy

Status: **PASS**

CPU-only parse of existing 20M dual-view reports and endpoint resolved dualview hardsurface row CSVs. No training, no model scoring, and `chck_82M` untouched.

## Broad score deltas vs `mlm_only_20M`
| arm | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sep_sparse20_aligned_20M | 0.5064 | 1.7200 | 0.1700 | 0.6700 | 0.4100 | -0.2200 | 1.0000 | -0.2050 |
| sep_sparse20_shuffled_20M | 0.2771 | 1.0200 | -0.9800 | 0.7800 | -0.0400 | 0.0400 | 1.5300 | -0.4100 |
| coupled_sparse20_aligned_20M | -0.9929 | -2.5200 | -3.4300 | 0.0500 | -0.4600 | -0.9800 | 1.5000 | -1.1100 |

## Coupled aligned broad damage/gains: largest report-level deltas
Losses (excluding GlobalPIQA item IDs):
- BLiMP / UID ACCURACY / existential_there_quantifiers_2: 33.26 -> 4.94 (delta -28.32)
- BLiMP / UID ACCURACY / ellipsis_n_bar_2: 63.77 -> 37.80 (delta -25.97)
- BLiMP / UID ACCURACY / anaphor_number_agreement: 75.08 -> 50.70 (delta -24.38)
- BLiMP / LINGUISTICS_TERM ACCURACY / anaphor_agreement: 67.09 -> 45.53 (delta -21.56)
- BLiMP / UID ACCURACY / determiner_noun_agreement_2: 85.71 -> 65.63 (delta -20.08)
- BLiMP / UID ACCURACY / irregular_past_participle_adjectives: 86.06 -> 67.01 (delta -19.05)
- BLiMP / UID ACCURACY / anaphor_gender_agreement: 59.42 -> 40.58 (delta -18.84)
- BLiMP / UID ACCURACY / determiner_noun_agreement_with_adj_2: 72.69 -> 54.30 (delta -18.39)
- Supplement / UID ACCURACY / turn_taking: 67.14 -> 51.43 (delta -15.71)
- BLiMP / UID ACCURACY / determiner_noun_agreement_with_adj_irregular_1: 78.55 -> 63.65 (delta -14.90)
- BLiMP / UID ACCURACY / regular_plural_subject_verb_agreement_1: 69.21 -> 54.38 (delta -14.83)
- BLiMP / UID ACCURACY / determiner_noun_agreement_with_adj_irregular_2: 82.74 -> 68.21 (delta -14.53)
- BLiMP / LINGUISTICS_TERM ACCURACY / determiner_noun_agreement: 77.45 -> 62.98 (delta -14.47)
- BLiMP / UID ACCURACY / existential_there_quantifiers_1: 87.53 -> 73.98 (delta -13.55)
- BLiMP / UID ACCURACY / determiner_noun_agreement_1: 76.21 -> 63.19 (delta -13.02)

Gains (excluding GlobalPIQA item IDs):
- BLiMP / UID ACCURACY / wh_island: 38.54 -> 65.00 (delta 26.46)
- BLiMP / UID ACCURACY / superlative_quantifiers_2: 49.80 -> 75.05 (delta 25.25)
- BLiMP / UID ACCURACY / left_branch_island_echo_question: 61.56 -> 79.94 (delta 18.38)
- BLiMP / UID ACCURACY / principle_A_domain_1: 58.64 -> 75.38 (delta 16.74)
- BLiMP / UID ACCURACY / distractor_agreement_relational_noun: 24.75 -> 40.99 (delta 16.24)
- BLiMP / UID ACCURACY / wh_questions_object_gap: 40.05 -> 56.23 (delta 16.18)
- BLiMP / UID ACCURACY / sentential_subject_island: 36.21 -> 50.78 (delta 14.57)
- BLiMP / UID ACCURACY / distractor_agreement_relative_clause: 31.69 -> 43.63 (delta 11.94)
- BLiMP / UID ACCURACY / only_npi_scope: 48.63 -> 59.38 (delta 10.75)
- EWoK / CONTEXT_CONTRAST ACCURACY / game: 50.00 -> 60.00 (delta 10.00)
- BLiMP / UID ACCURACY / wh_questions_subject_gap: 82.74 -> 90.09 (delta 7.35)
- BLiMP / UID ACCURACY / principle_A_c_command: 64.80 -> 71.56 (delta 6.76)

## Coupled aligned EWoK hard-subset repair by domain
Negative stable-failure delta means fewer stable reversals than `mlm_only_20M`.
| domain | n | acc_delta | stable_failure_delta | median_interaction_delta |
|---|---:|---:|---:|---:|
| agent-properties | 494 | 0.3462 | -250 | 1.2949 |
| physical-relations | 223 | 0.3812 | -129 | 3.4517 |
| social-relations | 303 | 0.1518 | -44 | 0.1167 |
| physical-interactions | 93 | 0.3118 | -42 | 0.5862 |
| spatial-relations | 147 | 0.0204 | -16 | 0.4036 |
| social-interactions | 38 | 0.3684 | -6 | 0.0775 |
| physical-dynamics | 13 | 0.2308 | -5 | 0.9973 |
| material-dynamics | 65 | 0.1846 | -3 | 0.1024 |
| material-properties | 9 | 0.1111 | -1 | 0.5047 |
| quantitative-properties | 46 | 0.0435 | 1 | 0.1221 |
| social-properties | 40 | -0.0750 | 3 | -0.2167 |

## Coupled aligned EWoK hard-subset repair by ContextDiff
| ContextDiff | n | acc_delta | stable_failure_delta | median_interaction_delta |
|---|---:|---:|---:|---:|
| variable swap | 841 | 0.3769 | -438 | 1.3251 |
| antonym | 485 | 0.0742 | -49 | 0.1898 |
| variable_swap | 7 | 0.4286 | -5 | 0.7666 |
| material | 73 | 0.1781 | -3 | 0.0749 |
| game | 1 | 1.0000 | -1 | 1.6863 |
| negation | 11 | -0.2727 | -1 | 0.1593 |
| active-passive | 5 | 0.0000 | 0 | 0.4131 |
| number | 11 | -0.1818 | 2 | -0.1240 |
| other | 37 | -0.0541 | 3 | -0.1885 |

## GlobalPIQA hard52 rank/margin transitions vs `mlm_only_20M`
| arm | n | improved_rank | worsened_rank | margin_improved | margin_worsened | flips_to_correct | lost_correct | mean_rank_delta | mean_margin_delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sep_sparse20_aligned_20M | 52 | 20 | 9 | 25 | 21 | 3 | 1 | -0.2692 | -0.0855 |
| sep_sparse20_shuffled_20M | 52 | 15 | 10 | 24 | 24 | 0 | 1 | -0.0769 | 0.0150 |
| coupled_sparse20_aligned_20M | 52 | 24 | 13 | 31 | 19 | 4 | 1 | -0.2308 | -0.1797 |

## Scientific reading
- Coupled aligned is the only existing arm with a large EWoK hard-row repair, but its broad losses are not uniform: strongest broad damage is BLiMP morphology/agreement/quantifier families plus Supplement turn-taking and QA congruence, with Reading and COMPS also down.
- The hard EWoK repair is concentrated in the largest load-bearing strata: agent-properties, physical-relations, variable swap, physical-interactions, social-relations, and spatial-relations. This is the surface a broad-preserving coupled variant must keep if the pending shuffled control confirms true-correspondence specificity.
- Separated aligned preserves broad score but fails the main hard EWoK surface; it is useful as a broad-preservation clue, not as the mechanism successor.
- This analysis deliberately does not decide true-correspondence specificity; that depends on the pending matched coupled_sparse20_shuffled_20M control read by the upgraded coupled correspondence control script.

JSON: `experiments/archive/representation_and_objectives/data/dualview_broad_damage_anatomy/dualview_broad_damage_anatomy.json`
