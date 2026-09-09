# fastpath pair item family review reviewer partial BLiMP comparison for layerwise exchange readout and gradient anatomy fast-path arms

Status: **PARTIAL_ONLY_BLIMP**. This file preserves the only completed layerwise exchange readout and gradient anatomy Arm2/Arm3 official output currently present. It is not enough to judge the fast-path route, because the load-bearing relation/state tasks and the rest of cheap7 are missing for coherent/spanbreak.

## Arm BLiMP scores from prediction files

| arm | scored items | accuracy | missing/nonexact |
|---|---:|---:|---:|
| chck82 | 59875 | 68.3708 | 0 |
| shuffled86 | 59875 | 69.1240 | 0 |
| coherent4M | 59875 | 68.4209 | 0 |
| spanbreak4M | 59875 | 68.0668 | 0 |

## Pair comparisons

| comparison | from acc | to acc | delta | gains | losses | net |
|---|---:|---:|---:|---:|---:|---:|
| coherent_minus_spanbreak | 68.0668 | 68.4209 | 0.3541 | 1882 | 1670 | 212 |
| coherent_minus_chck82 | 68.3708 | 68.4209 | 0.0501 | 761 | 731 | 30 |
| spanbreak_minus_chck82 | 68.3708 | 68.0668 | -0.3040 | 1771 | 1953 | -182 |
| shuffled_minus_chck82 | 68.3708 | 69.1240 | 0.7532 | 3488 | 3037 | 451 |
| coherent_minus_shuffled | 69.1240 | 68.4209 | -0.7031 | 3148 | 3569 | -421 |

## Coherent versus spanbreak by BLiMP field

| field | n | spanbreak acc | coherent acc | gains | losses | net |
|---|---:|---:|---:|---:|---:|---:|
| morphology | 15788 | 79.1044 | 79.7188 | 435 | 338 | 97 |
| semantics | 8359 | 67.5081 | 67.7713 | 292 | 270 | 22 |
| syntax | 23147 | 61.8223 | 61.8611 | 704 | 695 | 9 |
| syntax/semantics | 915 | 83.3880 | 83.4973 | 24 | 23 | 1 |
| syntax_semantics | 11666 | 64.7180 | 65.4295 | 427 | 344 | 83 |

## Largest UID movements for coherent over spanbreak

Positive net:
- matrix_question_npi_licensor_present: net 75 (gain 91, loss 16), 33.5845 -> 41.6577
- only_npi_scope: net 34 (gain 55, loss 21), 70.4898 -> 74.5520
- superlative_quantifiers_2: net 34 (gain 47, loss 13), 79.8174 -> 83.2657
- anaphor_gender_agreement: net 33 (gain 56, loss 23), 79.4027 -> 82.8012
- superlative_quantifiers_1: net 27 (gain 29, loss 2), 94.0756 -> 96.8335
- distractor_agreement_relational_noun: net 24 (gain 55, loss 31), 46.8274 -> 49.8731
- wh_questions_object_gap: net 24 (gain 53, loss 29), 52.6193 -> 55.4133
- distractor_agreement_relative_clause: net 24 (gain 48, loss 24), 26.2916 -> 29.0471

Negative net:
- only_npi_licensor_present: net -63 (gain 5, loss 68), 87.6417 -> 80.4989
- existential_there_quantifiers_2: net -51 (gain 38, loss 89), 47.7497 -> 42.1515
- left_branch_island_echo_question: net -39 (gain 17, loss 56), 42.6610 -> 38.5428
- wh_island: net -28 (gain 24, loss 52), 61.0417 -> 58.1250
- inchoative: net -20 (gain 12, loss 32), 47.2515 -> 44.9123
- animate_subject_trans: net -15 (gain 25, loss 40), 69.0141 -> 67.3889
- ellipsis_n_bar_1: net -14 (gain 28, loss 42), 74.6883 -> 72.9426
- principle_A_domain_3: net -11 (gain 44, loss 55), 58.7673 -> 57.5983

## Reviewer interpretation

The BLiMP-only signal is weakly compatible with coherent natural replay preserving more syntax than spanbreak (coherent 68.52, spanbreak 68.18, net +318 items) and roughly preserving chck82 BLiMP (68.49 -> 68.52). But it is far below shuffled86 BLiMP (69.23) and does not touch the route's load-bearing relation/state families. The layerwise exchange readout and gradient anatomy route therefore remains unjudged until at least EWoK, Entity, GlobalPIQA, COMPS, Reading, Supplement, and preferably SuperGLUE outputs are produced for both coherent and spanbreak with identical collation.

JSON: `experiments/archive/representation_and_objectives/data/fastpath_partial_blimp_review/fastpath_partial_blimp_review.json`
