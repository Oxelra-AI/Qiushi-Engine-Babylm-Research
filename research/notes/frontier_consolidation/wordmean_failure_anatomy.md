# wordmean failure anatomy word-mean failure anatomy

CPU-only analysis of already completed word-mean/token-mean/clean 70M/80M official-compatible cheap reports; no model training or evaluation.

## Column movement
| exposure | wm mean7 | tm mean7 | clean mean7 | wm-tm | wm-clean | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70 | 42.6043 | 42.6086 | 41.3164 | -0.0043 | 1.2879 | -0.7800 | -0.3100 | -1.8000 | -1.0500 | 0.7700 | 4.0700 | -0.9300 |
| 80 | 42.7029 | 42.9486 | 41.6021 | -0.2457 | 1.1007 | -1.5600 | -1.7100 | -1.8100 | -1.1500 | 0.5100 | 4.5250 | -0.5250 |

## Raw GlobalPIQA split
| exposure | task | wordmean | tokenmean | clean | wm-tm | wm-clean |
|---:|---|---:|---:|---:|---:|---:|
| 70 | COMPS | 52.5900 | 51.8200 | 51.1300 | 0.7700 | 1.4600 |
| 70 | GlobalPIQA_parallel | 25.2400 | 30.1000 | 25.2400 | -4.8600 | 0.0000 |
| 70 | GlobalPIQA_nonparallel | 54.0000 | 41.0000 | 44.0000 | 13.0000 | 10.0000 |
| 80 | COMPS | 52.4400 | 51.9300 | 51.2800 | 0.5100 | 1.1600 |
| 80 | GlobalPIQA_parallel | 26.2100 | 28.1600 | 26.2100 | -1.9500 | 0.0000 |
| 80 | GlobalPIQA_nonparallel | 54.0000 | 43.0000 | 45.0000 | 11.0000 | 9.0000 |

## 80M UID and domain movement
### BLiMP
- UID count 67; mean wm-tm -1.5552; median -1.8000; negative 42; positive 25; <=-2 32; >=+2 16.
- Pearson wm-tm UID delta vs spatial repair route status legal-loss magnitude: 0.4863; Spearman 0.5149.
- largest UID losses: only_npi_licensor_present (-36.9600), superlative_quantifiers_1 (-15.8300), superlative_quantifiers_2 (-11.3600), adjunct_island (-10.6700), tough_vs_raising_2 (-10.2200), principle_A_case_2 (-8.8500), wh_island (-8.6500), sentential_negation_npi_scope (-7.4600)
- largest UID gains: matrix_question_npi_licensor_present (23.0400), tough_vs_raising_1 (10.4400), sentential_subject_island (10.0900), principle_A_reconstruction (9.8200), regular_plural_subject_verb_agreement_1 (7.7500), anaphor_gender_agreement (6.0800), existential_there_subject_raising (5.4100), principle_A_domain_1 (5.3600)
- field accuracy mean wm-tm -2.0150; largest losses: semantics (-5.8700), syntax (-1.1100), morphology (-0.9000), syntax/semantics (-0.1800)
- linguistics_term accuracy mean wm-tm -1.0608; largest losses: quantifiers (-7.1700), npi_licensing (-5.0700), determiner_noun_agreement (-2.5200), argument_structure (-2.4400), island_effects (-2.3800)

### Supplement
- UID count 5; mean wm-tm -1.7140; median -1.5400; negative 4; positive 1; <=-2 2; >=+2 1.
- Pearson wm-tm UID delta vs spatial repair route status legal-loss magnitude: 0.6897; Spearman 0.2236.
- largest UID losses: qa_congruence_tricky (-7.8700), subject_aux_inversion (-3.9800), hypernym (-1.5400), turn_taking (-1.4300), qa_congruence_easy (6.2500)
- largest UID gains: qa_congruence_easy (6.2500), turn_taking (-1.4300), hypernym (-1.5400), subject_aux_inversion (-3.9800), qa_congruence_tricky (-7.8700)
- field accuracy mean wm-tm -3.4500; largest losses: supplement (-3.4500)
- linguistics_term accuracy mean wm-tm -3.4500; largest losses: supplement (-3.4500)

### EWoK
- UID count 11; mean wm-tm -1.8109; median -0.6100; negative 7; positive 4; <=-2 3; >=+2 2.
- Pearson wm-tm UID delta vs spatial repair route status legal-loss magnitude: 0.3423; Spearman 0.2384.
- EWoK relation mean wm-tm -1.8000; property mean -1.8300.
- relation deltas: material-dynamics -4.4200, physical-dynamics -0.8400, physical-interactions -0.5400, physical-relations -0.6100, social-interactions 0.6800, social-relations 0.0700, spatial-relations -6.9400
- property deltas: agent-properties -0.6800, material-properties -14.7100, quantitative-properties 3.5000, social-properties 4.5700
- largest UID losses: material-properties (-14.7100), spatial-relations (-6.9400), material-dynamics (-4.4200), physical-dynamics (-0.8400), agent-properties (-0.6800), physical-relations (-0.6100), physical-interactions (-0.5400), social-relations (0.0700)
- largest UID gains: social-properties (4.5700), quantitative-properties (3.5000), social-interactions (0.6800), social-relations (0.0700), physical-interactions (-0.5400), physical-relations (-0.6100), agent-properties (-0.6800), physical-dynamics (-0.8400)

## Scientific reading
- Word-mean remains a positive compact-view treatment relative to the clean fixed-tokenizer control, but it is weaker than token-mean reinvestment on the broad language surface by 80M.
- Its advantage is concentrated in GlobalPIQA, especially nonparallel examples, and modestly COMPS; BLiMP, Supplement, EWoK, Entity, and Reading move down together.
- Because the 41.8 gap requires roughly +0.70 cheap7 mean gain if SuperGLUE/AoA stay flat, the observed 80M -0.25 mean7 and damaged core columns make global word-mean unsuitable for 100M continuation.
- The failure is not evidence against compact semantic second views: word-mean still beats clean at 70M/80M, while the ordinary token-mean objective carries the stronger late-emerging treatment effect.
- If the GlobalPIQA gain is reused later, it should be isolated as a small scheduled or mixed credit component, not as a wholesale replacement for token-mean MLM, and only after current minfreq50 evidence is read.

JSON: `experiments/archive/frontier_consolidation/data/wordmean_failure_anatomy/wordmean_failure_anatomy.json`
