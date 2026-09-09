# earlier analysis paired item-level flips

Do private-phase endpoints that move scalar cheap7 retain and gain the same official items relative to the shared chck_82M trunk, or are the changes seed-specific exchanges?

## Payload scores

| label | cheap7 | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chck82 | 43.9594 | 68.49128403651986 | 62.9378112562002 | 50.05545332553276 | 28.314041930298774 | 52.19117509443596 | 37.57766990291262 | 8.148713589261902 |
| coherent86_s43022 | 44.1814 | 68.51 | 63.64 | 50.02 | 28.32 | 52.05 | 38.565 | 8.165 |
| coherent_s43122 | 43.6457 | 68.35 | 63.71 | 49.9 | 27.8 | 51.97 | 35.62 | 8.17 |

## Deltas versus anchor

```json
{
  "coherent86_s43022": {
    "BLiMP": 0.01871596348014748,
    "Supplement": 0.7021887437998018,
    "EWoK": -0.03545332553275671,
    "Entity": 0.00595806970122581,
    "COMPS": -0.14117509443596532,
    "GlobalPIQA": 0.98733009708738,
    "Reading": 0.01628641073809689,
    "cheap7": 0.22197869497684053
  },
  "coherent_s43122": {
    "BLiMP": -0.14128403651986332,
    "Supplement": 0.7721887437998021,
    "EWoK": -0.15545332553276126,
    "Entity": -0.5140419302987738,
    "COMPS": -0.22117509443596362,
    "GlobalPIQA": -1.9576699029126203,
    "Reading": 0.021286410738097672,
    "cheap7": -0.31373559073744417
  }
}
```

## coherent86_s43022_minus_chck82

Aggregate: gains 2332, losses 2418, net -86 over 170722 common discrete item rows; discrete payload mean Δ +0.256.

| column | payload Δ | reconstructed Δ | common | gains | losses | net | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---:|---|---|
| BLiMP | +0.019 | +0.025 | 59875 | 565 | 545 | +20 | wh_island +23/960, principle_A_domain_1 +20/914, superlative_quantifiers_2 +20/986, principle_A_c_command +12/946 | npi_present_2 -42/914, npi_present_1 -20/909, ellipsis_n_bar_1 -10/802, animate_subject_trans -11/923 |
| Supplement | +0.702 | +0.698 | 5218 | 27 | 37 | -10 | qa_congruence_easy +2/64, qa_congruence_tricky +1/165, hypernym +1/842, turn_taking +0/280 | subject_aux_inversion -14/3867, turn_taking +0/280, hypernym +1/842, qa_congruence_tricky +1/165 |
| EWoK | -0.035 | -0.036 | 7618 | 117 | 113 | +4 | material-properties +2/170, social-interactions +2/294, social-relations +9/1548, physical-relations +1/818 | social-properties -4/328, physical-dynamics -1/120, spatial-relations -2/490, quantitative-properties -1/314 |
| Entity | +0.006 | +0.008 | 6780 | 50 | 54 | -4 | regular_5_ops +1/94, ambiref_2_ops +4/413, move_contents_5_ops +1/116, move_contents_4_ops +3/353 | move_contents_3_ops -5/406, regular_2_ops -4/405, ambiref_0_ops -5/508, ambiref_5_ops -1/123 |
| COMPS | -0.141 | -0.144 | 91028 | 1570 | 1668 | -98 | wugs_dist_in_between +23/13896, base -25/49340, wugs -22/13896, wugs_dist_before -74/13896 | wugs_dist_before -74/13896, wugs -22/13896, base -25/49340, wugs_dist_in_between +23/13896 |
| GlobalPIQA | +0.987 | +0.985 | 203 | 3 | 1 | +2 | GlobalPIQA_nonparallel +1/100, GlobalPIQA_parallel +1/103 | GlobalPIQA_parallel +1/103, GlobalPIQA_nonparallel +1/100 |

## coherent_s43122_minus_chck82

Aggregate: gains 2360, losses 2633, net -273 over 170722 common discrete item rows; discrete payload mean Δ -0.370.

| column | payload Δ | reconstructed Δ | common | gains | losses | net | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---:|---|---|
| BLiMP | -0.141 | -0.138 | 59875 | 486 | 565 | -79 | principle_A_c_command +12/946, principle_A_reconstruction +11/967, irregular_past_participle_verbs +10/942, tough_vs_raising_1 +10/948 | npi_present_2 -22/914, matrix_question_npi_licensor_present -20/929, irregular_plural_subject_verb_agreement_2 -15/892, npi_present_1 -15/909 |
| Supplement | +0.772 | +0.773 | 5218 | 28 | 31 | -3 | qa_congruence_easy +2/64, turn_taking +2/280, qa_congruence_tricky +1/165, subject_aux_inversion -4/3867 | hypernym -4/842, subject_aux_inversion -4/3867, qa_congruence_tricky +1/165, turn_taking +2/280 |
| EWoK | -0.155 | -0.156 | 7618 | 123 | 143 | -20 | social-interactions +3/294, physical-dynamics +1/120, material-properties +1/170, social-relations +7/1548 | social-properties -7/328, spatial-relations -5/490, physical-interactions -5/556, agent-properties -14/2210 |
| Entity | -0.514 | -0.515 | 6780 | 70 | 97 | -27 | move_contents_2_ops +3/399, move_contents_0_ops +3/516, ambiref_4_ops +1/434, ambiref_0_ops +1/508 | move_contents_5_ops -3/116, move_contents_4_ops -7/353, regular_3_ops -5/425, ambiref_1_ops -5/428 |
| COMPS | -0.221 | -0.220 | 91028 | 1652 | 1792 | -140 | wugs_dist_before +45/13896, base -25/49340, wugs -64/13896, wugs_dist_in_between -96/13896 | wugs_dist_in_between -96/13896, wugs -64/13896, base -25/49340, wugs_dist_before +45/13896 |
| GlobalPIQA | -1.958 | -1.956 | 203 | 1 | 5 | -4 | GlobalPIQA_nonparallel -1/100, GlobalPIQA_parallel -3/103 | GlobalPIQA_parallel -3/103, GlobalPIQA_nonparallel -1/100 |

## coherent86_s43022_minus_coherent_s43122

Aggregate: gains 2758, losses 2571, net 187 over 170722 common discrete item rows; discrete payload mean Δ +0.626.

| column | payload Δ | reconstructed Δ | common | gains | losses | net | best groups | worst groups |
|---|---:|---:|---:|---:|---:|---:|---|---|
| BLiMP | +0.160 | +0.163 | 59875 | 596 | 497 | +99 | matrix_question_npi_licensor_present +27/929, principle_A_domain_1 +19/914, adjunct_island +19/928, only_npi_scope +16/837 | npi_present_2 -20/914, irregular_past_participle_verbs -10/942, coordinate_structure_constraint_object_extraction -10/949, principle_A_case_2 -9/915 |
| Supplement | -0.070 | -0.076 | 5218 | 31 | 38 | -7 | hypernym +5/842, qa_congruence_easy +0/64, qa_congruence_tricky +0/165, subject_aux_inversion -10/3867 | turn_taking -2/280, subject_aux_inversion -10/3867, qa_congruence_easy +0/64, qa_congruence_tricky +0/165 |
| EWoK | +0.120 | +0.120 | 7618 | 125 | 101 | +24 | social-properties +3/328, physical-interactions +5/556, spatial-relations +3/490, agent-properties +13/2210 | physical-dynamics -2/120, quantitative-properties -2/314, social-interactions -1/294, material-dynamics -1/770 |
| Entity | +0.520 | +0.524 | 6780 | 98 | 75 | +23 | move_contents_5_ops +4/116, move_contents_4_ops +10/353, regular_3_ops +8/425, ambiref_2_ops +6/413 | ambiref_0_ops -6/508, move_contents_3_ops -3/406, ambiref_4_ops -2/434, regular_4_ops -1/388 |
| COMPS | +0.080 | +0.076 | 91028 | 1901 | 1859 | +42 | wugs_dist_in_between +119/13896, wugs +42/13896, base +0/49340, wugs_dist_before -119/13896 | wugs_dist_before -119/13896, base +0/49340, wugs +42/13896, wugs_dist_in_between +119/13896 |
| GlobalPIQA | +2.945 | +2.942 | 203 | 7 | 1 | +6 | GlobalPIQA_parallel +4/103, GlobalPIQA_nonparallel +2/100 | GlobalPIQA_nonparallel +2/100, GlobalPIQA_parallel +4/103 |

JSON: `experiments/archive/relation_learning/data/pairwise_item_flips/pairwise_item_flips.json`
