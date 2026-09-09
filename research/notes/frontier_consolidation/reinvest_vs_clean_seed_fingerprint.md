# clean full trajectory seed control reinvest vs clean 100M seed-spread fingerprint

CPU-only comparison from existing reports. It is a 100M fingerprint, not the matched sparse temporal control.

## BLiMP
- shared subtasks: 67
- mean clean delta: -0.818
- mean reinvest delta: -0.873
- mean excess reinvest-clean: -0.056
- delta correlation clean vs reinvest: 0.326
- same-sign fraction: 0.552
- largest negative excess: superlative_quantifiers_1 r=-22.0 c=15.8 ex=-37.8; left_branch_island_simple_question r=-11.5 c=6.5 ex=-18.0; only_npi_scope r=1.0 c=17.2 ex=-16.2; distractor_agreement_relational_noun r=-5.5 c=8.1 ex=-13.6; distractor_agreement_relative_clause r=-12.0 c=-0.1 ex=-11.9
- largest positive excess: left_branch_island_echo_question r=12.0 c=-10.8 ex=22.8; adjunct_island r=0.5 c=-17.3 ex=17.8; npi_present_2 r=-0.5 c=-16.7 ex=16.2; matrix_question_npi_licensor_present r=-8.0 c=-19.8 ex=11.8; principle_A_domain_1 r=13.0 c=2.8 ex=10.2

## EWoK
- shared subtasks: 11
- mean clean delta: 0.235
- mean reinvest delta: -3.727
- mean excess reinvest-clean: -3.962
- delta correlation clean vs reinvest: -0.015
- same-sign fraction: 0.364
- largest negative excess: spatial-relations r=-15.0 c=2.1 ex=-17.1; material-dynamics r=-7.0 c=6.0 ex=-13.0; physical-relations r=-13.0 c=-0.9 ex=-12.1; physical-interactions r=-5.0 c=3.9 ex=-8.9; physical-dynamics r=0.0 c=8.0 ex=-8.0
- largest positive excess: material-properties r=1.0 c=-13.8 ex=14.8; quantitative-properties r=10.0 c=3.3 ex=6.7; social-interactions r=0.0 c=-4.4 ex=4.4; social-relations r=2.0 c=2.3 ex=-0.3; agent-properties r=-3.0 c=0.1 ex=-3.1

## Supplement
- shared subtasks: 5
- mean clean delta: -1.334
- mean reinvest delta: -3.200
- mean excess reinvest-clean: -1.866
- delta correlation clean vs reinvest: 0.822
- same-sign fraction: 0.600
- largest negative excess: hypernym r=-4.0 c=1.2 ex=-5.2; subject_aux_inversion r=0.0 c=2.0 ex=-2.0; qa_congruence_easy r=-8.0 c=-6.2 ex=-1.8; qa_congruence_tricky r=-4.0 c=-3.6 ex=-0.4; turn_taking r=0.0 c=0.0 ex=0.0
- largest positive excess: turn_taking r=0.0 c=0.0 ex=0.0; qa_congruence_tricky r=-4.0 c=-3.6 ex=-0.4; qa_congruence_easy r=-8.0 c=-6.2 ex=-1.8; subject_aux_inversion r=0.0 c=2.0 ex=-2.0; hypernym r=-4.0 c=1.2 ex=-5.2

Interpretation: where signs/magnitudes match clean-Qwen, the seed spread is likely inherited from the base recipe. Where reinvest has much more negative excess, the compact-view reinvestment route may be amplifying or changing instability. The running matched sparse trajectories are still needed for a time-resolved answer.

Machine-readable output: `experiments/archive/frontier_consolidation/data/reinvest_vs_clean_seed_fingerprint/reinvest_vs_clean_seed_fingerprint.json`
