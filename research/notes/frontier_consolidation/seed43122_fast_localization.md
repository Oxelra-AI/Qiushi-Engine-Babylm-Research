# seed43122 fast localization — seed43122 fast/no-AoA localization
CPU-only comparison of existing prediction files for seed43122 and seed43022 compact_view_reinvest. This does not use the running official-rowcount AoA task and does not run new evaluation.

## Column-level agreement and flip summary
- BLiMP: n=13400, seed43022=66.634, seed43122=65.761, delta=-0.873, pred_agreement=81.14%, 43022_only=1322, 43122_only=1205
- Supplement: n=250, seed43022=66.400, seed43122=63.200, delta=-3.200, pred_agreement=84.00%, 43022_only=24, 43122_only=16
- EWoK: n=1100, seed43022=53.091, seed43122=49.364, delta=-3.727, pred_agreement=64.09%, 43022_only=218, 43122_only=177
- Entity_fast: n=2238, seed43022=27.927, seed43122=26.988, delta=-0.938, pred_agreement=60.99%, 43022_only=220, 43122_only=199
- Entity_full: n=6780, seed43022=27.050, seed43122=26.386, delta=-0.664, pred_agreement=60.12%, 43022_only=633, 43122_only=588
- COMPS: n=91028, seed43022=52.876, seed43122=52.687, delta=-0.189, pred_agreement=60.35%, 43022_only=18132, 43122_only=17960
- GlobalPIQA_parallel: n=103, seed43022=25.243, seed43122=24.272, delta=-0.971, pred_agreement=75.73%, 43022_only=8, 43122_only=7
- GlobalPIQA_nonparallel: n=100, seed43022=46.000, seed43122=46.000, delta=+0.000, pred_agreement=74.00%, 43022_only=13, 43122_only=13

## Worst subtask slices by column

### BLiMP
- principle_A_reconstruction: n=200, delta=-29.50, 43022=51.50, 43122=22.00, agree=66.5%, net=-59
- superlative_quantifiers_1: n=200, delta=-22.00, 43022=88.00, 43122=66.00, agree=74.0%, net=-44
- principle_A_c_command: n=200, delta=-14.50, 43022=53.50, 43122=39.00, agree=64.5%, net=-29
- wh_questions_object_gap: n=200, delta=-14.00, 43022=56.00, 43122=42.00, agree=72.0%, net=-28
- sentential_subject_island: n=200, delta=-13.00, 43022=35.00, 43122=22.00, agree=74.0%, net=-26
- distractor_agreement_relative_clause: n=200, delta=-12.00, 43022=35.50, 43122=23.50, agree=74.0%, net=-24
- left_branch_island_simple_question: n=200, delta=-11.50, 43022=59.50, 43122=48.00, agree=74.5%, net=-23
- sentential_negation_npi_scope: n=200, delta=-9.00, 43022=72.50, 43122=63.50, agree=75.0%, net=-18

### Supplement
- qa_congruence_easy: n=50, delta=-8.00, 43022=74.00, 43122=66.00, agree=80.0%, net=-4
- hypernym: n=50, delta=-4.00, 43022=56.00, 43122=52.00, agree=76.0%, net=-2
- qa_congruence_tricky: n=50, delta=-4.00, 43022=54.00, 43122=50.00, agree=84.0%, net=-2
- subject_aux_inversion: n=50, delta=+0.00, 43022=80.00, 43122=80.00, agree=88.0%, net=0
- turn_taking: n=50, delta=+0.00, 43022=68.00, 43122=68.00, agree=92.0%, net=0

### EWoK
- spatial-relations: n=100, delta=-15.00, 43022=43.00, 43122=28.00, agree=69.0%, net=-15
- physical-relations: n=100, delta=-13.00, 43022=62.00, 43122=49.00, agree=67.0%, net=-13
- social-properties: n=100, delta=-11.00, 43022=54.00, 43122=43.00, agree=59.0%, net=-11
- material-dynamics: n=100, delta=-7.00, 43022=55.00, 43122=48.00, agree=47.0%, net=-7
- physical-interactions: n=100, delta=-5.00, 43022=57.00, 43122=52.00, agree=63.0%, net=-5
- agent-properties: n=100, delta=-3.00, 43022=48.00, 43122=45.00, agree=73.0%, net=-3
- physical-dynamics: n=100, delta=+0.00, 43022=62.00, 43122=62.00, agree=68.0%, net=0
- social-interactions: n=100, delta=+0.00, 43022=52.00, 43122=52.00, agree=60.0%, net=0

### Entity_full
- ambiref_5_ops: n=123, delta=-8.13, 43022=34.15, 43122=26.02, agree=63.4%, net=-10
- move_contents_3_ops: n=406, delta=-8.13, 43022=29.56, 43122=21.43, agree=59.6%, net=-33
- regular_4_ops: n=388, delta=-7.99, 43022=30.67, 43122=22.68, agree=55.7%, net=-31
- move_contents_4_ops: n=353, delta=-4.82, 43022=30.03, 43122=25.21, agree=56.4%, net=-17
- regular_1_ops: n=409, delta=-4.65, 43022=19.32, 43122=14.67, agree=62.3%, net=-19
- ambiref_3_ops: n=409, delta=-4.40, 43022=26.89, 43122=22.49, agree=57.7%, net=-18
- ambiref_2_ops: n=413, delta=-4.12, 43022=23.97, 43122=19.85, agree=58.1%, net=-17
- regular_2_ops: n=405, delta=-3.95, 43022=22.22, 43122=18.27, agree=61.0%, net=-16

### COMPS
- wugs_dist_in_between: n=13896, delta=-5.96, 43022=59.77, 43122=53.81, agree=55.3%, net=-828
- base: n=49340, delta=+0.19, 43022=54.30, 43122=54.49, agree=62.8%, net=93
- wugs: n=13896, delta=+0.22, 43022=52.17, 43122=52.40, agree=62.0%, net=31
- wugs_dist_before: n=13896, delta=+3.83, 43022=41.64, 43122=45.47, agree=55.1%, net=532

### GlobalPIQA_parallel
- global_piqa_parallel: n=103, delta=-0.97, 43022=25.24, 43122=24.27, agree=75.7%, net=-1

### GlobalPIQA_nonparallel
- global_piqa_nonparallel: n=100, delta=+0.00, 43022=46.00, 43122=46.00, agree=74.0%, net=0

## Aggregate reading
Weighted score-sum delta across BLiMP+Supplement+EWoK+full Entity+COMPS+GlobalPIQA mean (excluding Reading, SuperGLUE, AoA): -9.138514.
The weak seed43122 fast surface is not solely a one-slice artifact. It has a large EWoK deficit, broad BLiMP/Supplement deficits, a full-Entity deficit, a small COMPS deficit, and one lost GlobalPIQA-parallel example; Reading is the only observed improvement. This supports treating seed43022 as an upper-tail or fragile endpoint unless official-rowcount AoA is strongly positive for seed43122.

Machine-readable output: `experiments/archive/frontier_consolidation/data/seed43122_fast_localization/seed43122_fast_localization.json`.
