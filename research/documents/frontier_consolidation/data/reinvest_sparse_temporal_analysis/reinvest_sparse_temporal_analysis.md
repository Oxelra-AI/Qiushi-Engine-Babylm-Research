# earlier analysis — compact_view_reinvest sparse temporal seed analysis

The selected sparse-slice evaluation completed with failures=0. This note summarizes seed43122-minus-seed43022 before the clean-Qwen DiD control arrives.

## Task-group seed gaps by exposure (seed43122 - seed43022)
### 1M
- blimp_worst: -7.000 (43022=47.550, 43122=40.550)
- blimp_control: +7.938 (43022=58.562, 43122=66.500)
- supplement_all: -6.000 (43022=56.400, 43122=50.400)
- ewok_all: +0.455 (43022=50.182, 43122=50.636)
- entity_full: +0.620 (43022=16.258, 43122=16.878)
### 10M
- blimp_worst: +2.150 (43022=40.100, 43122=42.250)
- blimp_control: +1.688 (43022=55.875, 43122=57.562)
- supplement_all: -3.200 (43022=57.600, 43122=54.400)
- ewok_all: +0.000 (43022=49.818, 43122=49.818)
- entity_full: -1.334 (43022=17.220, 43122=15.886)
### 40M
- blimp_worst: -1.300 (43022=46.450, 43122=45.150)
- blimp_control: +1.062 (43022=55.562, 43122=56.625)
- supplement_all: +2.400 (43022=60.800, 43122=63.200)
- ewok_all: -2.455 (43022=52.182, 43122=49.727)
- entity_full: -2.903 (43022=26.425, 43122=23.523)
### 100M
- blimp_worst: -14.050 (43022=57.150, 43122=43.100)
- blimp_control: +10.062 (43022=51.750, 43122=61.812)
- supplement_all: -3.200 (43022=66.400, 43122=63.200)
- ewok_all: -3.727 (43022=53.091, 43122=49.364)
- entity_full: -1.460 (43022=27.746, 43122=26.285)

## Late changes 40M -> 100M
- blimp_worst: final=-14.050, late_change=-12.750
- blimp_control: final=+10.062, late_change=+9.000
- supplement_all: final=-3.200, late_change=-5.600
- ewok_all: final=-3.727, late_change=-1.273
- entity_full: final=-1.460, late_change=+1.443

## Most negative final subtasks
- blimp_worst / principle_A_reconstruction: final=-29.500, by_exp={'1': 0.0, '10': -2.0, '40': -8.5, '100': -29.5}
- blimp_worst / superlative_quantifiers_1: final=-22.000, by_exp={'1': -51.0, '10': 5.5, '40': 19.0, '100': -22.0}
- ewok_all / spatial-relations: final=-15.000, by_exp={'1': 0.0, '10': -1.0, '40': -4.0, '100': -15.0}
- blimp_worst / principle_A_c_command: final=-14.500, by_exp={'1': 27.5, '10': -2.5, '40': 6.5, '100': -14.5}
- blimp_worst / wh_questions_object_gap: final=-14.000, by_exp={'1': -18.0, '10': 3.5, '40': -6.5, '100': -14.0}
- blimp_worst / sentential_subject_island: final=-13.000, by_exp={'1': -2.0, '10': 4.5, '40': -9.0, '100': -13.0}
- ewok_all / physical-relations: final=-13.000, by_exp={'1': -5.0, '10': 5.0, '40': 4.0, '100': -13.0}
- blimp_worst / distractor_agreement_relative_clause: final=-12.000, by_exp={'1': -1.0, '10': -0.5, '40': -3.0, '100': -12.0}
- blimp_worst / left_branch_island_simple_question: final=-11.500, by_exp={'1': -5.5, '10': 9.5, '40': -12.5, '100': -11.5}
- ewok_all / social-properties: final=-11.000, by_exp={'1': 0.0, '10': 1.0, '40': -10.0, '100': -11.0}
- blimp_worst / sentential_negation_npi_scope: final=-9.000, by_exp={'1': -11.0, '10': -3.0, '40': 7.5, '100': -9.0}
- entity_full / ambiref_5_ops: final=-8.130, by_exp={'1': -0.8130081300813004, '10': 0.0, '40': -11.382113821138212, '100': -8.13008130081301}

## EWoK coordinate caveat
- fast temporal 100M EWoK delta: -3.727
- full official 7618-row 100M EWoK delta: -1.645
The fast temporal EWoK slice exaggerates the final official EWoK seed gap; use the full official repair for score arithmetic and the fast slice only for temporal patterning.

## Current interpretation
The reinvest trajectory is not one simple early-undertraining failure. It is task-specific at 1M and becomes sharply late-divergent on selected BLiMP/Supplement slices. Whether that late divergence is inherited from clean-Qwen or treatment-specific awaits the matched clean sparse control.

Machine-readable output: `experiments/archive/frontier_consolidation/data/reinvest_sparse_temporal_analysis/reinvest_sparse_temporal_analysis.json`
