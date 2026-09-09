# seed variance fast and dynamics repaired — repaired seed variance before full seed43122 delivery

Machine-readable JSON: `experiments/archive/representation_and_objectives/data/seed_variance_fast_and_dynamics_repaired/seed_variance_fast_and_dynamics.json`
Worst-subtask CSV: `experiments/archive/representation_and_objectives/data/seed_variance_fast_and_dynamics_repaired/seed_variance_worst_fast_subtasks.csv`

## Controlled comparison
- Same 100M stream hash: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691` vs `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`; same=True.
- Recipe differences other than initialization/training RNG seeds: `{}`.
- Order manifest differences: `{}`.
- Shared data-order seed: seed43022/seed43122=43/43; changed initialization/training RNG seeds: seed43022=43022/43023; seed43122=43122/43123.

## Known fast endpoint movement (seed43122 − seed43022)
- BLiMP: -0.8800 (equal7 contribution -0.1257)
- Supplement: -3.2000 (equal7 contribution -0.4571)
- EWoK: -3.7300 (equal7 contribution -0.5329)
- Entity: -1.3900 (equal7 contribution -0.1986)
- COMPS: -0.4300 (equal7 contribution -0.0614)
- GlobalPIQA_mean: -0.4850 (equal7 contribution -0.0693)
- Reading: +0.6250 (equal7 contribution +0.0893)
- Equal7: -1.3557 (42.932857 vs 44.288571).

## Training dynamics
- Final-step training loss: seed43022 2.565617, seed43122 2.641525, final-step delta +0.075908; last-100 matched-step mean delta +0.000336.
- Mean loss over all matched steps delta seed43122−seed43022: -0.003585.
- The late training loss is essentially tied, and mean matched-step loss is slightly lower for seed43122; the weaker fast endpoint is therefore not explained by a simple aggregate MLM-loss disadvantage. Loss still cannot predict the full official vector.

## Item/subfamily concentration
- EWoK: agreement 0.6409, only430_correct=218, only431_correct=177, net -41; worst subtasks: spatial-relations -15.00; physical-relations -13.00; social-properties -11.00; material-dynamics -7.00; physical-interactions -5.00
- Supplement: agreement 0.8400, only430_correct=24, only431_correct=16, net -8; worst subtasks: qa_congruence_easy -8.00; hypernym -4.00; qa_congruence_tricky -4.00; subject_aux_inversion +0.00; turn_taking +0.00
- Entity: agreement 0.6099, only430_correct=220, only431_correct=199, net -21; worst subtasks: regular_4_ops -7.99; regular_1_ops -4.65; regular_2_ops -3.95; regular_3_ops -2.59; regular_5_ops +0.00
- BLiMP: agreement 0.8114, only430_correct=1322, only431_correct=1205, net -117; worst subtasks: principle_A_reconstruction -29.50; superlative_quantifiers_1 -22.00; principle_A_c_command -14.50; wh_questions_object_gap -14.00; sentential_subject_island -13.00
- COMPS: agreement 0.6035, only430_correct=18132, only431_correct=17960, net -172; worst subtasks: wugs_dist_in_between -5.96; base +0.19; wugs +0.22; wugs_dist_before +3.83
- GlobalPIQA_parallel: agreement 0.7573, only430_correct=8, only431_correct=7, net -1; worst subtasks: 
- GlobalPIQA_nonparallel: agreement 0.7400, only430_correct=13, only431_correct=13, net 0; worst subtasks: 

## Sensitivity only, not final
If fast deltas were applied to seed43022's official full columns while holding SuperGLUE and AoA equal, the score would be 40.978690 (margin -0.821310). This is not a result; the managed full evaluation decides the actual seed43122 coordinate.

## Scientific interpretation
The known seed43122 weakness is already visible in deterministic zero-shot/Reading outputs under identical data, tokenizer, architecture, and schedule, and is concentrated most strongly in EWoK and Supplement, with Entity/BLiMP secondary. Because SuperGLUE fine-tuning and official AoA are still pending, this analysis should guide interpretation rather than trigger ad hoc repair. If seed43122 falls below the leader, the next work should compare full components and possibly repeated downstream fine-tuning/eval seeds before changing the pretraining recipe. If it clears the leader, both endpoint coordinates should be frozen and the prepared adjacency-broken control becomes the key mechanism test.
