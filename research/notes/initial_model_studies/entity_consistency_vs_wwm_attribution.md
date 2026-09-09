# downstream counterfactual materialization — Entity Consistency absolute attribution vs ordinary WWM

Evidence JSON: `experiments/archive/initial_model_studies/data/entity_consistency_vs_wwm_attribution.json`

Higher is better for all columns, including Reading. Deltas are arm minus ordinary WWM at matched exposure.

## 10M: arm minus protected ordinary WWM

| arm | BLiMP | Supp | Entity | COMPS | GPIQA mean | Reading mean | six-sum | target E+G | guard S+B+R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed1 consistency | +1.28 | +0.79 | +0.05 | +0.53 | +1.40 | +0.32 | +4.37 | +1.45 | +2.39 |
| seed1 shuffled | +1.01 | +0.32 | -0.01 | +0.30 | -0.57 | +0.46 | +1.50 | -0.58 | +1.78 |
| seed2 consistency | +0.34 | -0.01 | -0.43 | +0.07 | +3.41 | -0.76 | +2.62 | +2.98 | -0.43 |
| seed2 shuffled | +0.57 | -1.94 | -0.31 | +0.27 | +1.93 | -0.73 | -0.21 | +1.62 | -2.10 |

## 20M: arm minus protected ordinary WWM

| arm | BLiMP | Supp | Entity | COMPS | GPIQA mean | Reading mean | six-sum | target E+G | guard S+B+R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed1 consistency | +0.78 | -0.62 | +0.71 | -0.30 | +3.40 | -0.08 | +3.89 | +4.11 | +0.08 |
| seed1 shuffled | -1.26 | +0.36 | +0.32 | -0.27 | -0.52 | +0.13 | -1.24 | -0.20 | -0.77 |
| seed2 consistency | -0.93 | -1.96 | +0.52 | -0.02 | -1.05 | -0.33 | -3.78 | -0.54 | -3.22 |
| seed2 shuffled | -0.42 | -1.79 | +0.31 | -0.45 | -3.02 | +0.69 | -4.68 | -2.71 | -1.52 |

## Interpretation

- The wrong-pair `shuffled_pair` arm itself often changes scores substantially relative to ordinary WWM, especially GlobalPIQA and Reading, so it is not a neutral no-auxiliary baseline.
- Entity Mention Consistency remains closed for 100M scaling: its consistency-minus-shuffled effect did not replicate and its absolute pattern versus WWM is not a stable guard-safe broad improvement.
- The next route should use a structure-destroyed control that preserves lexical/frequency statistics but does not impose a harmful wrong-pair contrast objective; ordinary WWM must remain an explicit absolute baseline.
