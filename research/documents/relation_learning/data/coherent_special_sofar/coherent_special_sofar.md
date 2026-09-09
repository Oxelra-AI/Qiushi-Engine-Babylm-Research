# format attribution controls coherent-unsplit-special evidence so far

This record reads already written files before interpreting the other format arms.

## Training invariants

| seed | updates | words | first target ratio | last target ratio | initial CE diff | final coherent readout KL at scale1.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 98097 | 101 | 3992800 | 0.14948172394980905 | 0.15190891114925795 | 0.0 | 0.003226861357688904 |
| 98098 | 101 | 3992800 | 0.1523678792038435 | 0.15190891114925795 | 0.0 | 0.0037048112135380507 |

## Available column movement

| seed | column | score | Δ vs chck82 | Δ vs coherent86_s43022 | relation to coherent two-seed band |
|---:|---|---:|---:|---:|---|
| 98097 | BLiMP | 68.870000 | +0.378716 | +0.360000 | above coherent band |
| 98097 | Supplement | 60.990000 | -1.947811 | -2.650000 | below coherent band |
| 98097 | EWoK | 49.690000 | -0.365453 | -0.330000 | below coherent band |
| 98097 | Entity | 27.980000 | -0.334042 | -0.340000 | inside coherent band |
| 98097 | COMPS | 52.260000 | +0.068825 | +0.210000 | above coherent band |
| 98097 | GlobalPIQA | 36.590000 | -0.987670 | -1.975000 | inside coherent band |
| 98097 | Reading | 8.120000 | -0.028714 | -0.045000 | below coherent band |
| 98097 | cheap7 | 43.500000 | -0.459450 | -0.681429 | below coherent86_s43022 |
| 98098 | BLiMP | 68.810000 | +0.318716 | +0.300000 | above coherent band |
| 98098 | Supplement | 60.740000 | -2.197811 | -2.900000 | below coherent band |
| 98098 | EWoK | 49.220000 | -0.835453 | -0.800000 | below coherent band |
| 98098 | Entity | 28.720000 | +0.405958 | +0.400000 | above coherent band |
| 98098 | COMPS | 52.260000 | +0.068825 | +0.210000 | above coherent band |
| 98098 | GlobalPIQA | 39.050000 | +1.472330 | +0.485000 | above coherent band |
| 98098 | Reading | 8.065000 | -0.083714 | -0.100000 | below coherent band |
| 98098 | cheap7 | 43.837857 | -0.121593 | -0.343571 | below coherent86_s43022 |

## Interpretation

Seed 98097 has clean training invariants and a completed cheap7 score of 43.5000, which is -0.45945 against chck82 and -0.68143 against coherent86_s43022.  Its Supplement loss is -1.9478 against chck82 and -2.6500 against coherent86_s43022, while BLiMP rises +0.3787 against chck82.  Seed 98098 has clean training invariants and the available columns repeat the same main face: BLiMP +0.3187 against chck82, Supplement -2.1978, EWoK -0.8355, Entity +0.4060.  Because the repeated movement includes a large damaging Supplement/EWoK face, coherent-unsplit-special does not enter a composed candidate.  The result motivates the short isolated no-gradient readout and a no-special earlier analysis-trainer control before attributing the loss specifically to special tokens.

JSON: `experiments/archive/relation_learning/data/coherent_special_sofar/coherent_special_sofar.json`
