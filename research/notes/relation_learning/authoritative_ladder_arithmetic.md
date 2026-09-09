# authoritative ladder arithmetic: Authoritative Two-Seed Ladder Arithmetic

Source: `experiments/archive/functional_learning/data/strict_split_eval_admission_o_complete/strict_split_eval_admission.json`
and `experiments/archive/functional_learning/data/same_coordinate_with_ordinary_complete/same_coordinate_with_ordinary.json`

All values below are from the admitted strict-split evaluation surface. This document
should be the single source of truth for every ladder number in the report.

## Complete Seed64 Ladder

| rung | Overall | BLiMP | Supplement | EWoK | Entity | COMPS | SuperGLUE | GlobalPIQA | Reading | AoA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coherent86 | 42.02397 | 68.51 | 63.64 | 50.02 | 28.32 | 52.05 | 68.94573 | 38.565 | 8.165 | 0.0 |
| O64 | 42.09261 | 68.47 | 63.63 | 49.83 | 28.16 | 52.00 | 68.98845 | 39.535 | 8.22 | 0.0 |
| (M,S)64 | 42.20254 | 68.12 | 63.08 | 49.95 | 29.39 | 52.15 | 68.88784 | 40.05 | 8.195 | 0.0 |
| clean64 | 42.24641 | 68.26 | 63.28 | 49.82 | 29.40 | 52.16 | 69.04771 | 40.05 | 8.20 | 0.0 |

Exact Overall values: coherent86 42.023967991315104, O64 42.09260586321611,
(M,S)64 42.20253795433653, clean64 42.246412332209445.

## Complete Seed65 Ladder (earlier analysis admitted)

| rung | Overall | BLiMP | Supplement | EWoK | Entity | COMPS | SuperGLUE | GlobalPIQA | Reading | AoA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| O65 | 42.11592 | 68.52 | 63.62 | 49.81 | 28.09 | 52.00 | 69.27328 | 39.535 | 8.195 | 0.0 |
| (M,S)65 | 42.17887 | 68.09 | 63.09 | 49.82 | 29.29 | 52.14 | 68.93486 | 40.05 | 8.195 | 0.0 |
| clean65 | 42.23173 | 68.22 | 63.29 | 49.72 | 29.45 | 52.14 | 69.02058 | 40.05 | 8.195 | 0.0 |

Exact Overall values: O65 42.1159198215161, (M,S)65 42.17887384024569,
clean65 42.23173113265801.

## Complete Rung Increments

### Seed64

| increment | Overall | BLiMP | Supp | EWoK | Entity | COMPS | SuperGLUE | GPIQA | Read | AoA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coh→O | +0.069 | −0.04 | −0.01 | −0.19 | −0.16 | −0.05 | +0.043 | +0.970 | +0.055 | 0.0 |
| O→(M,S) | +0.110 | −0.35 | −0.55 | +0.12 | +1.23 | +0.15 | −0.101 | +0.515 | −0.025 | 0.0 |
| (M,S)→clean | +0.044 | +0.14 | +0.20 | −0.13 | +0.01 | +0.01 | +0.160 | 0.00 | +0.005 | 0.0 |
| O→clean | +0.154 | −0.21 | −0.35 | −0.01 | +1.24 | +0.16 | +0.059 | +0.515 | −0.020 | 0.0 |

### Seed65

| increment | Overall | BLiMP | Supp | EWoK | Entity | COMPS | SuperGLUE | GPIQA | Read | AoA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coh→O65 | +0.092 | +0.01 | −0.02 | −0.21 | −0.23 | −0.05 | +0.328 | +0.970 | +0.030 | 0.0 |
| O→(M,S) | +0.063 | −0.43 | −0.53 | +0.01 | +1.20 | +0.14 | −0.338 | +0.515 | 0.00 | 0.0 |
| (M,S)→clean | +0.053 | +0.13 | +0.20 | −0.10 | +0.16 | 0.00 | +0.086 | 0.00 | 0.00 | 0.0 |
| O→clean | +0.116 | −0.30 | −0.33 | −0.09 | +1.36 | +0.14 | −0.253 | +0.515 | 0.00 | 0.0 |

### Cross-seed O pair noise floor (identical recipe, two seeds)

| column | |O64−O65| |
|---|---:|
| BLiMP | 0.05 |
| Supplement | 0.01 |
| EWoK | 0.02 |
| Entity | 0.07 |
| COMPS | 0.00 |
| GlobalPIQA | 0.00 |
| Reading | 0.025 |
| AoA | 0.0 |
| SuperGLUE | 0.285 |
| Overall | 0.023 |

Max deterministic column delta: **0.07** (Entity).
SuperGLUE delta: **0.285** — no SuperGLUE difference below ~0.3 in the ladder may be
read as an effect.

## Key Observations

### O65 holds the highest SuperGLUE of the entire ladder

O65 SuperGLUE = 69.2733, above clean64 69.0477, clean65 69.0206, coherent86 68.9457,
and all other rungs. This compresses the seed65 O→(M,S) increment: (M,S)65's SuperGLUE
of 68.9349 is lower than O65's 69.2733 by −0.338 points, yet the Overall still rises
because Entity's +1.20 dominates. The seed65 gain over ordinary continuation therefore
**survives an unfavorable SuperGLUE draw and rests entirely on deterministic columns**.

### Every rung is positive at both seeds

| rung | seed64 Overall | seed65 Overall |
|---|---:|---:|
| coherent→O | +0.069 | +0.092 |
| O→(M,S) | +0.110 | +0.063 |
| (M,S)→clean | +0.044 | +0.053 |
| O→clean | +0.154 | +0.116 |

### Entity drives the O→clean gain

Entity alone supplies **+1.24/+1.36** of the O→clean Overall gain of **+0.154/+0.116**
(entity is the dominant positive component). GlobalPIQA contributes +0.515/+0.515
(identical, concentrated in a few fixed items). BLiMP+Supplement cost
**−0.56/−0.63**. SuperGLUE contributes **+0.059/−0.253** (stochastic; the seed65
draw is against us yet the ladder holds).

### Replication scope

- Two seeds of the recipe's own training randomness on one frozen trunk lineage
- The trunk's private-phase effect (slow+private initialization) did not itself
  replicate across trunk seed — the private phase was trained once and frozen
- Transfer to another trunk is untested
- Clean differs from (M,S) in both teacher KL constraint and additional ordinary-mask
  presentations; a no-teacher control was never run; exploration is closed
