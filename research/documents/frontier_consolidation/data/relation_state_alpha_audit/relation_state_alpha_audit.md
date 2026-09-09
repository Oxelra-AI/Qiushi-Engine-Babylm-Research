# private scale endpoint vs mechanism synthesis relation/state alpha audit

Status: **COMPLETE**

## Summary vs protected chck82 anchor

| candidate | column | delta score pts | gain | loss | net | changed | retention |
|---|---|---:|---:|---:|---:|---:|---:|
| ordinary86_backbone | EWoK | +0.084884 | 275 | 250 | +25 | 525 | 0.934280 |
| shuffled86_private | EWoK | +1.384817 | 709 | 677 | +32 | 1386 | 0.822029 |
| coherent86_alpha0p5 | EWoK | -0.059339 | 83 | 73 | +10 | 156 | 0.980810 |
| coherent86_alpha0p75 | EWoK | -0.035837 | 117 | 113 | +4 | 230 | 0.970294 |
| coherent86_alpha1 | EWoK | -0.149427 | 147 | 149 | -2 | 296 | 0.960831 |
| ordinary86_backbone | Entity | +0.268498 | 145 | 124 | +21 | 269 | 0.934426 |
| shuffled86_private | Entity | -1.542380 | 218 | 291 | -73 | 509 | 0.846113 |
| coherent86_alpha0p5 | Entity | -0.083845 | 33 | 38 | -5 | 71 | 0.979905 |
| coherent86_alpha0p75 | Entity | +0.008177 | 50 | 54 | -4 | 104 | 0.971444 |
| coherent86_alpha1 | Entity | +0.125905 | 73 | 66 | +7 | 139 | 0.965098 |

## Top subfamily movements for alpha endpoints

### coherent86_alpha0p5 / EWoK

| subfamily | n | gain | loss | net | net/100 |
|---|---:|---:|---:|---:|---:|
| social-interactions | 294 | 7 | 4 | +3 | +1.020 |
| quantitative-properties | 314 | 5 | 8 | -3 | -0.955 |
| social-relations | 1548 | 24 | 10 | +14 | +0.904 |
| physical-dynamics | 120 | 2 | 3 | -1 | -0.833 |
| physical-interactions | 556 | 0 | 3 | -3 | -0.540 |
| social-properties | 328 | 4 | 5 | -1 | -0.305 |
| material-dynamics | 770 | 13 | 11 | +2 | +0.260 |
| spatial-relations | 490 | 5 | 6 | -1 | -0.204 |
| agent-properties | 2210 | 20 | 20 | +0 | +0.000 |
| physical-relations | 818 | 2 | 2 | +0 | +0.000 |

### coherent86_alpha0p5 / Entity

| subfamily | n | gain | loss | net | net/100 |
|---|---:|---:|---:|---:|---:|
| move_contents|3_ops | 406 | 1 | 5 | -4 | -0.985 |
| ambiref|2_ops | 413 | 5 | 1 | +4 | +0.969 |
| move_contents|4_ops | 353 | 4 | 1 | +3 | +0.850 |
| ambiref|5_ops | 123 | 0 | 1 | -1 | -0.813 |
| ambiref|0_ops | 508 | 3 | 6 | -3 | -0.591 |
| ambiref|3_ops | 409 | 1 | 3 | -2 | -0.489 |
| regular|1_ops | 409 | 1 | 3 | -2 | -0.489 |
| move_contents|1_ops | 437 | 2 | 0 | +2 | +0.458 |
| move_contents|2_ops | 399 | 2 | 1 | +1 | +0.251 |
| regular|2_ops | 405 | 1 | 2 | -1 | -0.247 |

### coherent86_alpha0p75 / EWoK

| subfamily | n | gain | loss | net | net/100 |
|---|---:|---:|---:|---:|---:|
| social-properties | 328 | 4 | 8 | -4 | -1.220 |
| material-properties | 170 | 3 | 1 | +2 | +1.176 |
| physical-dynamics | 120 | 2 | 3 | -1 | -0.833 |
| social-interactions | 294 | 9 | 7 | +2 | +0.680 |
| social-relations | 1548 | 27 | 18 | +9 | +0.581 |
| spatial-relations | 490 | 6 | 8 | -2 | -0.408 |
| quantitative-properties | 314 | 8 | 9 | -1 | -0.318 |
| material-dynamics | 770 | 20 | 21 | -1 | -0.130 |
| physical-relations | 818 | 5 | 4 | +1 | +0.122 |
| agent-properties | 2210 | 29 | 30 | -1 | -0.045 |

### coherent86_alpha0p75 / Entity

| subfamily | n | gain | loss | net | net/100 |
|---|---:|---:|---:|---:|---:|
| move_contents|3_ops | 406 | 2 | 7 | -5 | -1.232 |
| regular|5_ops | 94 | 2 | 1 | +1 | +1.064 |
| regular|2_ops | 405 | 1 | 5 | -4 | -0.988 |
| ambiref|0_ops | 508 | 3 | 8 | -5 | -0.984 |
| ambiref|2_ops | 413 | 6 | 2 | +4 | +0.969 |
| move_contents|5_ops | 116 | 1 | 0 | +1 | +0.862 |
| move_contents|4_ops | 353 | 5 | 2 | +3 | +0.850 |
| ambiref|5_ops | 123 | 1 | 2 | -1 | -0.813 |
| ambiref|3_ops | 409 | 1 | 4 | -3 | -0.733 |
| regular|3_ops | 425 | 5 | 2 | +3 | +0.706 |

### coherent86_alpha1 / EWoK

| subfamily | n | gain | loss | net | net/100 |
|---|---:|---:|---:|---:|---:|
| physical-dynamics | 120 | 2 | 4 | -2 | -1.667 |
| social-properties | 328 | 5 | 9 | -4 | -1.220 |
| material-properties | 170 | 3 | 1 | +2 | +1.176 |
| quantitative-properties | 314 | 9 | 12 | -3 | -0.955 |
| physical-interactions | 556 | 9 | 5 | +4 | +0.719 |
| social-relations | 1548 | 40 | 29 | +11 | +0.711 |
| social-interactions | 294 | 10 | 8 | +2 | +0.680 |
| spatial-relations | 490 | 6 | 9 | -3 | -0.612 |
| agent-properties | 2210 | 34 | 42 | -8 | -0.362 |
| physical-relations | 818 | 6 | 8 | -2 | -0.244 |

### coherent86_alpha1 / Entity

| subfamily | n | gain | loss | net | net/100 |
|---|---:|---:|---:|---:|---:|
| move_contents|5_ops | 116 | 3 | 0 | +3 | +2.586 |
| move_contents|3_ops | 406 | 3 | 8 | -5 | -1.232 |
| regular|5_ops | 94 | 1 | 2 | -1 | -1.064 |
| move_contents|0_ops | 516 | 7 | 2 | +5 | +0.969 |
| ambiref|2_ops | 413 | 7 | 3 | +4 | +0.969 |
| ambiref|5_ops | 123 | 1 | 2 | -1 | -0.813 |
| regular|2_ops | 405 | 3 | 6 | -3 | -0.741 |
| regular|1_ops | 409 | 7 | 4 | +3 | +0.733 |
| regular|3_ops | 425 | 5 | 2 | +3 | +0.706 |
| ambiref|0_ops | 508 | 5 | 8 | -3 | -0.591 |

## Scientific reading

- The frozen-anchor fast path was motivated by relation/state retention. On EWoK/Entity specifically, alpha endpoints are modest: alpha0.5 has EWoK gain but Entity loss; alpha0.75 is near-neutral; alpha1 is near-neutral/slightly positive depending on column.
- The relation/state movement is far smaller than COMPS/BLiMP churn and far smaller than GlobalPIQA's apparent score swing from only five alpha-sensitive examples, so current alpha endpoints should be read as endpoint redistribution rather than a solved relation/state protection mechanism.

JSON: `experiments/archive/frontier_consolidation/data/relation_state_alpha_audit/relation_state_alpha_audit.json`
