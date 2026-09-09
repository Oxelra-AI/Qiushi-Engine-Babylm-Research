# relation filtered pair pools — no-update four-cell calibration on active relation pairs

Created: 2026-08-31T04:28:14Z  Runtime: 1019.3 s

## What was scored
For each pair, the scorer masked the saved target token positions and computed `M = s(Ca,Ta)+s(Cb,Tb)-s(Ca,Tb)-s(Cb,Ta)` with frozen checkpoint weights. Two controls used same-stratum target and context permutations. No training or optimizer update was performed.

Selected pairs: **18643**; by family: `{'causal_connector': 5340, 'comparative': 77, 'negation': 3536, 'physical_change': 1282, 'spatial': 4583, 'temporal': 3825}`
Donor assignment: `{'pairs': 18651, 'assigned': 18643, 'failed': 8, 'donor_unique': 18557, 'level_counts': {'exact': 17480, 'loose': 1163}}`
Context verification: `{'input_pairs': 18643, 'good_pairs': 18643, 'bad_contexts': 0, 'bad_examples': []}`

## Arm means for four-cell margin
| margin | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---|---:|
| true_m | 11.181909923369748 | 11.166183382050349 | 11.202089025857957 | interleaved > compact > rowblock | 0.035905643807607746 |
| tperm_m | 0.2394037991922831 | 0.22336947077025335 | 0.22675537673600318 | compact > interleaved > rowblock | 0.01603432842202976 |
| cperm_m | 0.2085693600438904 | 0.2195664902845953 | 0.19459181774468146 | rowblock > compact > interleaved | 0.02497467253991384 |

## Family means for true four-cell margin
| family | n | compact | rowblock | interleaved | true rank | true range | target-perm range | context-perm range |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| causal_connector | 5340 | 11.805978553565367 | 11.745214989261305 | 11.828927756163068 | interleaved > compact > rowblock | 0.08371276690176366 | 0.0018275247300776343 | 0.019926617967753135 |
| comparative | 77 | 10.543833232061429 | 10.741617116303026 | 10.645770688988753 | rowblock > interleaved > compact | 0.19778388424159665 | 0.37115497325921987 | 0.2679774648957438 |
| negation | 3536 | 11.594942613318484 | 11.568008487837913 | 11.650892339323118 | interleaved > compact > rowblock | 0.08288385148520483 | 0.043482618214672114 | 0.04608536285887643 |
| physical_change | 1282 | 11.660672637331045 | 11.636478982371635 | 11.632307439840359 | compact > rowblock > interleaved | 0.028365197490686 | 0.030987433409216614 | 0.09086063719832582 |
| spatial | 4583 | 10.029904060040549 | 10.053905530344108 | 10.053811349583288 | rowblock > interleaved > compact | 0.024001470303559813 | 0.03275701222060251 | 0.04225177231388311 |
| temporal | 3825 | 11.161515387596344 | 11.169963690716761 | 11.154916675358134 | rowblock > compact > interleaved | 0.015047015358627291 | 0.03613637851344215 | 0.03545068681142689 |

## Relation-oriented paired deltas
### rowblock_minus_compact
- true_m: mean=-0.01572654131939787 stderr=0.013930657345309398 success_gt0=0.5019578394035294 n=18643
- tperm_m: mean=-0.01603432842202976 stderr=0.01393355474465318 success_gt0=0.49836399721074937 n=18643
- cperm_m: mean=0.010997130240704912 stderr=0.014141670623838426 success_gt0=0.5069463069248511 n=18643

### interleaved_minus_compact
- true_m: mean=0.020179102488209127 stderr=0.013927455897051183 success_gt0=0.5141339913104114 n=18643
- tperm_m: mean=-0.01264842245627991 stderr=0.013946541768336847 success_gt0=0.4975594056750523 n=18643
- cperm_m: mean=-0.013977542299208924 stderr=0.013986547181091753 success_gt0=0.4983103577750362 n=18643

### rowblock_minus_interleaved
- true_m: mean=-0.035905643807606996 stderr=0.012865332900958449 success_gt0=0.48844070160381914 n=18643
- tperm_m: mean=-0.0033859059657498484 stderr=0.013319834978303498 success_gt0=0.5000268197178566 n=18643
- cperm_m: mean=0.024974672539913836 stderr=0.013434999483476566 success_gt0=0.5028160703749397 n=18643

## Files
- JSON summary: `experiments/archive/representation_and_objectives/data/relation_fourcell_calibration/fourcell_calibration_summary.json`
- Pair scores: `experiments/archive/representation_and_objectives/data/relation_fourcell_calibration/fourcell_pair_scores.jsonl`
- CSV: `experiments/archive/representation_and_objectives/data/relation_fourcell_calibration/fourcell_pair_scores_summary.csv`
