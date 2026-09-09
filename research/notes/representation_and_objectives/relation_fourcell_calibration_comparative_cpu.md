# relation filtered pair pools — no-update four-cell calibration on active relation pairs

Created: 2026-08-31T04:16:49Z  Runtime: 123.1 s

## What was scored
For each pair, the scorer masked the saved target token positions and computed `M = s(Ca,Ta)+s(Cb,Tb)-s(Ca,Tb)-s(Cb,Ta)` with frozen checkpoint weights. Two controls used same-stratum target and context permutations. No training or optimizer update was performed.

Selected pairs: **77**; by family: `{'comparative': 77}`
Donor assignment: `{'pairs': 77, 'assigned': 77, 'failed': 0, 'donor_unique': 76, 'level_counts': {'exact': 47, 'loose': 30}}`
Context verification: `{'input_pairs': 77, 'good_pairs': 77, 'bad_contexts': 0, 'bad_examples': []}`

## Arm means for four-cell margin
| margin | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---|---:|
| true_m | 10.536878460994014 | 10.744263581278457 | 10.656531158447653 | rowblock > interleaved > compact | 0.20738512028444234 |
| tperm_m | 0.29995902366452404 | -0.07201253825967963 | 0.2621189440999712 | compact > interleaved > rowblock | 0.37197156192420366 |
| cperm_m | 0.38528648173654234 | 0.12311741013031502 | 0.29142648059052306 | compact > interleaved > rowblock | 0.2621690716062273 |

## Family means for true four-cell margin
| family | n | compact | rowblock | interleaved | true rank | true range | target-perm range | context-perm range |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| comparative | 77 | 10.536878460994014 | 10.744263581278457 | 10.656531158447653 | rowblock > interleaved > compact | 0.20738512028444234 | 0.37197156192420366 | 0.2621690716062273 |

## Relation-oriented paired deltas
### rowblock_minus_compact
- true_m: mean=0.20738512028444123 stderr=0.22150513534203903 success_gt0=0.5454545454545454 n=77
- tperm_m: mean=-0.37197156192420366 stderr=0.2319666426731524 success_gt0=0.44155844155844154 n=77
- cperm_m: mean=-0.2621690716062273 stderr=0.2463565486831871 success_gt0=0.4805194805194805 n=77

### interleaved_minus_compact
- true_m: mean=0.11965269745363817 stderr=0.2228761373607326 success_gt0=0.5324675324675324 n=77
- tperm_m: mean=-0.037840079564552806 stderr=0.22496215102210523 success_gt0=0.5324675324675324 n=77
- cperm_m: mean=-0.09386000114601928 stderr=0.2481112078621448 success_gt0=0.44155844155844154 n=77

### rowblock_minus_interleaved
- true_m: mean=0.08773242283080306 stderr=0.22342966956738747 success_gt0=0.5064935064935064 n=77
- tperm_m: mean=-0.33413148235965084 stderr=0.2065502766150058 success_gt0=0.44155844155844154 n=77
- cperm_m: mean=-0.16830907046020804 stderr=0.22309286533686715 success_gt0=0.44155844155844154 n=77

## Files
- JSON summary: `experiments/archive/representation_and_objectives/data/relation_fourcell_calibration_comparative_cpu/fourcell_calibration_summary.json`
- Pair scores: `experiments/archive/representation_and_objectives/data/relation_fourcell_calibration_comparative_cpu/fourcell_pair_scores.jsonl`
- CSV: `experiments/archive/representation_and_objectives/data/relation_fourcell_calibration_comparative_cpu/fourcell_pair_scores_summary.csv`
