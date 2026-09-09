# relation filtered pair pools — stricter subpool counts inside active relation pairs

Total pairs: **18651**

| subset | n | frac | causal | spatial | temporal | negation | physical | comparative | top targets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| all | 18651 | 1.000 | 5340 | 4585 | 3826 | 3538 | 1285 | 77 | give:32, table:32, real:32, five:32, live:32 |
| non_generic | 16579 | 0.889 | 4199 | 4322 | 3730 | 3131 | 1132 | 65 | i'm:32, i'd:32, i've:32, there's:32, you're:32 |
| local_jaccard_ge_0p10 | 1803 | 0.097 | 653 | 446 | 347 | 292 | 57 | 8 | ages:10, way:8, city:8, stairs:8, government:7 |
| local_jaccard_ge_0p15 | 429 | 0.023 | 171 | 104 | 75 | 68 | 10 | 1 | ages:6, way:5, bad:4, fit:4, picture:4 |
| local_jaccard_ge_0p20 | 99 | 0.005 | 40 | 14 | 21 | 21 | 2 | 1 | gave:2, way:2, time:2, trusted:2, fit:2 |
| context_jaccard_ge_0p14 | 2435 | 0.131 | 1001 | 256 | 342 | 731 | 93 | 12 | i've:18, there's:15, bad:15, hafta:15, hadn't:15 |
| char_jaccard_ge_0p40 | 2640 | 0.142 | 819 | 568 | 627 | 470 | 147 | 9 | nothing:23, i'm:22, i'd:20, i've:19, able:18 |
| char_jaccard_ge_0p50 | 1335 | 0.072 | 433 | 277 | 310 | 243 | 67 | 5 | nothing:16, i'm:14, able:14, real:13, going:13 |
| edit_distance_le_0p60 | 1808 | 0.097 | 604 | 313 | 446 | 359 | 78 | 8 | really:27, nothing:23, actually:22, i'm:22, i'd:21 |
| same_suffix3 | 1063 | 0.057 | 284 | 162 | 367 | 208 | 37 | 5 | actually:22, doing:22, getting:22, really:21, looking:21 |
| antonym_hint | 2 | 0.000 | 1 | 0 | 0 | 1 | 0 | 0 | last:1, far:1, first:1, near:1 |
| non_generic_and_local_ge_0p10 | 1599 | 0.086 | 515 | 426 | 338 | 261 | 52 | 7 | ages:10, stairs:8, chance:7, clouds:7, streets:7 |
| non_generic_and_context_ge_0p14 | 2040 | 0.109 | 764 | 238 | 329 | 618 | 79 | 12 | i've:18, there's:15, hadn't:15, i'll:13, it's:13 |
| non_generic_and_char_ge_0p40 | 2441 | 0.131 | 705 | 546 | 622 | 426 | 134 | 8 | i'm:22, i'd:20, i've:19, nothing:18, there's:15 |
| candidate_competitive_broad | 5334 | 0.286 | 1683 | 1121 | 1150 | 1117 | 241 | 22 | i'm:25, there's:25, i've:24, i'd:23, hadn't:21 |
| candidate_competitive_strict | 1571 | 0.084 | 493 | 357 | 375 | 268 | 73 | 5 | nothing:17, i'm:14, taking:13, i'd:12, you're:11 |

Interpretation:
- If full no-update four-cell margins are huge and arm-separation does not survive permutation controls, the broad pool should not become a training objective.
- Any rebuilt pool should increase true competition of cross-targets: higher local-slot overlap, explicit relation-opposition pairs, or repeated target-alternative sets, while preserving legality and avoiding official-example wording.

JSON: `experiments/archive/representation_and_objectives/data/relation_pair_subpool_counts/relation_pair_subpool_counts.json`
