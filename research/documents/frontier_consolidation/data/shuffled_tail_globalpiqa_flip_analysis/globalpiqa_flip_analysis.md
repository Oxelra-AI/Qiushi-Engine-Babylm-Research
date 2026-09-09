# earlier analysis shuffled-tail GlobalPIQA flip analysis

Status: **COMPLETE**

## Scores
| model | parallel | nonparallel | GlobalPIQA |
|---|---:|---:|---:|
| protected_chck82 | 29/103 = 28.155340 | 47/100 = 47.000000 | 37.577670 |
| aligned_tail | 28/103 = 27.184466 | 47/100 = 47.000000 | 37.092233 |
| shuffled_tail | 30/103 = 29.126214 | 50/100 = 50.000000 | 39.563107 |

## Pairwise flips
| comparison | GP delta | improved | damaged | discordant | exact sign p |
|---|---:|---:|---:|---:|---:|
| shuffled_minus_protected | 1.985437 | 14 | 10 | 24 | 0.541256 |
| aligned_minus_protected | -0.485437 | 11 | 12 | 23 | 1.000000 |
| shuffled_minus_aligned | 2.470874 | 8 | 3 | 11 | 0.226562 |

Scientific reading: use this as a stability warning for the 41.984 endpoint hypothesis. A small positive Overall margin dominated by GlobalPIQA should be replayed and repeated before it replaces the bit-identical chck_82M carrier; it is not evidence for source-correspondence transfer.

JSON: `experiments/archive/frontier_consolidation/data/shuffled_tail_globalpiqa_flip_analysis/globalpiqa_flip_analysis.json`
