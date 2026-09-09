# lead factorial route reassessment MLM loader exposure audit for source wide skeleton recurrence integrated pools

This is a CPU/static audit of the exact DeBERTa MLM data-loader geometry. It is not training or evaluation.

Tokenizer JSON SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`; seq_len=256.

## Pool-level exposure
| variant | rows | words | raw tokens | active tokens | tokens/word active | trunc rows | trunc tokens | candidate groups | E[masked tokens] p=.15 | pair full visible | changed active source/view tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 64740 | 10000000 | 14664519 | 14294893 | 1.429489 | 15117 (23.35%) | 369626 | 9802194 | 2144233.9 | 99.55% | 359905/247289 |
| prefix_repeat | 64740 | 10000000 | 14633230 | 14264083 | 1.426408 | 15084 (23.30%) | 369147 | 9802439 | 2139612.4 | 99.84% | 359964/216436 |
| content_spread | 64740 | 10000000 | 14665547 | 14295757 | 1.429576 | 15122 (23.36%) | 369790 | 9802121 | 2144363.5 | 99.51% | 359870/248204 |
| scored_source_skeleton | 64740 | 10000000 | 14672401 | 14302299 | 1.430230 | 15148 (23.40%) | 370102 | 9801948 | 2145344.9 | 99.28% | 359840/254551 |

## Differences versus compact
| variant | Δ active tokens | Δ trunc tokens | Δ candidate groups | Δ changed view active tokens | Δ view WWM token mass | Δ pair full-visible fraction |
|---|---:|---:|---:|---:|---:|---:|
| prefix_repeat | -30810 | -479 | 245 | -30853 | -30869 | 0.30% |
| content_spread | 864 | 164 | -73 | 915 | 899 | -0.04% |
| scored_source_skeleton | 7406 | 476 | -246 | 7262 | 7447 | -0.26% |

## Scientific reading
The current source wide skeleton recurrence integrated scaffold fixes legal words and row positions, but it does not automatically fix model exposure. Large differences in active tokens, WWM group mass, truncation, or pair visibility mean the arms should not be treated as a clean factorial training design. A repaired design must match these quantities under this loader before H100 use.

Changed-row CSV: `experiments/archive/frontier_consolidation/data/mlm_loader_exposure_audit_pools/changed_row_loader_exposure.csv`
Pair visibility CSV: `experiments/archive/frontier_consolidation/data/mlm_loader_exposure_audit_pools/pair_visibility_loader_exposure.csv`
JSON: `experiments/archive/frontier_consolidation/data/mlm_loader_exposure_audit_pools/mlm_loader_exposure_audit.json`
