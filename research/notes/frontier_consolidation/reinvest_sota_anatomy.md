# compact repeat causal read plan compact-view-reinvest SOTA anatomy

CPU-only reading of the peer-evaluated compact_view_reinvest full official-compatible endpoint. This note does not launch training or evaluation; it parses existing reports/predictions and verification records.

## Score status
- compact_view_reinvest Overall 42.086786; NLP average 52.934439; Human-like average 4.120000; AoA status official_aoa_done.
- Scores: BLiMP 66.870, Supplement 63.280, EWoK 53.670, Entity 27.750, COMPS 51.970, SuperGLUE 71.381, GlobalPIQA 35.620, Reading 8.240, AoA 0.000
- Delta vs clean-Qwen: BLiMP +0.030, Supplement +0.440, EWoK +3.480, Entity +1.990, COMPS +0.190, SuperGLUE +1.072, GlobalPIQA -1.000, Reading +0.480, AoA +0.000, Overall +0.742
- Delta vs visible 41.8 leader: BLiMP -0.330, Supplement +7.270, EWoK -2.400, Entity -0.700, COMPS -1.600, SuperGLUE +1.591, GlobalPIQA -4.050, Reading +2.820, AoA +0.000, Overall +0.287
- Delta vs compact_view_core: BLiMP -0.280, Supplement +1.430, EWoK +2.410, Entity -0.100, COMPS -0.210, SuperGLUE +2.480, GlobalPIQA +0.485, Reading -0.010, AoA +12.687, Overall +2.099

## Verification signals read from - train 100M hash matches train command: True; exact 10M words: True; required AoA checkpoints present: True; missing columns: []; unexpected missing/empty artifacts: {}.
- refined overlap audit changed block vs heldout official rows: score-bearing overlap rows delta -12, GLUE-valid rows delta -14, unique 7-gram delta -103. This is a safeguard result, not proof that all overlap is harmless.

## Supplement anatomy
- Official equal-UID Supplement: clean 62.84, reinvest 63.28, delta +0.44.
| UID | clean | reinvest | delta |
|---|---:|---:|---:|
| qa_congruence_easy | 70.31 | 71.88 | +1.56 |
| qa_congruence_tricky | 49.09 | 46.06 | -3.03 |
| turn_taking | 64.64 | 64.29 | -0.36 |
| subject_aux_inversion | 80.76 | 83.68 | +2.92 |
| hypernym | 49.41 | 50.48 | +1.07 |

## GlobalPIQA anatomy
- GlobalPIQA_parallel: clean 25.24, reinvest 25.24, delta +0.00; lost 5, gained 5, n=103.
- GlobalPIQA_nonparallel: clean 48.00, reinvest 46.00, delta -2.00; lost 14, gained 12, n=100.

## SuperGLUE anatomy
| task | clean | reinvest | delta |
|---|---:|---:|---:|
| mrpc | 86.275 | 83.824 | -2.451 |
| rte | 69.784 | 67.626 | -2.158 |
| boolq | 68.379 | 67.339 | -1.040 |
| multirc | 68.276 | 68.647 | +0.371 |
| mnli | 60.065 | 60.575 | +0.509 |
| qqp | 77.843 | 78.580 | +0.737 |
| wsc | 61.538 | 73.077 | +11.538 |

## Scientific reading
- This is a real endpoint update, not merely a fast-screen projection: all nine official-like columns are present, AoA is submit-ready under the local official-compatible helper, and the recomputed arithmetic exceeds both clean-Qwen and the visible 41.8 leader.
- Reinvestment changes the meaning of compact density: the failed neutral compact-core did not show aggregate NLP gain, but reinvest turns compact savings into added source exposure and recovers SuperGLUE/AoA while preserving large EWoK and Entity gains. The principle is better stated as redundancy-reduced semantic second views plus reinvested source diversity under a fixed word budget, not compacting alone.
- Remaining weakness is GlobalPIQA/practical commonsense relative to both the visible leader and the target of a durable SOTA. Because the endpoint margin over 41.8 is only +0.2868, seed43122 and independent verification matter before packaging or trying to improve. Do not start a new 100M route until the pending seed/AoA/compact-repeat evidence is read; use this endpoint as the new reference.

Machine-readable JSON: `experiments/archive/frontier_consolidation/data/reinvest_sota_anatomy/reinvest_sota_anatomy.json`
