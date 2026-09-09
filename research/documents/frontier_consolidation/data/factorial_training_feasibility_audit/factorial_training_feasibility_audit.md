# lead route assessment after source use probe factorial training feasibility audit

CPU-only audit.  It does not authorize H100 training; it constrains whether a future exact-loader-matched factorial screen is possible after the DeBERTa trajectory and evidence are read.

## Variant aggregate target mass under exact DeBERTa loader
| variant | pairs | targets | active | WWM mass | whole-word copy | complete-BPE copy | contentlike | tail active | absent active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 12155 | 161,708 | 244,738 | 247,333 | 83.49% | 70.21% | 65.82% | 77,568 | 37,625 |
| compact_scrambled | 12155 | 161,708 | 244,739 | 247,334 | 83.49% | 70.20% | 65.82% | 77,645 | 37,633 |
| prefix_fluent | 12155 | 161,708 | 214,823 | 216,465 | 99.79% | 99.74% | 50.79% | 0 | 348 |
| prefix_scrambled | 12155 | 161,708 | 214,822 | 216,464 | 99.79% | 99.74% | 50.79% | 0 | 348 |
| sourcewide_onegap | 12155 | 161,708 | 234,687 | 236,850 | 99.83% | 99.68% | 58.64% | 100,272 | 300 |
| sourcewide_onegap_scrambled | 12155 | 161,708 | 234,688 | 236,851 | 99.83% | 99.68% | 58.64% | 100,356 | 300 |
| best_contiguous_span | 12155 | 161,708 | 224,082 | 226,071 | 99.76% | 99.67% | 55.74% | 89,905 | 409 |

## Clean single-factor controls already supported
- `compact_minus_compact_scrambled`: active delta -1, WWM delta -1, complete-BPE-copy delta 0.000072 mean fraction. This remains the cleanest ordered-vs-scrambled contrast within a fixed lexical multiset.
- `sourcewide_onegap_minus_sourcewide_onegap_scrambled`: active delta -1, WWM delta -1, complete-BPE-copy delta 0.000037 mean fraction. This remains the cleanest ordered-vs-scrambled contrast within a fixed lexical multiset.
- `prefix_fluent_minus_prefix_scrambled`: active delta 1, WWM delta 1, complete-BPE-copy delta 0.000003 mean fraction. This remains the cleanest ordered-vs-scrambled contrast within a fixed lexical multiset.

## Cross-family feasibility: compact vs sourcewide_onegap
All 12,155 pairs: compact-sourcewide active delta 10,051, WWM delta 10,483, tail-active delta -22,704, absent-active delta 37,325, complete-BPE-copy mean delta -0.3067.

Pair-level sign structure:
- active-token delta positive fraction 0.513; quantiles {'min': -17.0, 'p05': -3.0, 'p25': 0.0, 'median': 1.0, 'p75': 2.0, 'p95': 5.0, 'max': 19.0}
- tail-active delta positive fraction 0.131; quantiles {'min': -25.0, 'p05': -7.0, 'p25': -3.0, 'median': -1.0, 'p75': 0.0, 'p95': 2.0, 'max': 14.0}

Heuristic subset selection shows whether exposure can be numerically balanced by reducing pair dose.  This does **not** fix lexical-copy composition; it only tells whether active/WWM/tail mass can be made comparable.
| contrast | subset | n pairs | Δ active | Δ WWM | Δ tail active | Δ absent active | Δ complete-BPE-copy mean |
|---|---|---:|---:|---:|---:|---:|---:|
| compact_minus_sourcewide_onegap | all_pairs | 12155 | 10,051 | 10,483 | -22,704 | 37,325 | -0.3067 |
| compact_minus_sourcewide_onegap | heuristic_n9000 | 9000 | 7,236 | 7,551 | -16,753 | 27,826 | -0.3068 |
| compact_minus_sourcewide_onegap | heuristic_n6000 | 6000 | 4,727 | 4,950 | -11,130 | 18,460 | -0.3072 |
| compact_minus_sourcewide_onegap | heuristic_n3000 | 3000 | 2,213 | 2,303 | -5,328 | 9,034 | -0.3091 |
| compact_minus_sourcewide_onegap | heuristic_n1500 | 1500 | 989 | 1,038 | -2,671 | 4,411 | -0.3102 |
| sourcewide_onegap_minus_prefix_fluent | all_pairs | 12155 | 19,864 | 20,385 | 100,272 | -48 | -0.0007 |
| sourcewide_onegap_minus_prefix_fluent | heuristic_n9000 | 9000 | 14,505 | 14,887 | 73,866 | -48 | -0.0003 |
| sourcewide_onegap_minus_prefix_fluent | heuristic_n6000 | 6000 | 9,564 | 9,807 | 49,026 | -37 | -0.0007 |
| sourcewide_onegap_minus_prefix_fluent | heuristic_n3000 | 3000 | 4,568 | 4,691 | 24,120 | -19 | -0.0007 |
| sourcewide_onegap_minus_prefix_fluent | heuristic_n1500 | 1500 | 2,245 | 2,304 | 12,125 | -11 | -0.0007 |

## Scientific reading
- The fixed-word-multiset ordered/scrambled arms are mechanically clean and are the only immediately credible exact-loader training contrast from the present candidates.
- Compact vs sourcewide_onegap can be partially balanced on active/WWM/tail exposure by pair subset selection, but it cannot remove the central lexical-copy/semantic-recoding difference: compact contains many absent/non-source words and much lower complete-BPE copy fraction. Training such a contrast would test a coupled natural-compact-vs-extractive-sourcewide object, not a pure tail-coverage or pure semantic-transformation factor.
- Therefore the next possible H100 factorial screen, if pending trajectory/evidence leaves this route strongest, should be a minimal ordered-vs-scrambled contrast at fixed compact lexical multiset (compact vs compact_scrambled), optionally paired with sourcewide_onegap vs sourcewide_onegap_scrambled as a second wave. It should not be represented as proving tail coverage or budget efficiency by itself.

JSON: `experiments/archive/frontier_consolidation/data/factorial_training_feasibility_audit/factorial_training_feasibility_audit.json`
Pairwise CSVs: `experiments/archive/frontier_consolidation/data/factorial_training_feasibility_audit`
