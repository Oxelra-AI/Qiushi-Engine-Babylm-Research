# distribution proximity prediction register prediction comparison

Data state: `prediction_only_no_register_scores_yet`; complete child-minus-adult checkpoints: 0.

This file binds future MAX-register scores to pre-score predictions. It is valid to run before scores exist; rows then contain predictions only.

## Aggregate comparison

| quantity | observed | profile original | profile strict eval-text | word control | obs-profile original | obs-profile strict | obs-word |
|---|---:|---:|---:|---:|---:|---:|---:|
| exEntity5 | NA | 0.7483 | 0.7697 | -0.0692 | NA | NA | NA |
| cheap6 | NA | 0.8175 | 0.8441 | -0.0854 | NA | NA | NA |

## Family comparison

| family | observed | profile original | profile strict eval-text | word control |
|---|---:|---:|---:|---:|
| BLiMP | NA | 0.8220 | 0.7664 | -0.0419 |
| Supplement | NA | 1.0998 | 1.0255 | -0.1336 |
| EWoK | NA | 0.7015 | 0.9312 | -0.0647 |
| COMPS | NA | 0.1291 | 0.2035 | -0.0263 |
| Reading | NA | 0.9889 | 0.9221 | -0.0796 |
| Entity | NA | 1.1636 | 1.2157 | -0.1661 |

## Reading

The main pre-score expectation is positive child_minus_adult on exEntity5, around +0.75 to +0.77 over chck_10M..80M. Positive means adult-prose removal was costlier. The profile expectation is strengthened if the family pattern is Supplement/Reading/BLiMP/EWoK positive with COMPS small. Near-zero or negative exEntity weakens the surface/register-profile account.

The strict task-aware extraction reproduces the original profile magnitude and should be reported beside it to avoid over-reliance on heuristic evaluation-text parsing.

The word-unigram control had poor calibration before scores arrived. Matching its tiny negative value would contradict the profile account but would not by itself validate lexical proximity.

Any resulting principle must retain distinct-content persistence: repeated MAX FineWeb recurrence loses broad ex-Entity value late while distinct V/B content persists.

JSON: `experiments/archive/frontier_consolidation/data/register_prediction_comparison/register_prediction_comparison_summary.json`
