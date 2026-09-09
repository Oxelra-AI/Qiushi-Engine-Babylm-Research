# frozen anchor fastpath disruption design fast-path score decision

Status: **PENDING**
Route read: `coherent_has_cheap7_edge_over_spanbreak;coherent_above_anchor_cheap7;coherent_above_shuffled86_cheap7;item_read=needs_scientific_interpretation;superglue_pending_or_not_run`

## Score table

| arm | status | cheap7 | Δcheap7 vs chck82 | SuperGLUE | Overall(AoA0) | ΔOverall vs chck82 |
|---|---|---:|---:|---:|---:|---:|
| chck82 | available | 43.95944987645173 | None | 69.7661813713118 | 41.942481167385985 | None |
| shuffled86 | cheap_and_superglue_available | 44.01285714285714 | 0.053407266405415044 | 69.81922238969935 | 41.98991359885548 | 0.047432431469495384 |
| coherent | cheap_and_superglue_available | 44.10642857142857 | 0.1469786949768448 | 69.77796826428681 | 42.058107584920755 | 0.1156264175347701 |
| spanbreak | cheap_available | 43.121428571428574 | -0.8380213050231546 | None | None | None |

## Sources

- chck82: cheap `experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json`, SuperGLUE `None`
- shuffled86: cheap `experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_summary/frozen82_tail4M_shuffled_summary.json`, SuperGLUE `experiments/archive/frontier_consolidation/data/repeat_shuffled_tail_superglue_summary/repeat_frozen82_tail4M_shuffled_superglue_superglue_summary.json`
- coherent: cheap `experiments/archive/frontier_consolidation/data/fastpath4M_coherent_summary/fastpath4M_coherent_summary.json`, SuperGLUE `experiments/archive/frontier_consolidation/data/fastpath4M_coherent_superglue_summary/fastpath4M_coherent_superglue_summary.json`
- spanbreak: cheap `experiments/archive/frontier_consolidation/data/fastpath4M_spanbreak_summary/fastpath4M_spanbreak_summary.json`, SuperGLUE `experiments/archive/frontier_consolidation/data/fastpath4M_spanbreak_superglue_summary/fastpath4M_spanbreak_superglue_summary.json`

The frozen-anchor fast-path route should continue only if coherent private-ON replay retains anchor-correct fragile relation/state decisions and adds new correct decisions reproducibly, not merely because it has private-OFF reversibility or a small aggregate score edge over a disrupted input.

JSON: `experiments/archive/frontier_consolidation/data/fastpath_score_decision/fastpath_score_decision.json`
