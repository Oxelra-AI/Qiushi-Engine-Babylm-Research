# research synthesis correction: research synthesis correction and integration note

## Critical correction: research binding conclusion is superseded

The research synthesis report at `analysis/Research_Report.md` uses the 
independence-excess coordinate (joint - A·B/n) for the balanced binding factorial 
and concludes it is a "terminal negative decision." 

**This conclusion is WRONG** and is superseded by the corrected paired-null analysis 
at `data/corrected_binding_analysis/`.

The corrected analysis shows:
- The no-gating null for paired rows sharing identical context is **joint = 0**, 
  not A·B/n (which assumes independence between anti-correlated halves)
- answer_clean achieves 44/200 = 22% gating (vs 3% base, 3.5% WWM)
- both_wrong = 0: every joint success is a genuine flip
- Still rising at epoch 20 (6→18→30→35→44)
- Position audit confirms gating in both position classes (23.8% and 17.2%)

The binding route is NOT stopped. It is a partial positive that warrants the 
chck_82M practical candidate (training in progress at this stage).

## Useful research synthesis material

The following parts of the research synthesis remain valuable:

1. **Cross-study narrative** (`six_session_research_state.md`): Good overview of the 
   progression from INITIAL_MODEL_STUDIES→COMPACT_EXPERIENCE→REPRESENTATION_FRONTIER_STUDIES→FUNCTIONAL_RELATION_STUDIES, endpoint hierarchy, and fixed-budget principle

2. **Evidence matrix** (`evidence_matrix.md`): 31-row matrix with scope boundaries. 
   Binding rows need correction per above.

3. **Dose/substitution accounting**: Correctly describes dose as a bundled substitution 
   (duplicate originals + new rewrites + local colocation + displaced text). The dose 
   knee characterization at seed43022 is accurate.

4. **Practical route map** (`practical_route_map.md`): Route A (dose) decision tree is 
   sound — need seed43122 replication before any coherent replay. Route B (binding) 
   stop rule needs correction: the balanced factorial is a partial positive, not a 
   terminal negative.

5. **Pending results**: Measurements still in progress remain separate from completed evidence.

## Two-seed cheap7 comparison

| column | seed43022 base/d21/d25 | seed43122 base/d21/d25 |
|---|---|---|
| BLiMP | 68.62/67.61/66.00 | 67.07/67.22/67.16 |
| Entity | 27.46/27.12/26.83 | 25.77/27.45/27.99 |
| EWoK | 49.08/50.14/52.52 | 52.27/52.08/51.35 |
| Reading | 8.32/7.87/7.30 | 8.31/7.77/8.39 |
| Supplement | 63.88/63.94/62.14 | 61.48/58.13/62.26 |
| COMPS | 52.85/53.31/50.90 | 51.56/52.23/51.37 |
| GP | 42.58/41.52/39.04 | 40.54/36.12/39.14 |
| cheap7 | 43.54/43.92/42.98 | 43.86/43.00/43.95 |

Seed43022 monotone BLiMP-down/EWoK-up pattern does NOT replicate at seed43122.
Entity rises at both seeds. Cheap7 is dominated by noisy GlobalPIQA.
No reliable two-seed direction exists in the scalar metrics.

The dose substitution profile is real but seed-dependent in magnitude and direction.
