# research exact swap reading and pending step075 interpretation — innovation intervention comparison while earlier analysis 80M runs

CPU-only analysis; it did not inspect, wait for, or affect the managed earlier analysis training/evaluation tasks.

## Why this comparison matters

The research report delivered a different innovation-biased WWM design after the earlier analysis strict probability run had already been launched. The two mechanisms are scientifically related but not the same experiment, so the pending 80M score must not be overinterpreted.

## Launched earlier analysis probability reallocation

- Targets: 17,968 strict row-unique content-like innovation groups / 28,007 tokens per 10M; 131,675 copyable groups / 205,941 tokens.
- Probabilities: strict 0.500; copyable 0.102401659; source/filler/other ordinary at 0.15 in expectation.
- Per 10M expected change: strict token labels +9802.5, copyable token labels -9802.5; strict group labels +6288.8, copyable group labels -6267.5.
- Exposure multipliers: strict ×3.333; copyable ×0.683; extra strict group labels are 0.428% of global baseline selected groups.
- CPU smoke: strict rate 0.4984991423670669, copyable 0.10016390999850991, source 0.14844919465900244, selected-token mass ratio 1.0001655464688939, checks passed `True`.

## research exact-swap alternative

- Targets: 26,315 pair-local source-absent innovation groups / 36,899 tokens per 10M, including 22,815 content/other and 3,500 relation-cue groups.
- Mechanism: begin with exact baseline WWM; at most one innovation proposal per changed row; if not already selected, swap with an already-selected ordinary non-pair donor of identical token length, preferably from an ordinary row.
- Broad smoke projection per 10M: baseline innovation group labels 3947.2, extra 2545.8, treatment total 6493.0, exposure multiplier ×1.645.
- Exact controls in smoke: selected groups/tokens unchanged per batch; copyable delta 0; source-selection changed rows 0; ordinary-row donor fraction 1.0.
- Frozen batch audit: all 2529 actual batches contain changed rows but none are all-changed; changed rows per batch mean 11.882, range 2–24.

## Scientific interpretation

- Broad BLiMP/Supplement/EWoK gains at 70M/80M versus spatial repair route status would support continuing earlier analysis to 100M/full evaluation.
- Innovation-probe improvement with broad score damage or copyable-like damage would suggest the target is real but copyable suppression/pressure size is harmful; then exact-swap becomes the next lower-perturbation test.
- No innovation-probe improvement and weak cheap columns would weaken the whole masking-pressure route, not only the probability schedule.
- A gain confined to GlobalPIQA_nonparallel or COMPS repeats the redistribution pattern that closed word-mean and minfreq50 and should stop earlier analysis without 100M continuation.

The research result does not make the already-running earlier analysis experiment obsolete: earlier analysis directly tests whether copyable rewrite supervision is surplus enough to fund stronger strict-content innovation pressure. But earlier analysis and exact-swap are not interchangeable. If earlier analysis is negative in a way consistent with copyable suppression or excessive pressure, exact-swap remains a distinct lower-perturbation candidate; if earlier analysis is broadly positive, it is the stronger and already-running route toward 100M/full evaluation.

Full JSON: `experiments/archive/frontier_consolidation/data/innovation_intervention_comparison/innovation_intervention_comparison.json`
