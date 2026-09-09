# earlier analysis exact-swap CPU audit

Audited words: 1,999,862; rows: 12,953; batches: 51
Changed-row exposures: 610; proposals: 610; successful swaps: 511

## Selection mass
- Selected group delta: 0
- Selected token delta: 0
- Batch mass mismatches: 0
- Successful swaps / baseline selected groups: 0.001740

## Targeting and controls
- Innovation groups selected baseline→biased: 804 → 1,315 (multiplier 1.6356)
- Projected successful swaps per 10M words: 2555.18
- Collision rate: 0.1623; no-donor rate: 0.000000
- Copyable group delta: 0; source group delta: 0
- Donor protected pair/source/copyable counts: 0 / 0 / 0
- Donor ordinary-row count: 511; changed-nonpair donor count: 0

## Interpretation
- Exact-swap port preserves selected group and token mass exactly across audited real frozen-stream batches.
- Copyable rewrite and protected source selections are unchanged in aggregate; donors are ordinary non-pair groups.
- The intervention is much smaller than earlier analysis probability reallocation: it swaps a small fraction of global selected groups while forcing one candidate per eligible changed-row exposure when possible.
- No equal-length donor misses occurred in the audited real batch stream.

All checks passed: `True`
Event sample: `experiments/archive/frontier_consolidation/data/exact_swap_cpu_audit/exact_swap_events_sample.jsonl`
Full JSON: `experiments/archive/frontier_consolidation/data/exact_swap_cpu_audit/exact_swap_cpu_audit.json`
