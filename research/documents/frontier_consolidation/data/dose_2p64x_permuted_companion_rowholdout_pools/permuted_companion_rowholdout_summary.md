# permuted companion correspondence control MAX permuted-compact correspondence control

This CPU materialization keeps the MAX source rows and the exact compact-rewrite multiset, but reassigns every source to another source's compact view with no same-row or same-document donor.

## Core audit

- pair slots: 33,291; changed rows: 7,923; suffix rows reused from MAX view: 57,390
- exact 10M words: True; row-length sequence matches MAX view: True
- companion text multiset identical to MAX view: True; assignment is a permutation: True
- same-pair assignments: 0; same-doc assignments: 0; same-row assignments: 0; donor doc in target row docset: 0
- per-slot rewrite length changed only for 22 / 33291 slots; row companion sums preserved: True
- 100M training stream written: True; stream seed matches matched max dose execution note formula: True

## Token/WWM geometry versus MAX view

- legal16k token delta: 0 (+0.000000%)
- visible seq256 token delta: 1629 (+0.011399%)
- WWM group delta: 881 (+0.008973%)

## Expensive-work condition

Do not launch this arm merely because the file exists. Train it only if the already-running second-basin MAX view/repeat scoring reproduces the Entity carrier strongly enough that aligned-versus-permuted can decide whether record addressability, rather than compact-style filler, is the mechanism.

Metadata: `experiments/archive/frontier_consolidation/data/dose_2p64x_permuted_companion_rowholdout_pools/permuted_companion_rowholdout_metadata.json`
