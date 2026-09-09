# compact reinvest adjbreak control compact-view reinvest adjacency-broken control

## Purpose

This CPU-only construction prepares the next mechanism comparison without starting a GPU run.  It should be used only if the running `compact_view_reinvest` full evaluation and seed43122 screen keep the endpoint scientifically alive.

The constructed control keeps the exact compact-view reinvest source-text multiset and compact-rewrite multiset, keeps the 10M pool word total and the changed-block row word sequence, and breaks the original source -> its own compact rewrite adjacency by assigning each source a rewrite from another source with the same rewrite word length, usually inside the same broad relation/topic stratum.

## Main matching facts

- slots/source-view units: 12155
- same-pair assignments left: 3
- same original row assignments: 3
- same primary domain assignments: 12135 (0.998355)
- same domain-signature assignments: 10348 (0.851337)
- exact rewrite-token-length assignments: 9585 (0.788564)
- rewrite token absolute-delta mean/p95/max: 0.6159 / 2.0000 / 39.0000
- true rewrite stays in the same row as its original source: 3

## Row and sequence-interface facts

- pool rows / words: 64740 / 10000000
- row word sequence identical to original reinvest view: True
- changed-row token-with-special exact matches / rows: 1354 / 3006
- changed-row token-with-special delta mean_abs/p95_abs/max_abs: 2.3580 / 13.0000 / 45.0000
- adjbreak seq256 full source+donor-rewrite visibility: 12072/12155 (0.993172)
- adjbreak source visibility: 12142/12155 (0.998930)
- rows over seq256 with special tokens: 82

## Files

- 10M control pool: `experiments/archive/representation_and_objectives/data/compact_reinvest_adjbreak_control/cleanqwen_fineweb_compact_view_reinvest_adjbreak_10M.jsonl`
- changed-block row metadata: `experiments/archive/representation_and_objectives/data/compact_reinvest_adjbreak_control/cleanqwen_fineweb_compact_view_reinvest_adjbreak_changed_block_rows_meta.jsonl`
- JSON measurement: `experiments/archive/representation_and_objectives/data/compact_reinvest_adjbreak_control/compact_reinvest_adjbreak_control_measurement.json`

The 100M training file was not written. If written later, it will use the same pass-wise row-index orders as the original reinvest arms. The next expensive action should wait for the endpoint results already running.
