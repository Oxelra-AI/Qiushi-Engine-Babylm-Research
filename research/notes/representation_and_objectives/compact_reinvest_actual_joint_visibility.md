# compact core joint visibility audit compact_view_reinvest actual joint visibility

JSON: `experiments/archive/representation_and_objectives/data/compact_reinvest_actual_joint_visibility/compact_reinvest_actual_joint_visibility.json`

## Actual packed-row reconstruction

- Pair rows/non-pair rows in the changed block: 3005 / 1 across 3006 changed rows.
- Exact string reconstruction from selected pair order: 3005 rows; word-count reconstruction: 3005 rows.

## Seq256 joint visibility

- Total selected pair occurrences: 12155 unique 12155.
- Source visible pairs: 12137 (0.998519); full source+rewrite visible pairs: 12061 (0.992267).
- Partially visible pairs: 91 (0.007487); fully hidden pairs: 3 (0.000247).
- Rows over seq256 with special tokens: 91; rows where at least one pair loses full visibility: 91.

## Loss pattern

- Pair-position counts for non-full-visible units: {3: 46, 4: 29, 2: 17, 1: 1, 5: 1}.
- Non-full-visible units occur at the tail of packed rows if the position counts concentrate on later pair positions; this makes the first several adjacent source+rewrite signals robustly visible.

## Scientific reading

The compact reinvest intervention is not just a nominal word-level construction: almost every selected source+compact rewrite unit is physically available within the model's seq256 input. This strengthens the interpretation of the running downstream tests as tests of anchor-preserving compression and source reinvestment rather than a truncation accident.
