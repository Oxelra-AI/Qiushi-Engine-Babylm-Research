# research synthesis correction: Position-stratified binding gating

Position distribution (n=200 held pairs): {'unchanged_first': 160, 'updated_first': 29, 'ambiguous': 11}

If gating is driven by position shortcut, joint successes should concentrate
in unchanged_first pairs (where position is informative) and vanish in
updated_first pairs (where position would give the wrong assignment).

## Epoch 20 results

### answer_clean

| position | n | joint | gated% | a_correct | b_correct | both_wrong |
|---|---:|---:|---:|---:|---:|---:|
| unchanged_first | 160 | 38 | 23.8% | 71 | 127 | 0 |
| updated_first | 29 | 5 | 17.2% | 10 | 24 | 0 |
| ambiguous | 11 | 1 | 9.1% | 2 | 10 | 0 |

### uniform_wwm

| position | n | joint | gated% | a_correct | b_correct | both_wrong |
|---|---:|---:|---:|---:|---:|---:|
| unchanged_first | 160 | 6 | 3.8% | 54 | 111 | 1 |
| updated_first | 29 | 0 | 0.0% | 8 | 21 | 0 |
| ambiguous | 11 | 1 | 9.1% | 2 | 10 | 0 |

### answer_corrupt_update_state

| position | n | joint | gated% | a_correct | b_correct | both_wrong |
|---|---:|---:|---:|---:|---:|---:|
| unchanged_first | 160 | 24 | 15.0% | 96 | 88 | 0 |
| updated_first | 29 | 4 | 13.8% | 16 | 17 | 0 |
| ambiguous | 11 | 2 | 18.2% | 4 | 8 | 1 |

## answer_clean trajectory by position

| epoch | unch_first joint/n | upd_first joint/n | ambiguous joint/n |
|---:|---|---|---|
| 0 | 5/160 | 1/29 | 0/11 |
| 5 | 14/160 | 4/29 | 0/11 |
| 10 | 25/160 | 4/29 | 1/11 |
| 15 | 31/160 | 3/29 | 1/11 |
| 20 | 38/160 | 5/29 | 1/11 |

## Interpretation

If joint successes appear in BOTH position classes (unchanged_first AND
updated_first), position alone cannot explain the gating signal.
If joint successes are confined to unchanged_first, position is a viable shortcut.
