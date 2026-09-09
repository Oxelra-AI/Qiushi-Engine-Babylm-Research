# earlier analysis binding factorial gating analysis
The coordinate reported here is `joint_correct - a_correct*b_correct/n`. Positive values mean pairwise entity assignment succeeds more often than expected from the two marginal half-accuracies; negative values mean the two halves are anticorrelated or governed by a packet-level source/update preference.
## Dataset balance
- Train rows: 4998; answer kinds {'source_state': 1666, 'new_state': 3332}; pair halves {'A': 1666, 'B': 1666, 'single': 1666}.
- Heldout rows: 600; answer kinds {'source_state': 200, 'new_state': 400}; pair halves {'A': 200, 'B': 200, 'single': 200}.
- The 4,998 training rows therefore contain 1,666 source-state answer rows and 3,332 new-state answer rows; answer-only training admits a 2:1 update-state prior before entity identity is read.
## Epoch 20: all 200 binding pairs
| arm | joint | A source-retain | B update-select | expected joint | gating count | gating frac | mean A margin | mean B margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| answer_clean | 41/200 | 74/200 | 167/200 | 61.79 | -20.79 | -0.104 | -0.625 | 1.981 |
| uniform_wwm | 7/200 | 59/200 | 147/200 | 43.37 | -36.37 | -0.182 | -0.657 | 0.761 |
| answer_corrupt_update_state | 31/200 | 122/200 | 109/200 | 66.49 | -35.49 | -0.177 | 0.542 | 0.368 |
## Epoch 20: updated entity already present in source (119 pairs)
| arm | joint | unchanged-source retention | updated-state selection | expected joint | gating count | gating frac | mean A margin | mean B margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| answer_clean | 22/119 | 42/119 | 99/119 | 34.94 | -12.94 | -0.109 | -0.578 | 1.943 |
| uniform_wwm | 3/119 | 37/119 | 85/119 | 26.43 | -23.43 | -0.197 | -0.573 | 0.683 |
| answer_corrupt_update_state | 15/119 | 75/119 | 59/119 | 37.18 | -22.18 | -0.186 | 0.618 | 0.207 |
## Trajectory, all pairs
| arm | epoch | joint | A | B | expected | gating | mean A margin | mean B margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| answer_clean | 0 | 6 | 78 | 128 | 49.92 | -43.92 | -0.594 | 0.678 |
| answer_clean | 5 | 21 | 68 | 152 | 51.68 | -30.68 | -0.791 | 1.243 |
| answer_clean | 10 | 31 | 70 | 161 | 56.35 | -25.35 | -0.748 | 1.528 |
| answer_clean | 15 | 44 | 79 | 165 | 65.17 | -21.17 | -0.610 | 1.760 |
| answer_clean | 20 | 41 | 74 | 167 | 61.79 | -20.79 | -0.625 | 1.981 |
| uniform_wwm | 0 | 6 | 78 | 128 | 49.92 | -43.92 | -0.594 | 0.678 |
| uniform_wwm | 5 | 8 | 65 | 143 | 46.48 | -38.48 | -0.650 | 0.759 |
| uniform_wwm | 10 | 7 | 61 | 145 | 44.23 | -37.23 | -0.663 | 0.776 |
| uniform_wwm | 15 | 8 | 60 | 148 | 44.40 | -36.40 | -0.659 | 0.771 |
| uniform_wwm | 20 | 7 | 59 | 147 | 43.37 | -36.37 | -0.657 | 0.761 |
| answer_corrupt_update_state | 0 | 6 | 78 | 128 | 49.92 | -43.92 | -0.594 | 0.678 |
| answer_corrupt_update_state | 5 | 17 | 94 | 122 | 57.34 | -40.34 | -0.186 | 0.482 |
| answer_corrupt_update_state | 10 | 24 | 108 | 115 | 62.10 | -38.10 | 0.097 | 0.439 |
| answer_corrupt_update_state | 15 | 25 | 116 | 108 | 62.64 | -37.64 | 0.328 | 0.401 |
| answer_corrupt_update_state | 20 | 31 | 122 | 109 | 66.49 | -35.49 | 0.542 | 0.368 |
## Interpretation
- `answer_clean` raises raw joint over uniform WWM, but its epoch-20 gating coordinate remains negative on all pairs and on the 119-pair within-source subset. The main movement is stronger B/update selection with weak or declining A/source retention, not a clean entity gate.
- `answer_corrupt_update_state` improves A/source retention while suppressing B/update selection, showing the rows are sensitive to source-vs-update support. Its raw joint is still far below the independence product, so it also does not learn pairwise assignment.
- `uniform_wwm` barely moves from the base and remains strongly negative on the gating coordinate.
- The unbalanced row mix is a construction cause: source-state answer rows are outnumbered 2:1 by new-state answer rows because UPDATED single rows were stacked on top of the paired rows. A balanced paired rerun should drop or mirror singles and keep both halves of each pair in the same update.
