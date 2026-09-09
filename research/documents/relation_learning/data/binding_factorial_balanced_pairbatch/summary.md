# earlier analysis balanced paired binding factorial

This rerun drops UPDATED_USE single rows and trains only complete UNCHANGED_DISTRACTOR_USE A/B pairs, with both halves of each pair in the same optimizer update.

## Epoch 20 all-pair gating

| arm | joint | A source-retain | B update-select | expected joint | gating count | gating frac | mean A margin | mean B margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| answer_clean | 44/200 | 83/200 | 161/200 | 66.81 | -22.81 | -0.114 | -0.307 | 1.480 |
| uniform_wwm | 7/200 | 64/200 | 142/200 | 45.44 | -38.44 | -0.192 | -0.583 | 0.706 |
| answer_corrupt_update_state | 30/200 | 116/200 | 113/200 | 65.54 | -35.54 | -0.178 | 0.438 | 0.428 |

## Epoch 20 updated-entity-in-source subset

| arm | joint | A source-retain | B update-select | expected joint | gating count | gating frac | mean A margin | mean B margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| answer_clean | 24/119 | 49/119 | 94/119 | 38.71 | -14.71 | -0.124 | -0.271 | 1.397 |
| uniform_wwm | 2/119 | 38/119 | 83/119 | 26.50 | -24.50 | -0.206 | -0.507 | 0.615 |
| answer_corrupt_update_state | 13/119 | 71/119 | 60/119 | 35.80 | -22.80 | -0.192 | 0.488 | 0.301 |
