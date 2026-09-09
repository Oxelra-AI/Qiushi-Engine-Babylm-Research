# posalign repair and equality emergence posalign occurrence audit

This checks whether raw event strings contain multiple occurrences of a candidate/other name. posalign matcher design and capacity hard assignment canonicalizes one argmax token per query; raw span identity routing result hard oracle canonicalizes every exact occurrence.

| subset | n events | both single frac | any repeated frac | max occurrence | hist max-occ |
|---|---:|---:|---:|---:|---|
| train_state | 288 | 1.000 | 0.000 | 1 | {1: 288} |
| train_cmp | 384 | 1.000 | 0.000 | 1 | {1: 384} |
| eval_state | 384 | 1.000 | 0.000 | 1 | {1: 384} |
| eval_cmp | 1280 | 1.000 | 0.000 | 1 | {1: 1280} |

## First repeated examples

