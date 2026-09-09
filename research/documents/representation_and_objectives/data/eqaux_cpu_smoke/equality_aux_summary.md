# raw span identity routing result equality-auxiliary span-routing probe

Protected equality means matcher CharGRU + neither bias were pretrained on token/query equality labels and frozen before relational training. This is a controlled routing intervention, not natural span discovery.

| condition | bs | vocab | eq_pre | frozen | train_state | train_cmp | graph_same | unchanged | hh_closure | mixed | match_cand | match_other | eq both gate before | after pre | after rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shared_trunk | 1 | 60 | 1 | True | 0.517 | 0.521 | 0.375 | 0.500 | 0.570 | 0.549 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
