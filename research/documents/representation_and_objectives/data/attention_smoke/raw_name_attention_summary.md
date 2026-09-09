# Step292b raw-name binding with query-attention

## Per-run readout

| condition | bs | train_state | train_cmp | direct_same | graph_same | pair_both | unchanged | hh_closure | mixed_acc | mixed_margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tied | +1 | 0.521 | 0.479 | 0.375 | 0.625 | 0.312 | 0.500 | 0.492 | 0.500 | -0.0 |
| shared_trunk | +1 | 0.514 | 0.469 | 0.375 | 0.625 | 0.312 | 0.500 | 0.539 | 0.525 | 0.0 |
| untied | +1 | 0.521 | 0.505 | 0.375 | 0.625 | 0.312 | 0.500 | 0.500 | 0.480 | 0.0 |
