# raw name binding construction and design raw-name binding probe
Encoding: character-based names + <QRY> query (no <cand>/<other>)
Data filters: no_comparisons=False, no_bridge_changed_only=False
Fresh rename: True

## Per-run central readout

| condition | bridge_sign | seed | train_state | train_cmp | direct_same | graph_same | pair_both | unchanged | hh_closure | mixed_acc | mixed_margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tied | +1 | 29200 | 0.514 | 0.500 | 0.500 | 0.500 | 0.250 | 0.500 | 0.492 | 0.504 | 0.0 |

## Fresh rename readout

| condition | bridge_sign | seed | rename_direct_same | rename_graph_same | rename_pair_both | rename_hh_closure | rename_mixed_acc |
|---|---|---:|---:|---:|---:|---:|---:|
| tied | +1 | 29200 | 0.375 | 0.625 | 0.312 | 0.500 | 0.500 |
