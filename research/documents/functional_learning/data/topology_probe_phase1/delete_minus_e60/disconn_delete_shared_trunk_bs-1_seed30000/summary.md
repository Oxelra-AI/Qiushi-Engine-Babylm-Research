# topology phase1 result topology probe: disconn_delete bs-1

- condition: `shared_trunk`
- seed: `30000`
- bridge_sign: `-1`
- detach_state_trunk: `False`
- training comparisons: 96 / original 192
- matching after equality pretrain eval: 1.000000
- final event trunk hash: `1eb0c3662c47f92a`
- final event_state_head hash: `ddbb4db41ca5bdb1`
- final event_cmp_head hash: `1651e62d99cf5a1c`

## Final train metrics

```json
{
  "epoch": 60,
  "loss": 4.039547093270812e-06,
  "train_changed": 1.0,
  "train_cmp_modified": 1.0,
  "train_cmp_original_labels": 0.75,
  "train_state": 1.0,
  "train_unchanged": 1.0
}
```

## Central eval

```json
{
  "direct_same": 0.0,
  "graph_mean_de": -3.4310853481292725,
  "graph_same": 0.25,
  "graph_same_margin": -9.396676540374756,
  "hh_closure": 0.75,
  "hh_closure_margin": 5.2112899595439055,
  "mixed_acc": 0.125,
  "mixed_acc_margin": -5.620856642064076,
  "pair_both_graph_same": 0.125,
  "same_init_changed": 0.125,
  "unchanged": 0.5
}
```

## Modified comparison provenance

```json
{
  "action": "deleted_h1_incident_edges",
  "changed_labels_total": 0,
  "intervention": "disconn_delete",
  "modified_edge_counts": {
    "h0_dax->h2_norp": 48,
    "h2_norp->h3_ziv": 48
  },
  "modified_label_balance": {
    "False": 48,
    "True": 48
  },
  "modified_total": 96,
  "original_edge_counts": {
    "h0_dax->h1_mep": 48,
    "h0_dax->h2_norp": 48,
    "h1_mep->h2_norp": 48,
    "h2_norp->h3_ziv": 48
  },
  "original_label_balance": {
    "False": 96,
    "True": 96
  },
  "original_total": 192,
  "shuffle_seed": null,
  "target_edges_unordered": [
    [
      "h1_mep",
      "h2_norp"
    ],
    [
      "h0_dax",
      "h1_mep"
    ]
  ],
  "target_rows_original": 96,
  "target_rows_retained": 0,
  "target_rows_with_original_label_after_modification": 0
}
```
