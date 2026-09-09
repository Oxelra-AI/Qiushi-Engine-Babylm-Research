# topology phase1 result topology probe: disconn_delete bs+1

- condition: `shared_trunk`
- seed: `30000`
- bridge_sign: `1`
- detach_state_trunk: `False`
- training comparisons: 96 / original 192
- matching after equality pretrain eval: 1.000000
- final event trunk hash: `83f4c52f8c209254`
- final event_state_head hash: `0650a31d9de10b6c`
- final event_cmp_head hash: `b78068b694a9c107`

## Final train metrics

```json
{
  "epoch": 60,
  "loss": 1.5112233313629986e-06,
  "train_changed": 1.0,
  "train_cmp_modified": 1.0,
  "train_cmp_original_labels": 1.0,
  "train_state": 1.0,
  "train_unchanged": 1.0
}
```

## Central eval

```json
{
  "direct_same": 1.0,
  "graph_mean_de": 5.491301983594894,
  "graph_same": 0.5,
  "graph_same_margin": 2.5102782547473907,
  "hh_closure": 1.0,
  "hh_closure_margin": 9.45779367717944,
  "mixed_acc": 1.0,
  "mixed_acc_margin": 11.24773102509541,
  "pair_both_graph_same": 0.25,
  "same_init_changed": 0.75,
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
      "h0_dax",
      "h1_mep"
    ],
    [
      "h1_mep",
      "h2_norp"
    ]
  ],
  "target_rows_original": 96,
  "target_rows_retained": 0,
  "target_rows_with_original_label_after_modification": 0
}
```
