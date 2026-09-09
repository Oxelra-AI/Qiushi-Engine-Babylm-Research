# topology phase1 result topology probe: disconn_adversarial bs-1

- condition: `shared_trunk`
- seed: `30000`
- bridge_sign: `-1`
- detach_state_trunk: `False`
- training comparisons: 192 / original 192
- matching after equality pretrain eval: 1.000000
- final event trunk hash: `c6703725e742cd8b`
- final event_state_head hash: `d6349a2b85a93eb4`
- final event_cmp_head hash: `b10aa8db85680187`

## Final train metrics

```json
{
  "epoch": 60,
  "loss": 2.8020338504575193e-05,
  "train_changed": 1.0,
  "train_cmp_modified": 1.0,
  "train_cmp_original_labels": 0.5,
  "train_state": 1.0,
  "train_unchanged": 1.0
}
```

## Central eval

```json
{
  "direct_same": 0.0,
  "graph_mean_de": -1.6241649091243744,
  "graph_same": 0.5,
  "graph_same_margin": 0.8649598658084869,
  "hh_closure": 0.5,
  "hh_closure_margin": -0.48561792348200983,
  "mixed_acc": 0.25,
  "mixed_acc_margin": -4.962612966365078,
  "pair_both_graph_same": 0.25,
  "same_init_changed": 0.25,
  "unchanged": 0.5
}
```

## Modified comparison provenance

```json
{
  "action": "inverted_h1_incident_edge_labels",
  "changed_labels_total": 96,
  "intervention": "disconn_adversarial",
  "modified_edge_counts": {
    "h0_dax->h1_mep": 48,
    "h0_dax->h2_norp": 48,
    "h1_mep->h2_norp": 48,
    "h2_norp->h3_ziv": 48
  },
  "modified_label_balance": {
    "False": 96,
    "True": 96
  },
  "modified_total": 192,
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
  "target_rows_retained": 96,
  "target_rows_with_original_label_after_modification": 0
}
```
