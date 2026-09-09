# topology phase1 result topology probe: full_graph bs+1

- condition: `shared_trunk`
- seed: `30000`
- bridge_sign: `1`
- detach_state_trunk: `False`
- training comparisons: 192 / original 192
- matching after equality pretrain eval: 1.000000
- final event trunk hash: `fd456bd2bb204ab5`
- final event_state_head hash: `6fbaa327466b8586`
- final event_cmp_head hash: `7e88136692540770`

## Final train metrics

```json
{
  "epoch": 60,
  "loss": 7.317776180570945e-05,
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
  "graph_mean_de": -0.3399950861930847,
  "graph_same": 1.0,
  "graph_same_margin": 13.924900949001312,
  "hh_closure": 1.0,
  "hh_closure_margin": 10.360086577475528,
  "mixed_acc": 1.0,
  "mixed_acc_margin": 10.73875416877429,
  "pair_both_graph_same": 0.5,
  "same_init_changed": 1.0,
  "unchanged": 0.5
}
```

## Modified comparison provenance

```json
{
  "action": "unchanged",
  "changed_labels_total": 0,
  "intervention": "full_graph",
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
  "target_rows_with_original_label_after_modification": 96
}
```
