# topology phase1 result topology probe: detach_full_graph bs-1

- condition: `shared_trunk`
- seed: `30000`
- bridge_sign: `-1`
- detach_state_trunk: `True`
- training comparisons: 192 / original 192
- matching after equality pretrain eval: 1.000000
- final event trunk hash: `e9b18c26b6f63f8d`
- final event_state_head hash: `f5a61bfa06c9b398`
- final event_cmp_head hash: `fe25ea0e04835d13`

## Final train metrics

```json
{
  "epoch": 60,
  "loss": 0.4209097921848297,
  "train_changed": 0.9722222222222222,
  "train_cmp_modified": 0.4427083333333333,
  "train_cmp_original_labels": 0.4427083333333333,
  "train_state": 0.9861111111111112,
  "train_unchanged": 1.0
}
```

## Central eval

```json
{
  "direct_same": 0.25,
  "graph_mean_de": -0.038567617535591125,
  "graph_same": 0.5,
  "graph_same_margin": -0.0019130557775497437,
  "hh_closure": 0.46875,
  "hh_closure_margin": -3.7252902984619054e-09,
  "mixed_acc": 0.515625,
  "mixed_acc_margin": 3.7252902984619054e-09,
  "pair_both_graph_same": 0.25,
  "same_init_changed": 0.375,
  "unchanged": 0.5
}
```

## Modified comparison provenance

```json
{
  "action": "unchanged",
  "changed_labels_total": 0,
  "intervention": "detach_full_graph",
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

## Detach gradient check

```json
{
  "checked_query_key": "aligned_sparse_state_bridge|train_aligned_sparse_state_bridge_h0_dax_0000_v0_st0_ibsame|changed",
  "event_state_head_max_abs_grad": 0.0002982988953590393,
  "event_trunk_max_abs_grad_from_detached_state_loss": 0.0,
  "event_trunk_param_tensors": 15,
  "event_trunk_params_with_grad": 0,
  "loss": 0.6931648254394531
}
```
