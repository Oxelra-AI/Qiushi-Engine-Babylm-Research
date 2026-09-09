# topology phase1 result topology probe: detach_full_graph bs+1

- condition: `shared_trunk`
- seed: `30000`
- bridge_sign: `1`
- detach_state_trunk: `True`
- training comparisons: 192 / original 192
- matching after equality pretrain eval: 0.250000
- final event trunk hash: `5b0563362e6a7aeb`
- final event_state_head hash: `f69881b55e85c6ac`
- final event_cmp_head hash: `fd94e72b7350d35f`

## Final train metrics

```json
{
  "epoch": 1,
  "loss": 0.6931202411651611,
  "train_changed": 0.6111111111111112,
  "train_cmp_modified": 0.5625,
  "train_cmp_original_labels": 0.5625,
  "train_state": 0.5625,
  "train_unchanged": 0.5138888888888888
}
```

## Central eval

```json
{
  "direct_same": 0.21875,
  "graph_mean_de": -0.0001681518042460084,
  "graph_same": 0.59375,
  "graph_same_margin": 3.2779411412775517e-05,
  "hh_closure": 0.5078125,
  "hh_closure_margin": 1.0244548320770174e-08,
  "mixed_acc": 0.4921875,
  "mixed_acc_margin": -6.984919309616072e-10,
  "pair_both_graph_same": 0.296875,
  "same_init_changed": 0.40625,
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
      "h1_mep",
      "h2_norp"
    ],
    [
      "h0_dax",
      "h1_mep"
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
