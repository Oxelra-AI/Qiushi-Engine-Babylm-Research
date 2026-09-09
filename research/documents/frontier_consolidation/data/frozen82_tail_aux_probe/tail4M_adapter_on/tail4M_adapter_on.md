# frozen82 short tail plan frozen-82M tail auxiliary functional probe — tail4M_adapter_on

Created: `2026-09-01T00:51:03Z`

## Replay

```json
{
  "batches": 100,
  "cum_main_words": 3954265,
  "cum_aux_words": 38564,
  "cum_charged_words": 3992829,
  "aligned_aux_rows": 806,
  "shuffled_aux_rows": 806,
  "aligned_targets": 2708,
  "shuffled_targets": 2708,
  "aux_summary": {
    "status": "SPARSE_AUX_PAIR_DATA",
    "n_pair_rows": 1765,
    "n_ok": 3005,
    "n_fail": 1,
    "n_used_pairs": 2431,
    "shuffle_seed": 43022,
    "shuffle_singleton_groups_flagged": 0,
    "shuffle_diff_doc": 12151,
    "shuffle_same_doc": 4,
    "shuffle_missing_fallback_selfsource": 0,
    "tokenizer_sha": "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9",
    "pool_sha": "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23",
    "seq_length": 256,
    "elapsed_sec": 13.55,
    "source_pair_data": "experiments/archive/frontier_consolidation/data/aux_pair_data/aux_pair_data.json",
    "selection": "top_0.2_by_changed_source_absent_piece_per_charge",
    "top_fraction_pairs": 0.2,
    "selected_aux_charge_per_full_activation": 109051,
    "selected_changed_source_absent_pieces": 12369,
    "selected_alnum_pieces": 41929,
    "created_utc": "2026-08-31T22:06:10Z"
  },
  "stop_record": {
    "loader_step": 101,
    "event": "stop_before_cap",
    "current_charged": 3992829,
    "next_main_words": 39969,
    "next_aux_words": 162
  }
}
```

## View NLLs

| model | source assignment | view | NLL | targets |
|---|---|---|---:|---:|
| aligned | aligned | cond | 2.633332 | 2708 |
| aligned | aligned | free | 5.646571 | 2708 |
| aligned | shuffled | cond | 5.752682 | 2708 |
| aligned | shuffled | free | 5.646571 | 2708 |
| shuffled | aligned | cond | 2.782353 | 2708 |
| shuffled | aligned | free | 5.628545 | 2708 |
| shuffled | shuffled | cond | 5.628665 | 2708 |
| shuffled | shuffled | free | 5.628545 | 2708 |
| base | aligned | cond | 3.254242 | 2708 |
| base | aligned | free | 6.415290 | 2708 |
| base | shuffled | cond | 6.268773 | 2708 |
| base | shuffled | free | 6.415290 | 2708 |

## Key deltas (negative means first model lower NLL)

```json
{
  "aligned_model_minus_shuffled_model_on_true_free_view": 0.018026061741252875,
  "aligned_model_minus_shuffled_model_on_true_conditioned_view": -0.14902118248988838,
  "aligned_model_minus_base_on_true_free_view": -0.7687184490164416,
  "aligned_model_minus_base_on_true_conditioned_view": -0.6209096274664696,
  "shuffled_model_minus_base_on_shuffled_free_view": -0.7867445107576945,
  "aligned_model_minus_shuffled_model_on_shuffled_free_view": 0.018026061741252875
}
```

JSON: `experiments/archive/frontier_consolidation/data/frozen82_tail_aux_probe/tail4M_adapter_on/tail4M_adapter_on.json`
