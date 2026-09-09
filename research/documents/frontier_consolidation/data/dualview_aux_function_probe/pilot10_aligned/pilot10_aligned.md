# dualview pending panel interim dual-view auxiliary functional probe — pilot10_aligned

Created: `2026-08-31T21:59:52Z`

## Replay

```json
{
  "batches": 10,
  "cum_main_words": 395038,
  "cum_aux_words": 21364,
  "cum_charged_words": 416402,
  "aligned_aux_rows": 422,
  "shuffled_aux_rows": 422,
  "aligned_targets": 1503,
  "shuffled_targets": 1503,
  "aux_summary": {
    "status": "AUX_PAIR_DATA",
    "n_pair_rows": 3006,
    "n_ok": 3005,
    "n_fail": 1,
    "n_used_pairs": 12155,
    "shuffle_seed": 43022,
    "shuffle_singleton_groups_flagged": 0,
    "shuffle_diff_doc": 12151,
    "shuffle_same_doc": 4,
    "shuffle_missing_fallback_selfsource": 0,
    "tokenizer_sha": "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9",
    "pool_sha": "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23",
    "seq_length": 256,
    "elapsed_sec": 13.55
  },
  "stop_record": null
}
```

## View NLLs

| model | source assignment | view | NLL | targets |
|---|---|---|---:|---:|
| aligned | aligned | cond | 6.481114 | 1503 |
| aligned | aligned | free | 6.513626 | 1503 |
| aligned | shuffled | cond | 6.557970 | 1503 |
| aligned | shuffled | free | 6.513626 | 1503 |

## Key deltas (negative means first model lower NLL)

```json
{
  "aligned_model_minus_shuffled_model_on_true_free_view": null,
  "aligned_model_minus_shuffled_model_on_true_conditioned_view": null,
  "aligned_model_minus_mlm_only_on_true_free_view": null,
  "aligned_model_minus_mlm_only_on_true_conditioned_view": null,
  "aligned_model_minus_shuffled_model_on_shuffled_free_view": null,
  "shuffled_model_minus_mlm_only_on_shuffled_free_view": null
}
```

JSON: `experiments/archive/frontier_consolidation/data/dualview_aux_function_probe/pilot10_aligned/pilot10_aligned.json`
