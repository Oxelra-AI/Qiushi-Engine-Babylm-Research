# dualview pending panel interim dual-view auxiliary functional probe — full_adapter_on_r2

Created: `2026-08-31T22:32:38Z`

## Replay

```json
{
  "batches": 481,
  "cum_main_words": 19021584,
  "cum_aux_words": 966720,
  "cum_charged_words": 19988304,
  "aligned_aux_rows": 19339,
  "shuffled_aux_rows": 19339,
  "aligned_targets": 68432,
  "shuffled_targets": 68432,
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
  "stop_record": {
    "loader_step": 482,
    "event": "stop_before_cap",
    "current_charged": 19988304,
    "next_main_words": 39785,
    "next_aux_words": 1113
  }
}
```

## View NLLs

| model | source assignment | view | NLL | targets |
|---|---|---|---:|---:|
| aligned | aligned | cond | 6.435417 | 68432 |
| aligned | aligned | free | 6.465863 | 68432 |
| aligned | shuffled | cond | 6.504192 | 68432 |
| aligned | shuffled | free | 6.465863 | 68432 |
| shuffled | aligned | cond | 6.551705 | 68432 |
| shuffled | aligned | free | 6.555799 | 68432 |
| shuffled | shuffled | cond | 6.605849 | 68432 |
| shuffled | shuffled | free | 6.555799 | 68432 |
| mlm_only | aligned | cond | 6.268443 | 68432 |
| mlm_only | aligned | free | 6.509603 | 68432 |
| mlm_only | shuffled | cond | 6.525951 | 68432 |
| mlm_only | shuffled | free | 6.509603 | 68432 |

## Key deltas (negative means first model lower NLL)

```json
{
  "aligned_model_minus_shuffled_model_on_true_free_view": -0.08993636143104489,
  "aligned_model_minus_shuffled_model_on_true_conditioned_view": -0.11628732420390886,
  "aligned_model_minus_mlm_only_on_true_free_view": -0.0437399913474783,
  "aligned_model_minus_mlm_only_on_true_conditioned_view": 0.16697423514401155,
  "aligned_model_minus_shuffled_model_on_shuffled_free_view": -0.08993636143104489,
  "shuffled_model_minus_mlm_only_on_shuffled_free_view": 0.046196370083566585
}
```

JSON: `experiments/archive/frontier_consolidation/data/dualview_aux_function_probe/full_adapter_on_r2/full_adapter_on_r2.json`
