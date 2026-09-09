# leadershape fork alignment — leader-shape fork alignment

Evidence JSON: `experiments/archive/initial_model_studies/data/leadershape_fork_alignment.json`

## Exact accounting checks

| check | value |
|---|---|
| example_order_manifest_sha_equal | `True` |
| tokenization_summary_equal | `True` |
| config_equal | `True` |
| total_word_exposure_equal | `True` |
| source_words_equal | `True` |
| masked_tokens_total_equal | `False` |
| optimizer_steps_fork_equals_orig_steps | `True` |
| effective_batch_equals_orig_batch | `True` |

## Log summary

```json
{
  "orig_log_steps": 8,
  "fork_micro_steps": 16,
  "fork_aggregated_opt_steps": 8,
  "all_cum_words_equal": true,
  "all_batch_words_equal": true,
  "all_masked_tokens_equal": false,
  "max_lr_abs_diff": 0.0,
  "max_loss_abs_diff": 0.12852158389248736
}
```

## Tensor diff

- global max abs: `0.0075898319482803345` in `deberta.encoder.layer.11.output.dense.weight`
- global mean abs: `0.0012136529406739614`
- global RMS abs: `0.0017597360033690425`

## Interpretation
- Accounting alignment has differences; inspect exact_accounting/log_summary before using the fork for S1.
- Final tensors are not bit-identical (max_abs=0.00758983, mean_abs=0.00121365); this is expected under microbatch vs full-batch dropout/RNG unless all stochastic masks are synchronized, but accounting and optimizer-step equivalence are the load-bearing checks for the 10M S1 run.
