# leadershape micro split alignment — leader-shape fork micro-split alignment

Evidence JSON: `experiments/archive/initial_model_studies/data/leadershape_micro_split_alignment.json`

## Exact checks

| check | value |
|---|---|
| example_order_manifest_bytes_equal | `True` |
| tokenization_summary_bytes_equal | `True` |
| hf_config_equal | `True` |
| word_exposure_equal | `True` |
| source_words_equal | `True` |
| masked_tokens_total_equal | `True` |
| orig_steps_equal_fork_steps | `True` |
| effective_batch_equal | `True` |
| micro_batch_size_recorded | `True` |

## Log summary

```json
{
  "orig_log_steps": 8,
  "fork_log_steps": 8,
  "all_cum_words_equal": true,
  "all_batch_words_equal": true,
  "all_masked_tokens_equal": true,
  "all_seq_len_equal": true,
  "max_lr_abs_diff": 0.0,
  "max_loss_abs_diff": 0.010149631500244283
}
```

## Tensor diff

- global max abs: `0.007206318899989128` in `deberta.encoder.layer.9.intermediate.dense.weight`
- global mean abs: `0.0006843279424116833`
- global RMS abs: `0.0011108194867469085`

## Interpretation
- The isolated fork now preserves the protected trainer accounting and masking semantics exactly at this short-run level: same examples, tokenizer summary, HF config, exposure, masked-token totals/per-step masked tokens, sequence length, optimizer steps, and LR trajectory.
- Final tensors/losses need not be bit-identical because the micro-split forward/backward changes dropout RNG grouping relative to one full-batch forward. This does not change the data, masks, loss reduction, optimizer-step schedule, or checkpoint accounting; it is an unavoidable memory-rescue difference unless dropout RNG is specially synchronized.
