# packed targetselect static load and readout patch v2 packed static label-load profile

JSON: `experiments/archive/representation_and_objectives/data/packed_static_label_load_profile_v2/packed_static_label_load_profile_v2.json`
Batch table: `experiments/archive/representation_and_objectives/data/packed_static_label_load_profile_v2/packed_static_label_load_batches.jsonl`

This was computed without reading the running 100M arms.

## Main fields

```json
{
  "unweighted_mean_expected_selected_minus_abs_deleted_bpe_per_batch": 0.165480427,
  "lr_after_weighted_mean_expected_selected_minus_abs_deleted_bpe_per_batch": 0.1687432351,
  "lr_used_weighted_mean_expected_selected_minus_abs_deleted_bpe_per_batch": 0.1686973652,
  "sum_lr_after_x_expected_selected_minus_abs_deleted_bpe": 0.213375820847,
  "sum_lr_used_x_expected_selected_minus_abs_deleted_bpe": 0.213317818237,
  "lr_after_weighted_denom_ratio_abs_over_copied": 0.999980365906,
  "lr_used_weighted_denom_ratio_abs_over_copied": 0.999980370977,
  "max_abs_batch_denom_ratio_deviation_from_1": 0.0018208433
}
```

The primary arm comparison still waits for the trained endpoints; this file only narrows a loss-normalization/timing confound.
