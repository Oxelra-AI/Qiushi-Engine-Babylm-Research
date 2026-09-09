# earlier analysis geometry-arm support profile

Created: `2026-09-08T07:27:22Z`

## Full 80-update support totals

| quantity | value |
|---|---:|
| qwen_rows | 3831 |
| ordinary_full_targets | 104186 |
| dense_masked_tokens | 176607 |
| dense_sparse_label_targets | 28590 |
| dense_nonlabel_targets | 148017 |
| common_targets | 22171 |
| ordinary_full_examples | 3831 |
| dense_nonlabel_full_examples | 3831 |
| ordinary_common_examples | 3769 |

## Ratios

| ratio | value |
|---|---:|
| dense_nonlabel_over_ordinary_full_targets | 1.420700 |
| common_over_ordinary_full_targets | 0.212802 |
| common_over_dense_nonlabel_targets | 0.149787 |
| dense_sparse_label_over_dense_masked_tokens | 0.161885 |
| dense_nonlabel_over_dense_masked_tokens | 0.838115 |

## Macro summaries

{
  "ordinary_full_targets_by_macro": {
    "n": 80,
    "sum": 104186,
    "mean": 1302.325,
    "median": 1289.0,
    "min": 891,
    "max": 1787
  },
  "dense_nonlabel_targets_by_macro": {
    "n": 80,
    "sum": 148017,
    "mean": 1850.2125,
    "median": 1836.5,
    "min": 1254,
    "max": 2542
  },
  "common_targets_by_macro": {
    "n": 80,
    "sum": 22171,
    "mean": 277.1375,
    "median": 270.5,
    "min": 183,
    "max": 420
  },
  "qwen_rows_by_macro": {
    "n": 80,
    "sum": 3831,
    "mean": 47.8875,
    "median": 47.0,
    "min": 33,
    "max": 66
  }
}

The proposed dense-corrupted non-label KL changes both prediction-state rendering and target support relative to clean ordinary-WWM KL if run as a full arm. Common-support gradient measurements are needed to understand rendering geometry, while these totals describe the resource/support side of the intervention.
