# lamb numeric comparison — local LAMB numerical comparison

JSON: `experiments/archive/initial_model_studies/data/lamb_numeric_comparison.json`

## Result

- local_reference_style_debias_false_vs_torch_optimizer_default: final max abs parameter delta = `0`
- local_reference_style_debias_true_vs_torch_optimizer_debias_true: final max abs parameter delta = `0`

## Interpretation

The public leader metadata does not expose training code or exact LAMB dependency. The staged `torch_optimizer` implementation (citing cybertronai/pytorch-lamb) is used as the recognized reference. Its default has `debias=False`; original Algorithm 2 in the LAMB paper uses bias-corrected moments. The isolated trainer now exposes this as an explicit `--lamb_debias` switch rather than silently measuring a custom variant. Near-zero deltas here are required before LAMB smokes or 10M screens.
