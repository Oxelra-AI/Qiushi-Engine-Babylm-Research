# xspan 1m available coordinate manifest — XSpan 1M mechanism comparison

Evidence JSON: `experiments/archive/initial_model_studies/data/xspan_1m_mechanism_comparison.json`

| arm | rho | context | counted words | WWM loss last | XSpan loss last | true loss | wrong loss | no loss | Δ true-wrong | Δ true-no |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| xspan_true | 0.15 | true_s1 | 1018928 | 6.1516 | 8.2673 | 8.1065 | 8.1099 | 8.1468 | +0.0034 | +0.0403 |
| xspan_wrong | 0.15 | wrong_s1 | 1018897 | 6.2680 | 8.3478 | 8.2279 | 8.2284 | 8.2383 | +0.0005 | +0.0104 |
| wwm_only | 0.0 | true_s1 | 1000000 | 6.0933 | 0.0000 | 8.8194 | 8.8209 | 8.8223 | +0.0015 | +0.0029 |

## Deltas
```json
{
  "matched_wwm_exposure": true,
  "matched_official_steps": true,
  "all_hf_clean": true,
  "true_minus_wrong_delta_logprob_true_minus_wrong": 0.0028712052166959268,
  "true_minus_wwm_delta_logprob_true_minus_wrong": 0.0018949075851253383,
  "true_minus_wrong_delta_logprob_true_minus_no": 0.029920127430656862,
  "true_minus_wwm_delta_logprob_true_minus_no": 0.03741384939781156,
  "true_minus_wwm_true_s1_loss": -0.7129538394019423,
  "wrong_minus_wwm_true_s1_loss": -0.5915043041445287,
  "true_minus_wwm_final_wwm_loss": 0.05828380584716797,
  "wrong_minus_wwm_final_wwm_loss": 0.17471599578857422
}
```

## Interpretation
- At 1M, true-s1 XSpan does not induce a meaningful heldout true-vs-wrong specificity over the wrong-s1 control; the gain is only a few thousandths of logprob/token.
- Both XSpan arms greatly improve absolute XSpan target likelihood versus WWM-only, showing the primary span objective is learned, but this can be target-span familiarity rather than s1 binding.
- XSpan rho=0.15 slightly worsens ordinary WWM loss at 1M, so guard columns may be vulnerable.
