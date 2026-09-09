# family specific tokenizer predictor — Family-specific legal tokenizer predictor

This refines the global support-frontier predictor by using actual official-evaluation family statistics: family eval-token counts as a segmentation proxy and family `frac_lt50` as under-trained-token exposure. It is CPU-only and calibrates only on the measured legal16k/legal40k official endpoints.

- Anchor self-check passed: **True**
- Uncertainty baseline: legal40k two-seed population SD 0.3602; model-form uncertainty is half the disagreement between the sign-index and manual-dominant predictors.

## Candidate Overall predictions

| tokenizer | sign-index | manual-dominant | central | ± band | upper vs 41.8 |
|---|---:|---:|---:|---:|---:|
| legal_byte_bpe_40k_minfreq25 | 41.0236 | 41.0213 | 41.0224 | 0.3602 | -0.4173 |
| legal_byte_bpe_40k_minfreq50 | 41.0239 | 41.0150 | 41.0195 | 0.3603 | -0.4203 |
| legal_byte_bpe_24k | 41.0143 | 41.0093 | 41.0118 | 0.3602 | -0.4280 |
| legal_byte_bpe_32k | 40.8818 | 40.8803 | 40.8810 | 0.3602 | -0.5587 |

Best central candidate: **legal_byte_bpe_40k_minfreq25** at **41.0224**.
Pure support-floor crosses 41.8 within band: **False**.

## Route implication

The calibrated family-specific model does not support spending next full two-seed slot on a pure tokenizer support-floor interpolation alone. The evidence supports the current single-seed 12×384 depth test as a distinct factor, and if depth alone does not cross, the strongest next route should combine a support-floored tokenizer with another leader factor (masking curriculum or faithfully implemented sequence curriculum) rather than run minfreq25/minfreq50 as an isolated pair.

JSON: `experiments/archive/representation_and_objectives/data/family_specific_tokenizer_predictor/family_specific_tokenizer_predictor.json`
CSV: `experiments/archive/representation_and_objectives/data/family_specific_tokenizer_predictor/family_specific_predictions.csv`
