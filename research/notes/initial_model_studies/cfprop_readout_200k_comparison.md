# cfprop readout 200k comparison — counterfactual propagation v3 readout localization at 200k

Evidence JSON: `experiments/archive/initial_model_studies/data/cfprop_readout_200k_comparison.json`

| mode | heldout pair | heldout random | train pair | tail20 pair | final CF loss | final MLM loss | aux pair words | HF clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dependent | 0.578125 | 0.625 | 0.4944044764903333 | 0.5195723682641983 | 1.0593039989471436 | 7.194316864013672 | 131448 | True |
| s2_nondependent | 0.609375 | 0.6444444643126593 | 0.515987210303283 | 0.5107730269432068 | 1.0403603315353394 | 7.195166110992432 | 131448 | True |
| s1_local | 0.65625 | 0.640625 | 0.5155875296901455 | 0.5392269730567932 | 1.0434010028839111 | 7.1974310874938965 | 131448 | True |

## Key deltas
```json
{
  "matched_wwm_exposure": true,
  "matched_aux_pair_words": true,
  "all_hf_clean": true,
  "dependent_minus_s2_nondep_heldout_pair_acc": -0.03125,
  "dependent_minus_s2_nondep_tail20_pair_acc": 0.008799341320991472,
  "dependent_minus_s2_nondep_final_cf_loss": 0.0189436674118042,
  "s1_local_minus_dependent_heldout_pair_acc": 0.078125,
  "s1_local_minus_dependent_tail20_pair_acc": 0.019654604792594954,
  "s1_local_minus_dependent_final_cf_loss": -0.015902996063232422
}
```

## Interpretation
- Dependent and s2 nondependent heldout pair accuracies remain similar; no convincing localization to dependent token.
