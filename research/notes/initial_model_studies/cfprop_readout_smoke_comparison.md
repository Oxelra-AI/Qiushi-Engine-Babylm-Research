# cfprop readout 200k comparison — counterfactual propagation readout-location smoke comparison

Evidence JSON: `experiments/archive/initial_model_studies/data/cfprop_readout_smoke_comparison.json`

| mode | heldout pair | heldout random | train pair | tail pair | final CF loss | aux pair words | HF clean |
|---|---:|---:|---:|---:|---:|---:|---:|
| dependent | 0.46875 | 0.578125 | 0.609375 | 0.609375 | 1.0120360851287842 | 3142 | True |
| s2_nondependent | 0.46875 | 0.5111111164093017 | 0.515625 | 0.515625 | 1.0242750644683838 | 3142 | True |
| s1_local | 0.640625 | 0.640625 | 0.515625 | 0.515625 | 1.0444645881652832 | 3142 | True |

## Key deltas
{
  "all_wwm_exposure_equal": true,
  "all_aux_pair_words_equal": true,
  "all_hf_clean": true,
  "dependent_minus_s2_nondep_heldout_pair_acc": 0.0,
  "dependent_minus_s2_nondep_heldout_random_acc": 0.06701388359069826,
  "dependent_minus_s1_local_heldout_pair_acc": -0.171875,
  "s1_local_minus_dependent_train_pair_acc": -0.09375,
  "s2_nondep_minus_dependent_tail_pair_acc": -0.09375,
  "s1_local_minus_dependent_tail_pair_acc": -0.09375,
  "dependent_minus_s2_nondep_cf_loss_last": -0.01223897933959961
}

## Interpretation
- 20k smoke does not localize heldout pair signal to the dependent token; dependent and s2-nondependent are similar at this scale.
- Readout smokes are matched in WWM exposure, auxiliary exposure, and HF cleanliness.
- s1-local readout is much easier, suggesting local anomaly is an upper-bound shortcut.
