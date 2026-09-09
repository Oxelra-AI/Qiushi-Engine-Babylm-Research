# natural candidate contrast pilot natural candidate-contrast pilot

Summary JSON: `experiments/archive/representation_and_objectives/data/natural_candidate_contrast_pilot/natural_candidate_contrast_summary.json`; row CSV: `experiments/archive/representation_and_objectives/data/natural_candidate_contrast_pilot/natural_candidate_rows.csv`; hard subset: `experiments/archive/representation_and_objectives/data/natural_candidate_contrast_pilot/natural_candidate_hard_subset.csv`.

This is a corpus readout/mining pilot only. It samples legal compact-view corpus tokens and never uses official evaluation text for training.

## Key counts
{
  "scale_full_errors_vs_alt": 17,
  "unsaturated_full_abs_margin_le_0p5": 1,
  "context_decides_local_alt_full_gold": 5,
  "scale_weaker_than_base_by_0p5": 14
}

## Margin summaries
{
  "scale_full_target_minus_alt": {
    "n": 31,
    "mean": 0.44076396188428324,
    "median": -0.5589541792869568,
    "p10": -3.757467269897461,
    "p90": 6.250340461730957,
    "min": -8.2003812789917,
    "max": 8.902383804321289
  },
  "scale_local_target_minus_alt": {
    "n": 31,
    "mean": -2.5609430420783257,
    "median": -2.879924774169922,
    "p10": -7.286560535430908,
    "p90": 2.4430999755859375,
    "min": -11.962902545928955,
    "max": 5.263389587402344
  },
  "context_gain_full_minus_local": {
    "n": 31,
    "mean": 3.001707003962609,
    "median": 2.4611597061157227,
    "p10": -0.50909423828125,
    "p90": 8.591475129127502,
    "min": -1.241856575012207,
    "max": 11.851521968841553
  },
  "scale_minus_base_full_margin": {
    "n": 31,
    "mean": -0.33010984620740336,
    "median": -0.3554849624633789,
    "p10": -2.3498687744140625,
    "p90": 1.137986421585083,
    "min": -2.8094005584716797,
    "max": 2.5936164259910583
  }
}

## Research implication
A short repair fork is scientifically worth building only if hard natural rows are abundant and differ from ordinary MLM: local alternatives must be plausible, full-context scale1.75 margins should be unsaturated or wrong, and matched-base comparison should show scale1.75-specific weakening. Otherwise this points away from generic candidate contrast and toward a stronger representation or counterfactual source.
