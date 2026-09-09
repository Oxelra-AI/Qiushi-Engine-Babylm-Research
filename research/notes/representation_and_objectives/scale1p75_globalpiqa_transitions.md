# natural candidate contrast pilot GlobalPIQA row-transition analysis for scale1.75

Summary JSON: `experiments/archive/representation_and_objectives/data/globalpiqa_row_transition_analysis/globalpiqa_row_transition_summary.json`. Row CSVs are under `experiments/archive/representation_and_objectives/data/globalpiqa_row_transition_analysis`.

This analysis uses already-produced GlobalPIQA row readouts only; it does not train and is not a source of training examples.

## scale1p75_80M_disabled_to_scale1p75_80M_live__parallel
{
  "n_shared": 103,
  "transitions": {
    "W->W": 70,
    "C->C": 19,
    "C->W": 6,
    "W->C": 8
  },
  "net_correct_gain_b_minus_a_rows": 2,
  "same_choice_frac": 0.8155339805825242,
  "mean_margin_delta_b_minus_a": 0.021021745810440622,
  "median_margin_delta_b_minus_a": 0.0,
  "rank_delta_counts_b_minus_a": {
    "-3": 1,
    "-1": 13,
    "0": 76,
    "1": 12,
    "2": 1
  }
}

## legal16k_100M_to_scale1p75_100M__parallel
{
  "n_shared": 103,
  "transitions": {
    "W->W": 68,
    "C->C": 22,
    "C->W": 8,
    "W->C": 5
  },
  "net_correct_gain_b_minus_a_rows": -3,
  "same_choice_frac": 0.7572815533980582,
  "mean_margin_delta_b_minus_a": 0.07319026551382878,
  "median_margin_delta_b_minus_a": 0.0,
  "rank_delta_counts_b_minus_a": {
    "-2": 1,
    "-1": 17,
    "0": 65,
    "1": 15,
    "2": 4,
    "3": 1
  }
}

## scale1p75_80M_live_to_scale1p75_100M__parallel
{
  "n_shared": 103,
  "transitions": {
    "W->W": 73,
    "C->C": 24,
    "C->W": 3,
    "W->C": 3
  },
  "net_correct_gain_b_minus_a_rows": 0,
  "same_choice_frac": 0.9320388349514563,
  "mean_margin_delta_b_minus_a": 0.0009537453750642348,
  "median_margin_delta_b_minus_a": 0.0,
  "rank_delta_counts_b_minus_a": {
    "-1": 8,
    "0": 90,
    "1": 3,
    "2": 2
  }
}

## Research implication
If the 80M-to-100M comparison shows mostly W->W rank/margin reshuffling rather than many C->W losses, relational consolidation is weakly supported as a main repair. The stronger next route is to create a natural candidate-contrast or counterfactual credit signal that differs from ordinary MLM and is selected on held-out EWoK/GlobalPIQA movement before endpoint-scale training.
