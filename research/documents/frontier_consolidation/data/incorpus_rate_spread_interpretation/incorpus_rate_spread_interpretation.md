# register rate and spread interpretation in-corpus rate spread interpretation

This file-only readout places the future in-corpus admitted-block late rate against the register rate and spread interpretation resampled repeat/breadth/view rate scale. It intentionally does not rerun model inference.

## Status
- In-corpus post-training rate available: False
- Classification: pending_future_incorpus_60M_100M_rate
- Interpretation: The post-training in-corpus rate ladder remains pending; repeat the analysis when it is available.

## In-corpus rate
- Owner admitted-block loss by checkpoint: {}
- 60M→100M loss reduction: None

## Reference spread
{
  "repeat": {
    "n": 3,
    "mean": 0.15403,
    "sd": 0.007174,
    "min": 0.147678,
    "max": 0.161811,
    "values": [
      0.147678,
      0.161811,
      0.1526
    ]
  },
  "breadth": {
    "n": 3,
    "mean": 0.229361,
    "sd": 0.01041,
    "min": 0.220871,
    "max": 0.240976,
    "values": [
      0.220871,
      0.226236,
      0.240976
    ]
  },
  "view": {
    "n": 3,
    "mean": 0.303934,
    "sd": 0.019549,
    "min": 0.292553,
    "max": 0.326507,
    "values": [
      0.292553,
      0.326507,
      0.292741
    ]
  }
}

- Transition bands: {'repeat_breadth_midpoint': {'n': 3, 'mean': 0.191695, 'sd': 0.006574, 'min': 0.184275, 'max': 0.196788, 'values': [0.184275, 0.194024, 0.196788]}, 'repeat_view_midpoint': {'n': 3, 'mean': 0.228982, 'sd': 0.013206, 'min': 0.220116, 'max': 0.244159, 'values': [0.220116, 0.244159, 0.22267]}, 'repeat_range': {'min': 0.147678, 'max': 0.161811}, 'breadth_range': {'min': 0.220871, 'max': 0.240976}, 'view_range': {'min': 0.292553, 'max': 0.326507}}
- Rate→late exEntity5 fit spread: None

## Static profile comparator
{
  "primary_profile_predictions_mass_scaled": {
    "incorpus_minus_clean_exEntity4_noReading": 0.27543065398934624,
    "incorpus_minus_clean_exEntity5_withReading": 0.28927327530058433,
    "incorpus_minus_clean_cheap5_noReading": 0.3115962258155697,
    "incorpus_minus_full1x_exEntity4_noReading": 0.020123307236453336,
    "incorpus_minus_full1x_cheap5_noReading": 0.042285993744071604
  },
  "word_js_control_predictions_mass_scaled": {
    "incorpus_minus_clean_exEntity4_noReading": -0.024001987762539246,
    "incorpus_minus_full1x_exEntity4_noReading": -0.01742863003330492
  },
  "interpretation_rule": {
    "static_profile_proximity": "supported if in-corpus official ex-Entity movement is positive and closer to the mass-scaled profile prediction than to zero even if the late admitted-block loss rate is small",
    "out_of_corpus_novelty": "supported if in-corpus remains near clean while full1x/FineWeb stays positive despite similar or positive profile prediction for in-corpus",
    "rate_account": "supported if the official in-corpus movement follows the future 60M->100M admitted-block loss reduction rather than the static profile prediction alone"
  }
}

## Files
- summary_json: `experiments/archive/frontier_consolidation/data/incorpus_rate_spread_interpretation/incorpus_rate_spread_interpretation.json`
- summary_md: `research/documents/frontier_consolidation/data/incorpus_rate_spread_interpretation/incorpus_rate_spread_interpretation.md`
- spread_source: `experiments/archive/frontier_consolidation/data/register_removal_rate_and_reference_spread/register_removal_rate_and_reference_spread.json`
- future_rate_source: `experiments/archive/frontier_consolidation/data/incorpus_rate_ladder_after_training/incorpus_rate_prediction.json`
