# natural candidate contrast pilot relation-cue natural candidate-contrast pilot

Summary JSON: `experiments/archive/representation_and_objectives/data/relation_cue_candidate_contrast_pilot/relation_cue_candidate_contrast_summary.json`; rows: `experiments/archive/representation_and_objectives/data/relation_cue_candidate_contrast_pilot/relation_cue_candidate_rows.csv`; hard subset: `experiments/archive/representation_and_objectives/data/relation_cue_candidate_contrast_pilot/relation_cue_candidate_hard_subset.csv`.

This pilot filters the first natural candidate readout to relation/physical/comparative cue contexts and legal-corpus frequency-supported single-token words.

## Counts
{
  "scale_full_errors_vs_alt": 52,
  "unsaturated_full_abs_margin_le_0p5": 10,
  "context_decides_local_alt_full_gold": 66,
  "scale_weaker_than_base_by_0p5": 64
}

## Margins
{
  "scale_full_target_minus_alt": {
    "n": 160,
    "mean": 2.8001648541539907,
    "median": 2.195016622543335,
    "p10": -4.314408302307129,
    "p90": 10.460074424743652,
    "min": -8.70426082611084,
    "max": 17.110074996948242
  },
  "scale_local_target_minus_alt": {
    "n": 160,
    "mean": -2.958781836275011,
    "median": -3.4908249378204346,
    "p10": -7.847990036010742,
    "p90": 2.579059600830078,
    "min": -13.867896795272827,
    "max": 7.056475639343262
  },
  "context_gain_full_minus_local": {
    "n": 160,
    "mean": 5.758946690429002,
    "median": 3.2336554527282715,
    "p10": 0.09346771240234375,
    "p90": 15.5451078414917,
    "min": -2.013552665710449,
    "max": 26.574491918087006
  },
  "scale_minus_base_full_margin": {
    "n": 160,
    "mean": -0.12786445086821913,
    "median": -0.1381591558456421,
    "p10": -2.2632219791412354,
    "p90": 1.7780919075012207,
    "min": -6.208059787750244,
    "max": 5.558993101119995
  }
}

## Cue/source distribution
{
  "cue_counts": {
    "around": 8,
    "higher": 4,
    "more": 21,
    "first": 10,
    "under": 1,
    "inside": 2,
    "later": 2,
    "same": 6,
    "near": 3,
    "over": 14,
    "drop": 1,
    "into": 11,
    "left": 3,
    "close": 2,
    "turn": 3,
    "before": 8,
    "last": 3,
    "above": 6,
    "break": 2,
    "face": 3,
    "facing": 2,
    "below": 1,
    "holds": 2,
    "lower": 1,
    "after": 4,
    "less": 6,
    "previous": 1,
    "longer": 2,
    "right": 9,
    "different": 2,
    "through": 3,
    "between": 3,
    "outside": 1,
    "push": 1,
    "throw": 1,
    "front": 3,
    "next": 3,
    "onto": 1,
    "hold": 1
  },
  "source_counts": {
    "cleanqwen_fineweb_compact_view_reinvest": 96,
    "childes": 40,
    "open_subtitles": 8,
    "gutenberg": 6,
    "qwen_pair_packed": 9,
    "bnc_spoken": 1
  }
}

## Research implication
If this stricter relation-cue set still contains many scale-specific weak or wrong target-vs-alt margins, a short candidate-contrast fork is plausible. If rows are noisy or mostly target-token frequency artifacts, natural counterfactual/context transformation should be prioritized instead.
