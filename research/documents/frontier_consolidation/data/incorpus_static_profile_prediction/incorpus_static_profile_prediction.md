# earlier analysis in-corpus static profile prediction

This CPU-only prediction was made without reading new official scorer outputs. It quantifies the static adult-prose profile/proximity account for the in-corpus final cell and keeps word-JS as a failed lexical control.

## Blocks

- incorpus_adultprose: changed_rows=3005, admitted_words=442987, removed_words=442987, mass_ratio_to_MAX=0.3960, admitted_sources={'gutenberg': 272048, 'simple_wiki': 170939}, removed_sources={'bnc_spoken': 51115, 'open_subtitles': 159331, 'childes': 231058, 'switchboard': 1483}
- full1x_view: changed_rows=3006, admitted_words=423692, removed_words=423692, mass_ratio_to_MAX=0.3788, admitted_sources={'compact_view_dose2p64x_matched_rowholdout': 423559, 'gutenberg': 133}, removed_sources={'gutenberg': 95801, 'simple_wiki': 59596, 'bnc_spoken': 32147, 'open_subtitles': 95009, 'childes': 140456, 'switchboard': 683}

## Aggregate predictions

| arm/model | metric | quantity | raw Δ points | mass-scaled Δ points |
|---|---|---|---:|---:|
| incorpus_adultprose | word_js | exEntity4_noReading | -0.0606 | -0.0240 |
| incorpus_adultprose | word_js | cheap5_noReading | -0.0770 | -0.0305 |
| incorpus_adultprose | profile_js | exEntity4_noReading | +0.6955 | +0.2754 |
| incorpus_adultprose | profile_js | cheap5_noReading | +0.7868 | +0.3116 |
| full1x_view | word_js | exEntity4_noReading | -0.0174 | -0.0066 |
| full1x_view | word_js | cheap5_noReading | -0.0275 | -0.0104 |
| full1x_view | profile_js | exEntity4_noReading | +0.6740 | +0.2553 |
| full1x_view | profile_js | cheap5_noReading | +0.7110 | +0.2693 |
| incorpus_minus_full1x_static_prediction | word_js | exEntity4_noReading | -0.0433 | -0.0174 |
| incorpus_minus_full1x_static_prediction | word_js | cheap5_noReading | -0.0495 | -0.0201 |
| incorpus_minus_full1x_static_prediction | profile_js | exEntity4_noReading | +0.0215 | +0.0201 |
| incorpus_minus_full1x_static_prediction | profile_js | cheap5_noReading | +0.0758 | +0.0423 |

## Frozen commitment

{
  "status": "INCORPUS_STATIC_PROFILE_PRE_SCORE_COMMITMENT",
  "created_utc": "2026-09-04T22:35:58Z",
  "made_without_reading_step294_official_scores": true,
  "model": "distribution proximity prediction strict-eval-text beta applied to in-corpus and full1x changed/removed text blocks; word-JS retained as failed lexical control; mass-scaled values reflect ~0.38-0.40x MAX word substitution.",
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
- summary_json: `experiments/archive/frontier_consolidation/data/incorpus_static_profile_prediction/incorpus_static_profile_prediction.json`
- summary_md: `research/documents/frontier_consolidation/data/incorpus_static_profile_prediction/incorpus_static_profile_prediction.md`
- commitment_json: `experiments/archive/frontier_consolidation/data/incorpus_static_profile_prediction/prediction_commitment.json`
- distance_rows_csv: `experiments/archive/frontier_consolidation/data/incorpus_static_profile_prediction/distance_rows.csv`
- prediction_rows_csv: `experiments/archive/frontier_consolidation/data/incorpus_static_profile_prediction/prediction_rows.csv`
- aggregate_prediction_rows_csv: `experiments/archive/frontier_consolidation/data/incorpus_static_profile_prediction/aggregate_prediction_rows.csv`
