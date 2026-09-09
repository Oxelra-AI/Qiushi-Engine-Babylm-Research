# register rate and spread interpretation register-removal rate and reference-rate spread

This CPU-only forward MLM loss measurement was made before inspecting any new childspeech official scorer result. It freezes how the active-error/rate account predicts the MAX-register contrast and quantifies how much the repeat/view/breadth rate scale moves under row/mask resampling.

## Register removal-side rate commitment
- Contrast: `childspeech_removed - adultprose_removed`
- Positive contrast means: the arm removing child/speech rows scores higher, so removing adult prose was costlier
- Child/speech removed block 60M→100M clean loss reduction: {'n': 3, 'mean': 0.150145, 'sd': 0.003279, 'min': 0.147004, 'max': 0.153546, 'values': [0.149885, 0.153546, 0.147004]}
- Adult-prose removed block 60M→100M clean loss reduction: {'n': 3, 'mean': 0.212764, 'sd': 0.008398, 'min': 0.203253, 'max': 0.219157, 'values': [0.215882, 0.219157, 0.203253]}
- Adult-minus-child removed-block rate: {'n': 3, 'mean': 0.062619, 'sd': 0.00552, 'min': 0.056249, 'max': 0.065997, 'values': [0.065997, 0.065611, 0.056249]}
- Rate sign: positive_register_contrast_predicted_by_rate: adult-prose block has larger clean late loss reduction, so removing adult prose should be costlier and childspeech_removed - adultprose_removed should be positive

## Static-profile comparison frozen earlier
- profile_pred_exEntity5: 0.748
- profile_pred_cheap6: 0.817
- strict_profile_pred_exEntity5: 0.7697
- strict_profile_pred_cheap6: 0.8441
- word_js_pred_exEntity5: -0.068
- word_js_pred_cheap6: -0.085

## Reference rate spread
- Historical single-sample rates: {'repeat': 0.148226, 'breadth': 0.22266, 'view': 0.29571}
- Resampled rates: {'repeat': {'n': 3, 'mean': 0.15403, 'sd': 0.007174, 'min': 0.147678, 'max': 0.161811, 'values': [0.147678, 0.161811, 0.1526]}, 'breadth': {'n': 3, 'mean': 0.229361, 'sd': 0.01041, 'min': 0.220871, 'max': 0.240976, 'values': [0.220871, 0.226236, 0.240976]}, 'view': {'n': 3, 'mean': 0.303934, 'sd': 0.019549, 'min': 0.292553, 'max': 0.326507, 'values': [0.292553, 0.326507, 0.292741]}}
- Repeat/breadth midpoint spread: {'n': 3, 'mean': 0.191695, 'sd': 0.006574, 'min': 0.184275, 'max': 0.196788, 'values': [0.184275, 0.194024, 0.196788]}
- Repeat/view midpoint spread: {'n': 3, 'mean': 0.228982, 'sd': 0.013206, 'min': 0.220116, 'max': 0.244159, 'values': [0.220116, 0.244159, 0.22267]}

## Row-block audit
- Register selected rows: child/speech 6992 rows / 1118720 words; adult prose 6992 rows / 1118720 words.
- Child/speech sources: {'childes': 578080, 'open_subtitles': 411040, 'bnc_spoken': 124160, 'switchboard': 5440}
- Adult-prose sources: {'gutenberg': 710560, 'simple_wiki': 408160}

## Files
- summary_json: `experiments/archive/frontier_consolidation/data/register_removal_rate_and_reference_spread/register_removal_rate_and_reference_spread.json`
- measurements_csv: `experiments/archive/frontier_consolidation/data/register_removal_rate_and_reference_spread/loss_measurements.csv`
- rates_csv: `experiments/archive/frontier_consolidation/data/register_removal_rate_and_reference_spread/rate_rows.csv`
- commitment_json: `experiments/archive/frontier_consolidation/data/register_removal_rate_and_reference_spread/prediction_commitment.json`
- summary_md: `research/documents/frontier_consolidation/data/register_removal_rate_and_reference_spread/register_removal_rate_and_reference_spread.md`
