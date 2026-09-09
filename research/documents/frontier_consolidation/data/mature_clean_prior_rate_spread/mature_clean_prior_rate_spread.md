# earlier analysis mature-window clean-prior active-error spread

CPU-only forward MLM measurement using the clean DeBERTa MAX-geometry model at `chck_80M` and `chck_100M`. The script measures an ex-ante quantity on text the clean model did not train on for the intervention admitted blocks: `L_clean(80M) - L_clean(100M)`. No training, official benchmark scoring, upload, or leaderboard action is performed.

## Single-sample anchor from earlier analysis
- view_changed: 0.064341
- breadth_changed: 0.042806
- repeat_changed: 0.034349
- clean_displaced: 0.038125
- In the broader 40M→100M window, repeat exceeds breadth (`repeat=0.901819`, `breadth=0.430716`), so the mature terminal window is the mechanism-bearing coordinate here.

## Three-replicate mature rates
- view_changed: {'n': 3, 'mean': 0.05707, 'sd': 0.003073, 'min': 0.05356, 'max': 0.059276, 'values': [0.05356, 0.059276, 0.058373]}
- breadth_changed: {'n': 3, 'mean': 0.033848, 'sd': 0.002734, 'min': 0.031005, 'max': 0.036459, 'values': [0.036459, 0.031005, 0.034079]}
- repeat_changed: {'n': 3, 'mean': 0.037811, 'sd': 0.004865, 'min': 0.032201, 'max': 0.040873, 'values': [0.032201, 0.040358, 0.040873]}
- clean_displaced: {'n': 3, 'mean': 0.042389, 'sd': 0.001187, 'min': 0.04143, 'max': 0.043717, 'values': [0.04143, 0.043717, 0.042019]}

## In-corpus adult-prose mature-window commitment
- rate_80_to_100: 0.046387
- distinct_words: 442987
- mass_rate_product: 20548.838
- position_vs_reference: above clean-displaced mature drift and at/above breadth-scale clean-prior mature rate
- prediction: predict non-flat persistence for in-corpus adult prose; if downstream is flat, active error alone is insufficient

## Files
- summary_json: `experiments/archive/frontier_consolidation/data/mature_clean_prior_rate_spread/mature_clean_prior_rate_spread.json`
- measurements_csv: `experiments/archive/frontier_consolidation/data/mature_clean_prior_rate_spread/loss_measurements.csv`
- rates_csv: `experiments/archive/frontier_consolidation/data/mature_clean_prior_rate_spread/mature_rate_rows.csv`
- commitment_json: `experiments/archive/frontier_consolidation/data/mature_clean_prior_rate_spread/incorpus_mature_rate_commitment.json`
- summary_md: `research/documents/frontier_consolidation/data/mature_clean_prior_rate_spread/mature_clean_prior_rate_spread.md`
