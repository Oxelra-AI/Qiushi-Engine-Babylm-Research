# earlier analysis in-corpus adult-prose rate prediction

This file was written before inspecting any new earlier analysis decisive official score output. It is a CPU-only forward MLM loss readout and prediction commitment for the final in-corpus adult-prose cell.

## Measured in-corpus block
- Changed rows: 3005 / 65313 rows
- Changed words: 442987
- Changed sources: {'gutenberg': 272048, 'simple_wiki': 170939}
- Sampled changed rows: 300 ; sampled unchanged rows: 300

## Loss table

| model_arm | checkpoint | block | mean_loss | masked tokens |
|---|---:|---|---:|---:|
| clean | chck_60M | incorpus_admitted_changed | 3.7022 | 9230 |
| clean | chck_60M | incorpus_unchanged | 2.934421 | 9904 |
| clean | chck_60M | clean_displaced_same_positions | 2.838568 | 9782 |
| clean | chck_80M | incorpus_admitted_changed | 3.541737 | 9230 |
| clean | chck_80M | incorpus_unchanged | 2.796581 | 9904 |
| clean | chck_80M | clean_displaced_same_positions | 2.73526 | 9782 |
| clean | chck_100M | incorpus_admitted_changed | 3.49535 | 9230 |
| clean | chck_100M | incorpus_unchanged | 2.755179 | 9904 |
| clean | chck_100M | clean_displaced_same_positions | 2.699011 | 9782 |
| incorpus | chck_60M | incorpus_admitted_changed | 3.686713 | 9230 |
| incorpus | chck_60M | incorpus_unchanged | 2.974282 | 9904 |
| incorpus | chck_60M | clean_displaced_same_positions | 2.929267 | 9782 |
| incorpus | chck_80M | incorpus_admitted_changed | 3.467647 | 9230 |
| incorpus | chck_80M | incorpus_unchanged | 2.807021 | 9904 |
| incorpus | chck_80M | clean_displaced_same_positions | 2.812446 | 9782 |
| incorpus | chck_100M | incorpus_admitted_changed | 3.412794 | 9230 |
| incorpus | chck_100M | incorpus_unchanged | 2.769192 | 9904 |
| incorpus | chck_100M | clean_displaced_same_positions | 2.780432 | 9782 |

## Pre-score commitment

view_breadth_like_rate_prediction: in-corpus adult-prose rows remain an active late error source; predict a positive broad late ex-Entity gain comparable to the persistent view/breadth regime.

### Clean-prior position
- incorpus_clean_prior_100M: 3.49535
- reference_clean_prior_100M: {'view': 3.26586, 'repeat': 2.949448, 'breadth': 5.266488}
- nearest_reference_by_level: view
- distance_to_nearest: 0.22949
- distance_to_breadth_prior: 1.771138

### In-corpus owner rate
- loss_reduction_40_to_100: None
- loss_reduction_60_to_100: 0.273919
- loss_reduction_80_to_100: 0.054853
- loss_reduction_40_to_60: None
- loss_reduction_60_to_80: 0.219066
- loss_reduction_latest20M: None

### Existing MAX-arm rate anchors
- repeat: changed_loss_60M=2.097209, changed_loss_100M=1.948983, loss_reduction_60_to_100=0.148226, late_exEntity5=0.0343
- view: changed_loss_60M=2.846306, changed_loss_100M=2.550596, loss_reduction_60_to_100=0.29571, late_exEntity5=0.3853
- breadth: changed_loss_60M=4.950789, changed_loss_100M=4.728129, loss_reduction_60_to_100=0.22266, late_exEntity5=0.335

## Files
- JSON: `experiments/archive/frontier_consolidation/data/incorpus_rate_ladder_after_training/incorpus_rate_prediction.json`
- CSV: `experiments/archive/frontier_consolidation/data/incorpus_rate_ladder_after_training/incorpus_rate_losses.csv`
- Commitment JSON: `experiments/archive/frontier_consolidation/data/incorpus_rate_ladder_after_training/prediction_commitment.json`
