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
| incorpus | chck_10M | incorpus_admitted_changed | 5.432695 | 9230 |
| incorpus | chck_10M | incorpus_unchanged | 4.676025 | 9904 |
| incorpus | chck_10M | clean_displaced_same_positions | 4.105253 | 9782 |
| incorpus | chck_20M | incorpus_admitted_changed | 4.716829 | 9230 |
| incorpus | chck_20M | incorpus_unchanged | 4.047264 | 9904 |
| incorpus | chck_20M | clean_displaced_same_positions | 3.573964 | 9782 |

## Pre-score commitment

conditional_rate_commitment: official in-corpus broad late gain should be small/repeat-like if the same-script in-corpus admitted-block loss reduction from 60M to 100M is at or below the repeat-distinct midpoint; it should approach the view/breadth late-retention regime only if that 60M->100M reduction remains distinct-like. The current clean-prior level is recorded now and must not be used alone as the mechanism.

### Clean-prior position
- incorpus_clean_prior_100M: 3.49535
- reference_clean_prior_100M: {'view': 3.26586, 'repeat': 2.949448, 'breadth': 5.266488}
- nearest_reference_by_level: view
- distance_to_nearest: 0.22949
- distance_to_breadth_prior: 1.771138

### In-corpus owner rate
- loss_reduction_40_to_100: None
- loss_reduction_60_to_100: None
- loss_reduction_80_to_100: None
- loss_reduction_40_to_60: None
- loss_reduction_60_to_80: None
- loss_reduction_latest20M: None

### Existing MAX-arm rate anchors
- repeat: changed_loss_60M=2.097209, changed_loss_100M=1.948983, loss_reduction_60_to_100=0.148226, late_exEntity5=0.0343
- view: changed_loss_60M=2.846306, changed_loss_100M=2.550596, loss_reduction_60_to_100=0.29571, late_exEntity5=0.3853
- breadth: changed_loss_60M=4.950789, changed_loss_100M=4.728129, loss_reduction_60_to_100=0.22266, late_exEntity5=0.335

## Files
- JSON: `experiments/archive/frontier_consolidation/data/incorpus_rate_prediction/incorpus_rate_prediction.json`
- CSV: `experiments/archive/frontier_consolidation/data/incorpus_rate_prediction/incorpus_rate_losses.csv`
- Commitment JSON: `experiments/archive/frontier_consolidation/data/incorpus_rate_prediction/prediction_commitment.json`
