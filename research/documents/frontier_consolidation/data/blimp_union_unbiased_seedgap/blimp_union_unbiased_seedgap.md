# earlier analysis — unbiased BLiMP union seed-gap (reinvest sparse temporal)

blimp_worst and blimp_control were defined by seed43122's 100M outcome, so their individual 100M gaps (-14.05 and +10.06) are inflated by selection. The 18-file union gap is the unbiased reinvest BLiMP seed spread and is the correct quantity to compare against clean-Qwen.

## Union (18 BLiMP files) macro seed gap seed43122-minus-seed43022 by exposure
- 1M: union_gap=-0.361 (worst=-7.000, control=7.938)
- 10M: union_gap=1.944 (worst=2.150, control=1.688)
- 40M: union_gap=-0.250 (worst=-1.300, control=1.062)
- 100M: union_gap=-3.333 (worst=-14.050, control=10.062)

## Trajectory
- union early change 1->10M: 2.3055555555555554
- union late change 40->100M: -3.0833333333333335
- union final 100M gap: -3.3333333333333335

At 100M the union macro gap is -3.3333333333333335, versus the selected worst -14.05 and control 10.0625.

Machine-readable output: `experiments/archive/frontier_consolidation/data/blimp_union_unbiased_seedgap/blimp_union_unbiased_seedgap.json`
