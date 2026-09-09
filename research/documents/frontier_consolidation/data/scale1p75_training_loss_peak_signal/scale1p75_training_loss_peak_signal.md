# chck82 reproducibility and peak characterization scale1.75 training-loss peak signal

Status: `SCALE1P75_TRAINING_LOSS_PEAK_SIGNAL`

| checkpoint | cheap7 | step | lr | loss | trail25 | trail50 | trail100 | centered51 | slope50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chck_77M | 43.28214285714286 | 1948 | 0.00014019757 | 2.501409 | 2.515159 | 2.521203 | 2.534974 | 2.510378 | -0.00054251 |
| chck_78M | 43.70214285714286 | 1973 | 0.00012892893 | 2.594009 | 2.508044 | 2.511601 | 2.524599 | 2.515359 | -3.43679e-05 |
| chck_79M | 43.57857142857143 | 1998 | 0.00011806504 | 2.561792 | 2.523233 | 2.515638 | 2.518421 | 2.520318 | 0.000938027 |
| chck_80M | 43.81214285714286 | 2024 | 0.00010720866 | 2.583350 | 2.517500 | 2.521510 | 2.516808 | 2.499123 | -0.00033021 |
| chck_81M | 43.64928571428572 | 2049 | 9.7206687e-05 | 2.528909 | 2.480423 | 2.498961 | 2.508132 | 2.489626 | -0.00115009 |
| chck_82M | 43.96 | 2074 | 8.7644049e-05 | 2.511317 | 2.495081 | 2.487752 | 2.504631 | 2.489822 | 0.00058401 |
| chck_83M | 43.807857142857145 | 2099 | 7.8531179e-05 | 2.490498 | 2.483000 | 2.489041 | 2.494001 | 2.474599 | 0.000228083 |

## Selector extrema

```json
{
  "matched_log_index0": {
    "min_checkpoint": "chck_77M",
    "min_value": 1947,
    "max_checkpoint": "chck_83M",
    "max_value": 2098
  },
  "matched_step": {
    "min_checkpoint": "chck_77M",
    "min_value": 1948,
    "max_checkpoint": "chck_83M",
    "max_value": 2099
  },
  "matched_cumulative_words": {
    "min_checkpoint": "chck_77M",
    "min_value": 77028243,
    "max_checkpoint": "chck_83M",
    "max_value": 83000458
  },
  "matched_loss": {
    "min_checkpoint": "chck_83M",
    "min_value": 2.4904980659484863,
    "max_checkpoint": "chck_78M",
    "max_value": 2.5940093994140625
  },
  "matched_lr": {
    "min_checkpoint": "chck_83M",
    "min_value": 7.853117862802173e-05,
    "max_checkpoint": "chck_77M",
    "max_value": 0.00014019756674789498
  },
  "trailing_loss_10": {
    "min_checkpoint": "chck_81M",
    "min_value": 2.477108860015869,
    "max_checkpoint": "chck_78M",
    "max_value": 2.528304839134216
  },
  "trailing_loss_25": {
    "min_checkpoint": "chck_81M",
    "min_value": 2.480422992706299,
    "max_checkpoint": "chck_79M",
    "max_value": 2.5232331943511963
  },
  "trailing_loss_50": {
    "min_checkpoint": "chck_82M",
    "min_value": 2.4877520179748536,
    "max_checkpoint": "chck_80M",
    "max_value": 2.52151038646698
  },
  "trailing_loss_100": {
    "min_checkpoint": "chck_83M",
    "min_value": 2.4940010356903075,
    "max_checkpoint": "chck_77M",
    "max_value": 2.534973752498627
  },
  "centered_loss_21": {
    "min_checkpoint": "chck_82M",
    "min_value": 2.4782287166232155,
    "max_checkpoint": "chck_79M",
    "max_value": 2.5255311784290133
  },
  "centered_loss_51": {
    "min_checkpoint": "chck_83M",
    "min_value": 2.474598861208149,
    "max_checkpoint": "chck_79M",
    "max_value": 2.5203176573211072
  },
  "forward_loss_25": {
    "min_checkpoint": "chck_83M",
    "min_value": 2.464728832244873,
    "max_checkpoint": "chck_78M",
    "max_value": 2.5232331943511963
  },
  "forward_loss_50": {
    "min_checkpoint": "chck_83M",
    "min_value": 2.467093825340271,
    "max_checkpoint": "chck_78M",
    "max_value": 2.518843822479248
  },
  "trailing_loss_slope_per_step_25": {
    "min_checkpoint": "chck_77M",
    "min_value": -0.0011435378514803372,
    "max_checkpoint": "chck_83M",
    "max_value": 0.004397616386413574
  },
  "trailing_loss_slope_per_step_50": {
    "min_checkpoint": "chck_81M",
    "min_value": -0.0011500949332980263,
    "max_checkpoint": "chck_79M",
    "max_value": 0.0009380274975285525
  },
  "trailing_loss_slope_per_step_100": {
    "min_checkpoint": "chck_78M",
    "min_value": -0.000481261671966452,
    "max_checkpoint": "chck_80M",
    "max_value": 5.08463178852675e-05
  }
}
```

## Correlations with cheap7

```json
{
  "matched_log_index0": 0.7702349729568428,
  "matched_step": 0.7702349729568428,
  "matched_cumulative_words": 0.7702999806711863,
  "matched_loss": 0.11380133414469643,
  "matched_lr": -0.780660206407826,
  "trailing_loss_10": 0.16015613530168882,
  "trailing_loss_25": -0.4010987399273683,
  "trailing_loss_50": -0.6318329577348245,
  "trailing_loss_100": -0.7495652467007263,
  "centered_loss_21": -0.3293968682904622,
  "centered_loss_51": -0.5503335928086344,
  "forward_loss_25": -0.578474943276563,
  "forward_loss_50": -0.717295967088511,
  "trailing_loss_slope_per_step_25": 0.32172710106013525,
  "trailing_loss_slope_per_step_50": 0.33307716802449266,
  "trailing_loss_slope_per_step_100": 0.07560807513654225
}
```

Training loss would be a useful legal stopping signal only if its extrema or slope singled out chck_82M, or at least warned against continuing. If loss keeps improving or chooses another checkpoint while cheap7 peaks at 82M, then simple MLM-loss monitoring is not a sufficient benchmark-independent selector.

JSON: `experiments/archive/frontier_consolidation/data/scale1p75_training_loss_peak_signal/scale1p75_training_loss_peak_signal.json`
