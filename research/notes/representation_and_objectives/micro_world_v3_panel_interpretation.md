# counterfactual micro world v3 static repair micro-world v3 panel interpretation

Status: **AGGREGATED**

This is inference-only scoring of the frozen v3 crossed-sign counterfactual micro-world. It is not a submission metric and was not tuned to make chck_82M rank highly.

## Scored target table

| target | family | crossed | balanced crossed | shuffled crossed | crossed excess | interaction median | min-margin median | renamed crossed | EWoK acc | EWoK stable frac |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scale1p75_77M | scale1p75_late_ladder | 0.7475 | 0.9949 | 0.0650 | 0.6825 | 7.3539 | 3.0946 | 0.7500 |  |  |
| scale1p75_78M | scale1p75_late_ladder | 0.7475 | 0.9949 | 0.0550 | 0.6925 | 7.3750 | 3.1280 | 0.7500 |  |  |
| scale1p75_79M | scale1p75_late_ladder | 0.7475 | 0.9949 | 0.0650 | 0.6825 | 7.3352 | 3.0598 | 0.7500 |  |  |
| scale1p75_80M | scale1p75_late_ladder | 0.7450 | 0.9899 | 0.0450 | 0.7000 | 7.1763 | 2.9013 | 0.7500 | 0.4928 | 0.3514 |
| scale1p75_81M | scale1p75_late_ladder | 0.7400 | 0.9848 | 0.0900 | 0.6500 | 7.4313 | 3.0239 | 0.7450 |  |  |
| scale1p75_82M | scale1p75_late_ladder | 0.7450 | 0.9899 | 0.0725 | 0.6725 | 7.3990 | 3.0586 | 0.7500 |  |  |
| scale1p75_83M | scale1p75_late_ladder | 0.7450 | 0.9899 | 0.0750 | 0.6700 | 7.2776 | 2.8517 | 0.7500 |  |  |
| scale1p75_100M | scale1p75_late_ladder | 0.7475 | 0.9949 | 0.0625 | 0.6850 | 7.4524 | 3.0407 | 0.7500 | 0.4965 | 0.3519 |
| legal16k_base_80M_seed43022 | legal16k_compact_seed43022 | 0.7475 | 0.9949 | 0.0625 | 0.6850 | 7.4652 | 3.0841 | 0.7425 | 0.5092 | 0.3261 |
| legal16k_base_100M_seed43022 | legal16k_compact_seed43022 | 0.7500 | 1.0000 | 0.0375 | 0.7125 | 7.6307 | 3.2218 | 0.7500 | 0.5075 | 0.3304 |
| legal16k_80M_seed43122 | legal16k_compact_seed43122 | 0.7450 | 0.9949 | 0.0475 | 0.6975 | 8.0340 | 2.9675 | 0.7450 |  |  |
| legal16k_100M_seed43122 | legal16k_compact_seed43122 | 0.7450 | 0.9949 | 0.0500 | 0.6950 | 8.0198 | 2.9476 | 0.7250 |  |  |
| legal40k_8x480_100M_seed43022 | legal40k_8x480 | 0.7050 | 0.9444 | 0.0575 | 0.6475 | 6.7142 | 2.1283 | 0.7025 | 0.5113 | 0.3347 |
| legal40k_depth12_100M_seed43022 | legal40k_depth12 | 0.6875 | 0.9091 | 0.0425 | 0.6450 | 6.1781 | 1.7156 | 0.6675 | 0.5129 | 0.3274 |
| fw_compact_100M_seed43022 | fineweb_allocation | 0.7500 | 1.0000 | 0.0700 | 0.6800 | 8.8975 | 3.5613 | 0.7500 | 0.5043 | 0.3416 |
| fw_rowblock_100M_seed43022 | fineweb_allocation | 0.7475 | 0.9949 | 0.1025 | 0.6450 | 5.4857 | 2.1799 | 0.7450 | 0.4957 | 0.3257 |
| mlm_only_20M | coupled_sparse20_turnover_panel | 0.2375 | 0.4343 | 0.0275 | 0.2100 | 0.6787 | -0.8205 | 0.2700 | 0.4992 | 0.3431 |
| coupled_aligned_20M | coupled_sparse20_turnover_panel | 0.0050 | 0.0051 | 0.0000 | 0.0050 | 0.0281 | -1.2258 | 0.0075 | 0.4982 | 0.3106 |
| coupled_shuffled_20M | coupled_sparse20_turnover_panel | 0.0150 | 0.0303 | 0.0050 | 0.0100 | 0.0070 | -1.5035 | 0.0175 | 0.5018 | 0.2936 |

## Coupled turnover reading

```json
{
  "aligned_minus_mlm_crossed": -0.23249999999999998,
  "aligned_minus_mlm_crossed_excess": -0.205,
  "aligned_minus_shuffled_crossed": -0.009999999999999998,
  "aligned_minus_shuffled_crossed_excess": -0.005,
  "available": [
    "coupled_aligned_20M",
    "coupled_shuffled_20M",
    "mlm_only_20M"
  ],
  "full_ewok_reminder": {
    "aligned_accuracy": 0.498162247309005,
    "aligned_stable_failure_frac": 0.310580204778157,
    "mlm_accuracy": 0.4992123917038593,
    "mlm_stable_failure_frac": 0.3431346810186401,
    "shuffled_accuracy": 0.5018377526909951,
    "shuffled_stable_failure_frac": 0.29364662641113154
  },
  "shuffled_minus_mlm_crossed": -0.22249999999999998,
  "shuffled_minus_mlm_crossed_excess": -0.19999999999999998,
  "turnover_rejection_reading": "sensitive_to_coupled_trajectory_or_turnover; do not use as training selector without deeper analysis"
}
```

## Correlations to existing full-EWoK four-cell references

```json
{
  "balanced_crossed": {
    "ewok_accuracy": {
      "n": 11,
      "pearson": 0.2183397927456013,
      "spearman": 0.04138204408845326
    },
    "ewok_interaction_mean": {
      "n": 11,
      "pearson": 0.4575510118843377,
      "spearman": 0.3494483723024942
    },
    "ewok_interaction_median": {
      "n": 11,
      "pearson": 0.545891628545029,
      "spearman": 0.5241725584537413
    },
    "ewok_stable_failure_frac_all": {
      "n": 11,
      "pearson": 0.6991097544507414,
      "spearman": 0.37703640169479635
    }
  },
  "crossed": {
    "ewok_accuracy": {
      "n": 11,
      "pearson": 0.22570898553212643,
      "spearman": 0.04138204408845326
    },
    "ewok_interaction_mean": {
      "n": 11,
      "pearson": 0.46354463145964564,
      "spearman": 0.3494483723024942
    },
    "ewok_interaction_median": {
      "n": 11,
      "pearson": 0.5637728989154266,
      "spearman": 0.5241725584537413
    },
    "ewok_stable_failure_frac_all": {
      "n": 11,
      "pearson": 0.6620536946995682,
      "spearman": 0.37703640169479635
    }
  },
  "crossed_excess": {
    "ewok_accuracy": {
      "n": 11,
      "pearson": 0.24499710759287532,
      "spearman": 0.022831288248861156
    },
    "ewok_interaction_mean": {
      "n": 11,
      "pearson": 0.44367732894280726,
      "spearman": 0.07762638004612793
    },
    "ewok_interaction_median": {
      "n": 11,
      "pearson": 0.5557631786274058,
      "spearman": 0.35616809668223404
    },
    "ewok_stable_failure_frac_all": {
      "n": 11,
      "pearson": 0.6661432065420089,
      "spearman": 0.5707822062215289
    }
  },
  "interaction_mean": {
    "ewok_accuracy": {
      "n": 11,
      "pearson": 0.17933283244762627,
      "spearman": 0.08181818181818182
    },
    "ewok_interaction_mean": {
      "n": 11,
      "pearson": 0.3906201738143277,
      "spearman": 0.3181818181818182
    },
    "ewok_interaction_median": {
      "n": 11,
      "pearson": 0.4455705967069442,
      "spearman": 0.4727272727272727
    },
    "ewok_stable_failure_frac_all": {
      "n": 11,
      "pearson": 0.570016273899655,
      "spearman": 0.4
    }
  },
  "interaction_median": {
    "ewok_accuracy": {
      "n": 11,
      "pearson": 0.2786793597182659,
      "spearman": 0.22727272727272727
    },
    "ewok_interaction_mean": {
      "n": 11,
      "pearson": 0.3403291393252473,
      "spearman": 0.18181818181818182
    },
    "ewok_interaction_median": {
      "n": 11,
      "pearson": 0.46829025803407837,
      "spearman": 0.4
    },
    "ewok_stable_failure_frac_all": {
      "n": 11,
      "pearson": 0.6220278889980099,
      "spearman": 0.5181818181818182
    }
  },
  "min_margin_median": {
    "ewok_accuracy": {
      "n": 11,
      "pearson": 0.1863606520112623,
      "spearman": 0.07272727272727272
    },
    "ewok_interaction_mean": {
      "n": 11,
      "pearson": 0.342324488074837,
      "spearman": 0.23636363636363636
    },
    "ewok_interaction_median": {
      "n": 11,
      "pearson": 0.44850659450565966,
      "spearman": 0.42727272727272725
    },
    "ewok_stable_failure_frac_all": {
      "n": 11,
      "pearson": 0.6405801889400846,
      "spearman": 0.4636363636363636
    }
  },
  "strong_crossed_gt_0p1": {
    "ewok_accuracy": {
      "n": 11,
      "pearson": 0.22045038887532,
      "spearman": 0.032261147658728506
    },
    "ewok_interaction_mean": {
      "n": 11,
      "pearson": 0.46243074380817645,
      "spearman": 0.3456551534863768
    },
    "ewok_interaction_median": {
      "n": 11,
      "pearson": 0.5588669749066026,
      "spearman": 0.5161783625396561
    },
    "ewok_stable_failure_frac_all": {
      "n": 11,
      "pearson": 0.6563557055964807,
      "spearman": 0.3825250365249237
    }
  }
}
```

JSON: `experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3_scores/micro_world_v3_panel_summary.json`
