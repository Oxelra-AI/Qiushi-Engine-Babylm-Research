# earlier analysis source/rank decomposition including dense-mask/sparse-label

Input: `experiments/archive/functional_learning/data/temperature_source_readout_five_models/temperature_source_readout.json`

## Compact model comparison

| model | T | calib NLL | Qwen Δspec Tfit | Qwen Δrank | common Δswing Tfit | common Δrank swing | both T1/rank |
|---|---:|---:|---:|---:|---:|---:|---|
| coherent86 | 1.1 | None | 0.0 | 0.0 | 0.0 | 0.0 | 20/20 |
| sparse_focus_seed62064 | 1.1 | None | 0.0020557124912738848 | -5.413194444444444 | -0.04505195023048483 | 2.2390476190476343 | 21/21 |
| dense_focus_seed62064 | 1.1 | None | 0.15194935254894923 | 69.89072916666667 | 0.538546733387879 | 115.20476190476192 | 21/20 |
| dense_focus_seed62065 | 1.1 | None | 0.15540364948243626 | 71.86121527777777 | 0.531859248904955 | 116.42809523809525 | 21/20 |
| densemask_sparselabel_seed62064 | 1.1 | None | 0.14678865148714978 | 66.97135416666667 | 0.5443574689115799 | 117.79809523809524 | 21/20 |

## Pairwise localization

### MS_minus_SS_dense_input_at_sparse_labels

{
  "qwen_specific_delta_a_minus_b": {
    "T1": 0.16747582704449693,
    "Tfit": 0.1447329389958759,
    "rank": 72.38454861111111
  },
  "qwen_condition_delta_a_minus_b": {
    "correct_source": {
      "nll_T1": 0.05196394950668845,
      "nll_Tfit": 0.04025855213403701,
      "rank": 11.904131944444444
    },
    "wrong_source": {
      "nll_T1": 0.2194397765511854,
      "nll_Tfit": 0.1849914911299129,
      "rank": 84.28868055555557
    },
    "view_only": {
      "nll_T1": 0.16379138877304894,
      "nll_Tfit": 0.13711334431854388,
      "rank": 49.98961805555555
    }
  },
  "common_delta_a_minus_b": {
    "swing_T1": 0.6586676554452806,
    "swing_Tfit": 0.5894094191420647,
    "rank_swing": 115.5590476190476,
    "both_T1": 0.0,
    "both_rank": -1.0
  }
}

### MS_minus_MM64_sparse_labels_vs_dense_coverage

{
  "qwen_specific_delta_a_minus_b": {
    "T1": -0.00516286078737016,
    "Tfit": -0.005160701061799444,
    "rank": -2.9193750000000023
  },
  "qwen_condition_delta_a_minus_b": {
    "correct_source": {
      "nll_T1": 0.006190931787714366,
      "nll_Tfit": 0.005002241651010181,
      "rank": -0.06847222222222271
    },
    "wrong_source": {
      "nll_T1": 0.001028071000344244,
      "nll_Tfit": -0.00015845941078927728,
      "rank": -2.9878472222222143
    },
    "view_only": {
      "nll_T1": 0.002600810405694806,
      "nll_Tfit": 0.0014071802427578356,
      "rank": -2.0621875000000074
    }
  },
  "common_delta_a_minus_b": {
    "swing_T1": 0.0064764638457979196,
    "swing_Tfit": 0.005810735523700927,
    "rank_swing": 2.5933333333333195,
    "both_T1": 0.0,
    "both_rank": 0.0
  }
}

### MM64_minus_SS_dense_total_effect

{
  "qwen_specific_delta_a_minus_b": {
    "T1": 0.1726386878318671,
    "Tfit": 0.14989364005767533,
    "rank": 75.30392361111112
  },
  "qwen_condition_delta_a_minus_b": {
    "correct_source": {
      "nll_T1": 0.04577301771897408,
      "nll_Tfit": 0.035256310483026826,
      "rank": 11.972604166666667
    },
    "wrong_source": {
      "nll_T1": 0.21841170555084116,
      "nll_Tfit": 0.18514995054070219,
      "rank": 87.27652777777777
    },
    "view_only": {
      "nll_T1": 0.16119057836735412,
      "nll_Tfit": 0.13570616407578606,
      "rank": 52.05180555555556
    }
  },
  "common_delta_a_minus_b": {
    "swing_T1": 0.6521911915994827,
    "swing_Tfit": 0.5835986836183638,
    "rank_swing": 112.96571428571428,
    "both_T1": 0.0,
    "both_rank": -1.0
  }
}

## Interpretation

- The dense-mask/sparse-label arm reproduces the dense source/rank signature under sparse labels: its Qwen Δspecific advantage, common source-follow swing, decision counts, and rank-swing are dense-like rather than sparse-like.
- Against dense seed62064, dense-mask differs only slightly on these source diagnostics: Qwen Δspecific Tfit is lower by about 0.005 and Δrank by about 2.9, while common source-follow Tfit/rank swings are slightly larger. Broad dense target coverage is therefore not necessary for the observed source-conditioned shift under this 80-update policy.
- The preservation problem remains: dense-mask calibration NLL on ordinary legal-tail text is worse than dense and sparse, and earlier analysis/earlier analysis fast/AoA-adjacent evidence shows broad likelihood/rank costs are not fixed by thinning labels. The next method should preserve the clue-suppressed source-visible acquisition condition while explicitly constraining drift on matched view-only or ordinary-corruption renderings where evidence is absent or underdetermined.
