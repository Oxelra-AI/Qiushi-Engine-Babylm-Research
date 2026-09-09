# earlier analysis source/rank decomposition including dense-mask/sparse-label

Input: `experiments/archive/functional_learning/data/temperature_source_readout_clean/temperature_source_readout.json`

## Compact model comparison

| model | T | calib NLL | Qwen Δspec Tfit | Qwen Δrank | common Δswing Tfit | common Δrank swing | both T1/rank |
|---|---:|---:|---:|---:|---:|---:|---|
| coherent86 | 1.1 | None | 0.0 | 0.0 | 0.0 | 0.0 | 20/20 |
| sparse_focus_seed62064 | 1.1 | None | 0.00205627047249841 | -5.413194444444444 | -0.0450534470308394 | 2.2390476190476343 | 21/21 |
| densemask_sparselabel_seed62064 | 1.1 | None | 0.14678872493386735 | 66.97135416666667 | 0.5443552198083627 | 117.79809523809524 | 21/20 |
| dense_focus_seed62064 | 1.1 | None | 0.15194994646137477 | 69.8903125 | 0.5385464426733197 | 115.20476190476192 | 21/20 |
| dense_focus_seed62065 | 1.1 | None | 0.15540449084933952 | 71.86121527777777 | 0.5318558227590153 | 116.42809523809525 | 21/20 |
| clean_pres_lambda1_eval_full80 | 1.1 | None | 0.10074627038440666 | 42.69649305555556 | 0.34587322203885923 | 74.87380952380953 | 21/20 |
| clean_pres_lambda1_train_full80 | 1.1 | None | 0.10118413289058177 | 44.335972222222225 | 0.4939070497382255 | 99.60714285714286 | 21/20 |
| pres_lambda1_trainmode_confounded | 1.1 | None | 0.10053548814127376 | 44.140277777777776 | 0.4917391726374624 | 99.52047619047622 | 21/21 |

## Pairwise localization

### MS_minus_SS_dense_input_at_sparse_labels

{
  "qwen_specific_delta_a_minus_b": {
    "T1": 0.16747531998708537,
    "Tfit": 0.14473245446136895,
    "rank": 72.38454861111111
  },
  "qwen_condition_delta_a_minus_b": {
    "correct_source": {
      "nll_T1": 0.05196436447091401,
      "nll_Tfit": 0.04025885993407832,
      "rank": 11.904131944444444
    },
    "wrong_source": {
      "nll_T1": 0.21943968445799933,
      "nll_Tfit": 0.18499131439544725,
      "rank": 84.28868055555557
    },
    "view_only": {
      "nll_T1": 0.16379145303534137,
      "nll_Tfit": 0.1371133878731376,
      "rank": 49.9903125
    }
  },
  "common_delta_a_minus_b": {
    "swing_T1": 0.6586666736858231,
    "swing_Tfit": 0.589408666839202,
    "rank_swing": 115.5590476190476,
    "both_T1": 0.0,
    "both_rank": -1.0
  }
}

### MS_minus_MM64_sparse_labels_vs_dense_coverage

{
  "qwen_specific_delta_a_minus_b": {
    "T1": -0.0051634183708422765,
    "Tfit": -0.00516122152750742,
    "rank": -2.9189583333333218
  },
  "qwen_condition_delta_a_minus_b": {
    "correct_source": {
      "nll_T1": 0.006191093917522164,
      "nll_Tfit": 0.005002395746608573,
      "rank": -0.068888888888889
    },
    "wrong_source": {
      "nll_T1": 0.001027675546679846,
      "nll_Tfit": -0.0001588257808988469,
      "rank": -2.9878472222222143
    },
    "view_only": {
      "nll_T1": 0.002601123638968486,
      "nll_Tfit": 0.001407447772669712,
      "rank": -2.0583680555555546
    }
  },
  "common_delta_a_minus_b": {
    "swing_T1": 0.0064745394814581525,
    "swing_Tfit": 0.0058087771350429085,
    "rank_swing": 2.5933333333333195,
    "both_T1": 0.0,
    "both_rank": 0.0
  }
}

### MM64_minus_SS_dense_total_effect

{
  "qwen_specific_delta_a_minus_b": {
    "T1": 0.17263873835792765,
    "Tfit": 0.14989367598887637,
    "rank": 75.30350694444444
  },
  "qwen_condition_delta_a_minus_b": {
    "correct_source": {
      "nll_T1": 0.045773270553391844,
      "nll_Tfit": 0.03525646418746975,
      "rank": 11.973020833333333
    },
    "wrong_source": {
      "nll_T1": 0.21841200891131948,
      "nll_Tfit": 0.1851501401763461,
      "rank": 87.27652777777777
    },
    "view_only": {
      "nll_T1": 0.16119032939637287,
      "nll_Tfit": 0.13570594010046788,
      "rank": 52.048680555555556
    }
  },
  "common_delta_a_minus_b": {
    "swing_T1": 0.652192134204365,
    "swing_Tfit": 0.5835998897041591,
    "rank_swing": 112.96571428571428,
    "both_T1": 0.0,
    "both_rank": -1.0
  }
}

## Interpretation

- The dense-mask/sparse-label arm reproduces the dense source/rank signature under sparse labels: its Qwen Δspecific advantage, common source-follow swing, decision counts, and rank-swing are dense-like rather than sparse-like.
- Against dense seed62064, dense-mask differs only slightly on these source diagnostics: Qwen Δspecific Tfit is lower by about 0.005 and Δrank by about 2.9, while common source-follow Tfit/rank swings are slightly larger. Broad dense target coverage is therefore not necessary for the observed source-conditioned shift under this 80-update policy.
- The preservation problem remains: dense-mask calibration NLL on ordinary legal-tail text is worse than dense and sparse, and earlier analysis/earlier analysis fast/AoA-adjacent evidence shows broad likelihood/rank costs are not fixed by thinning labels. The next method should preserve the clue-suppressed source-visible acquisition condition while explicitly constraining drift on matched view-only or ordinary-corruption renderings where evidence is absent or underdetermined.
