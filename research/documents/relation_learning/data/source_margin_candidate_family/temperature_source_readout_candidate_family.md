# earlier analysis source/rank decomposition for candidate family

Anchor is `chck82_slow_scale1p75`, the protected parent before private dense-credit training. Temperatures are fitted only on ordinary legal-tail text, then source-visible probes are scored on Qwen-pair and common source-reversal banks. This readout is explanatory, not official BabyLM scoring.

## Compact source-use table

| model | T | Qwen Δspecific Tfit | Qwen Δspecific rank | view-only ΔNLL Tfit | view-only Δrank | common Δswing Tfit | common Δrank swing | common both Tfit/rank |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| chck82_slow_scale1p75 | 1.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 20/20 |
| coherent86_alpha075 | 1.1 | 0.007922278449016934 | -4.670208333333334 | -0.003875005129246155 | -2.8005902777777787 | 0.037433195632128324 | 5.347619047619027 | 20/20 |
| dense64_u0080 | 1.1 | 0.15987163099796617 | 65.22052083333332 | 0.06636718352021286 | 41.634652777777774 | 0.5759799290200074 | 120.55238095238096 | 21/20 |
| dense65_u0080 | 1.1 | 0.1633259279314532 | 67.19100694444445 | 0.06845254282628756 | 42.45333333333333 | 0.5692924445370832 | 121.77571428571427 | 21/20 |
| clean_pres64_u0080 | 1.1 | 0.10866824603717154 | 38.02628472222223 | 0.04728865928436991 | 24.484930555555554 | 0.3833086365816139 | 80.22142857142856 | 21/20 |
| clean_pres65_u0080 | 1.1 | 0.10832930971092235 | 38.471250000000005 | 0.04628406787827973 | 24.284930555555555 | 0.379381176382303 | 78.48142857142855 | 21/20 |

## Seed-paired clean-preservation minus dense

Positive Qwen/common deltas mean preservation kept or increased source-responsive movement; negative view-only / ordinary drift deltas mean preservation reduced evidence-absent drift relative to dense.

### clean_minus_dense_seed64

{
  "qwen_delta_spec_Tfit_vs_chck82": -0.05120338496079463,
  "qwen_delta_spec_rank_vs_chck82": -27.194236111111096,
  "qwen_delta_total_Tfit_vs_chck82": -0.02904318213669789,
  "common_delta_swing_Tfit_vs_chck82": -0.19267129243839348,
  "common_delta_rank_swing_vs_chck82": -40.3309523809524,
  "correct_source_delta_nll_Tfit_vs_chck82": 0.009964657900854944,
  "correct_source_delta_rank_vs_chck82": -4.591597222222222,
  "wrong_source_delta_nll_Tfit_vs_chck82": -0.04123872705993968,
  "wrong_source_delta_rank_vs_chck82": -31.785833333333336,
  "view_only_delta_nll_Tfit_vs_chck82": -0.019078524235842942,
  "view_only_delta_rank_vs_chck82": -17.14972222222222
}

### clean_minus_dense_seed65

{
  "qwen_delta_spec_Tfit_vs_chck82": -0.054996618220530855,
  "qwen_delta_spec_rank_vs_chck82": -28.71975694444445,
  "qwen_delta_total_Tfit_vs_chck82": -0.03187503278087306,
  "common_delta_swing_Tfit_vs_chck82": -0.18991126815478016,
  "common_delta_rank_swing_vs_chck82": -43.29428571428572,
  "correct_source_delta_nll_Tfit_vs_chck82": 0.009706557832865254,
  "correct_source_delta_rank_vs_chck82": -4.864583333333333,
  "wrong_source_delta_nll_Tfit_vs_chck82": -0.0452900603876656,
  "wrong_source_delta_rank_vs_chck82": -33.584340277777784,
  "view_only_delta_nll_Tfit_vs_chck82": -0.022168474948007826,
  "view_only_delta_rank_vs_chck82": -18.16840277777778
}

### dense65_minus_dense64

{
  "qwen_delta_spec_Tfit_vs_chck82": 0.0034542969334870333,
  "qwen_delta_spec_rank_vs_chck82": 1.9704861111111285,
  "qwen_delta_total_Tfit_vs_chck82": 0.00259333479787327,
  "common_delta_swing_Tfit_vs_chck82": -0.006687484482924222,
  "common_delta_rank_swing_vs_chck82": 1.223333333333315,
  "correct_source_delta_nll_Tfit_vs_chck82": -0.0005079754917985888,
  "correct_source_delta_rank_vs_chck82": -0.15736111111111128,
  "wrong_source_delta_nll_Tfit_vs_chck82": 0.0029463214416884514,
  "wrong_source_delta_rank_vs_chck82": 1.8131249999999994,
  "view_only_delta_nll_Tfit_vs_chck82": 0.002085359306074702,
  "view_only_delta_rank_vs_chck82": 0.8186805555555594
}

### clean65_minus_clean64

{
  "qwen_delta_spec_Tfit_vs_chck82": -0.00033893632624919234,
  "qwen_delta_spec_rank_vs_chck82": 0.4449652777777757,
  "qwen_delta_total_Tfit_vs_chck82": -0.00023851584630189826,
  "common_delta_swing_Tfit_vs_chck82": -0.003927460199310895,
  "common_delta_rank_swing_vs_chck82": -1.740000000000009,
  "correct_source_delta_nll_Tfit_vs_chck82": -0.0007660755597882797,
  "correct_source_delta_rank_vs_chck82": -0.4303472222222222,
  "wrong_source_delta_nll_Tfit_vs_chck82": -0.0011050118860374686,
  "wrong_source_delta_rank_vs_chck82": 0.014618055555551734,
  "view_only_delta_nll_Tfit_vs_chck82": -0.0010045914060901814,
  "view_only_delta_rank_vs_chck82": -0.1999999999999993
}

## Initial interpretation

- The source-responsive quantities are Qwen correct-vs-wrong specific advantage and common source-follow swing, especially rank movement because it cannot be explained by global temperature.
- The evidence-absent drift quantities are view-only NLL/rank and CDI endpoint NLL/rank in the companion earlier analysis CDI profile; source probes alone cannot settle broad lexical preservation.
- If clean-preservation has dense-like source-responsive deltas while view-only and CDI drifts shrink relative to dense, the recipe is not simply less private movement: it selectively preserves evidence-supported acquisition while damping unsupported distributional movement.
