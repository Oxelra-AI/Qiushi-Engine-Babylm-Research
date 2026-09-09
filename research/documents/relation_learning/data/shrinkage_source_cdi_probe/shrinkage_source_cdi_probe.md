# earlier analysis shrinkage-null source/CDI probe

Dense seed62064 was evaluated at lower private adapter scales without changing weights. Scale 0.60 was chosen because the earlier analysis KL screen makes its coherent-row KL nearly equal to clean-pres64.

## Source-responsive and CDI drift table

| model | scale | Qwen Δspec Tfit | Qwen Δspec rank | common Δswing Tfit | common Δrank | view-only ΔNLL | CDI ΔNLL | CDI Δrank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chck82 | -1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dense64_scale0p75 | 0.75 | 0.15987163099796617 | 65.22052083333332 | 0.5759799290200074 | 120.55238095238096 | 0.06636718352021286 | 0.05203152800383989 | 41.81016533986528 |
| dense64_scale0p55 | 0.55 | 0.09799624592169291 | 35.60375 | 0.44797383837047067 | 89.61476190476189 | 0.03437902509584091 | 0.02911808717514252 | 23.45499081445193 |
| dense64_scale0p57 | 0.57 | 0.10361014041661595 | 38.067118055555554 | 0.4620764822690259 | 92.58238095238096 | 0.03712778833237986 | 0.031086593459661 | 24.924066135946113 |
| dense64_scale0p58 | 0.58 | 0.106467993604278 | 39.44829861111111 | 0.4690238355383985 | 94.26238095238094 | 0.038540511915246795 | 0.03209710478142762 | 25.775260257195345 |
| dense64_scale0p60 | 0.6 | 0.11228396419325794 | 42.29944444444445 | 0.4827130446192763 | 97.21904761904761 | 0.04144226582973108 | 0.034171250909501294 | 27.3943661971831 |
| clean_pres64 | 0.75 | 0.10866824603717154 | 38.02628472222223 | 0.3833086365816139 | 80.22142857142856 | 0.04728865928436991 | -0.014410393931782823 | -22.10838946723821 |

## Matched-KL contrast: clean_pres64 minus dense64_scale0p60

{
  "qwen_delta_spec_Tfit": -0.0036157181560863977,
  "cdi_delta_rank": -49.502755664421315,
  "common_delta_rank_swing": -16.997619047619054,
  "cdi_word_mean_delta_rank": -51.628502747183475,
  "common_delta_swing_Tfit": -0.09940440803766237,
  "view_only_delta_rank": -4.235590277777781,
  "cdi_improved_fraction_rank": 0.08756889161053272,
  "calib_nll_T1": 0.0035872075420160243,
  "cdi_delta_nll_Tfit": -0.04858164484128412,
  "qwen_delta_total_Tfit": 0.0006895165431261563,
  "temperature": 0.0,
  "cdi_mean_rank": -49.50275566442133,
  "correct_source_delta_rank": -2.1222569444444437,
  "wrong_source_delta_rank": -6.395416666666669,
  "correct_source_delta_nll_Tfit": 0.005156876911512679,
  "scale": 0.15000000000000002,
  "qwen_delta_total_rank": -2.1133333333333333,
  "view_only_delta_nll_Tfit": 0.005846393454638832,
  "cdi_mean_nll_Tfit": -0.04858164484128302,
  "cdi_word_mean_delta_nll_Tfit": -0.04923024177644618,
  "qwen_delta_spec_rank": -4.273159722222218,
  "common_both_Tfit": 0.0,
  "wrong_source_delta_nll_Tfit": 0.0015411587554263056
}

## Interpretation

- The matched-KL shrinkage probe is strongest at dense64_scale0p60 because its coherent-row KL nearly matches clean_pres64 in the earlier analysis KL screen.
- If clean_pres64 has clearly larger source-responsive Qwen/common movement than dense64_scale0p60 while not having worse CDI drift, preservation cannot be reduced to a smaller effective private-adapter scale.
- If dense64_scale0p60 matches clean_pres64 on both source movement and CDI drift, the shrinkage null remains plausible and the full cheap7 GPU control becomes more important.
