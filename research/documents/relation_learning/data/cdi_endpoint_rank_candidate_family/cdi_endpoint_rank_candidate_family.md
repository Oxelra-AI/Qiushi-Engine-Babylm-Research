# earlier analysis CDI endpoint rank profile for candidate family

Anchor is `chck82_slow_scale1p75`. This endpoint-only CDI readout measures evidence-absent lexical/word-context drift. It is separate from official AoA trajectory scoring.

## Compact table

| model | T | mean NLL Tfit | mean rank | ΔNLL Tfit vs chck82 | Δrank vs chck82 | rank improved fraction |
|---|---:|---:|---:|---:|---:|---:|
| chck82_slow_scale1p75 | None | 8.209130253223902 | 1754.5523576240048 | 0.0 | 0.0 | 0.0 |
| coherent86_alpha075 | None | 8.132485504525064 | 1686.4427434170238 | -0.07664474869883789 | -68.10961420698102 | 0.6117575015309247 |
| dense64_u0080 | None | 8.261161781227742 | 1796.36252296387 | 0.05203152800383989 | 41.81016533986528 | 0.4188609920391917 |
| dense65_u0080 | None | 8.26458585750858 | 1798.5137783221066 | 0.055455604284677815 | 43.961420698101655 | 0.4219228413962033 |
| clean_pres64_u0080 | None | 8.19471985929212 | 1732.4439681567667 | -0.014410393931782823 | -22.10838946723821 | 0.515003061849357 |
| clean_pres65_u0080 | None | 8.199311857615909 | 1737.6080832823025 | -0.009818395607992414 | -16.94427434170239 | 0.5070422535211268 |

## Seed-paired clean-preservation minus dense

Negative ΔNLL/Δrank here means preservation reduced dense endpoint CDI drift.

### clean_minus_dense_seed64

{
  "delta_nll_T1_vs_chck82": -0.07422999416028,
  "delta_nll_Tfit_vs_chck82": -0.0664419219356227,
  "delta_rank_vs_chck82": -63.91855480710349,
  "median_delta_rank_vs_chck82": -3.0,
  "improved_fraction_nll_Tfit": 0.09001837109614208,
  "improved_fraction_rank": 0.09614206981016526,
  "word_mean_delta_nll_Tfit": -0.06371339572749271,
  "word_mean_delta_rank": -64.3155270903773,
  "word_improved_fraction_rank": 0.12903225806451613
}

### clean_minus_dense_seed65

{
  "delta_nll_T1_vs_chck82": -0.07360471268136007,
  "delta_nll_Tfit_vs_chck82": -0.06527399989267023,
  "delta_rank_vs_chck82": -60.905695039804044,
  "median_delta_rank_vs_chck82": -2.0,
  "improved_fraction_nll_Tfit": 0.08573178199632575,
  "improved_fraction_rank": 0.08511941212492347,
  "word_mean_delta_nll_Tfit": -0.06179631972238311,
  "word_mean_delta_rank": -60.85658244958655,
  "word_improved_fraction_rank": 0.11827956989247312
}

### dense65_minus_dense64

{
  "delta_nll_T1_vs_chck82": 0.004066156977371296,
  "delta_nll_Tfit_vs_chck82": 0.0034240762808379274,
  "delta_rank_vs_chck82": 2.1512553582363765,
  "median_delta_rank_vs_chck82": -1.0,
  "improved_fraction_nll_Tfit": -0.0018371096142069665,
  "improved_fraction_rank": 0.0030618493570115923,
  "word_mean_delta_nll_Tfit": 0.002423256851237357,
  "word_mean_delta_rank": 1.235374632266165,
  "word_improved_fraction_rank": -0.021505376344086002
}

### clean65_minus_clean64

{
  "delta_nll_T1_vs_chck82": 0.004691438456291228,
  "delta_nll_Tfit_vs_chck82": 0.004591998323790409,
  "delta_rank_vs_chck82": 5.164115125535822,
  "median_delta_rank_vs_chck82": 0.0,
  "improved_fraction_nll_Tfit": -0.006123698714023296,
  "improved_fraction_rank": -0.007960808328230207,
  "word_mean_delta_nll_Tfit": 0.004340332856346961,
  "word_mean_delta_rank": 4.694319273056923,
  "word_improved_fraction_rank": -0.032258064516129004
}

## Interpretation

- Dense endpoints increase CDI NLL/rank relative to the chck82 parent; this is evidence-absent lexical drift rather than official AoA movement.
- If clean-preservation reduces the dense CDI drift while earlier analysis source probes show it retains a substantial source-responsive shift, the preservation objective is not equivalent to uniform step-size shrinkage on the private adapter.
- The readout uses a deterministic 96-word CDI subset for quick mechanism pressure; official endpoint columns and preservation admission standard raw AoA refits remain the score evidence.
