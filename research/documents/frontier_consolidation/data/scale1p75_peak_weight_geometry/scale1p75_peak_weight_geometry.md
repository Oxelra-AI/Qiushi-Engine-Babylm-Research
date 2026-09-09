# chck82 reproducibility and peak characterization scale1.75 peak weight geometry

Status: `SCALE1P75_PEAK_WEIGHT_GEOMETRY`

## Cheap7 window

| checkpoint | cheap7 | adapter_rms | adapter/stock norm | prev Δ total | prev Δ adapter frac | Δ-cos total | Δ-cos adapter |
|---|---:|---:|---:|---:|---:|---:|---:|
| chck_77M | 43.28214285714286 | 0.06617267 | 0.286516 |  |  |  |  |
| chck_78M | 43.70214285714286 | 0.06616984 | 0.286466 | 2.977626 | 0.167091 |  |  |
| chck_79M | 43.57857142857143 | 0.06616675 | 0.286413 | 2.757009 | 0.166390 | 0.197837 | 0.182190 |
| chck_80M | 43.81214285714286 | 0.06616411 | 0.286372 | 2.596184 | 0.166710 | 0.202605 | 0.185074 |
| chck_81M | 43.64928571428572 | 0.06616083 | 0.286336 | 2.291127 | 0.167315 | 0.201272 | 0.183750 |
| chck_82M | 43.96 | 0.06615896 | 0.286295 | 2.070420 | 0.166906 | 0.205538 | 0.192444 |
| chck_83M | 43.807857142857145 | 0.06615717 | 0.286274 | 1.882903 | 0.167012 | 0.209347 | 0.191971 |

## Local peak summary

```json
{
  "window": [
    "chck_77M",
    "chck_78M",
    "chck_79M",
    "chck_80M",
    "chck_81M",
    "chck_82M",
    "chck_83M"
  ],
  "cheap7": {
    "chck_77M": 43.28214285714286,
    "chck_78M": 43.70214285714286,
    "chck_79M": 43.57857142857143,
    "chck_80M": 43.81214285714286,
    "chck_81M": 43.64928571428572,
    "chck_82M": 43.96,
    "chck_83M": 43.807857142857145
  },
  "peak_by_cheap7": "chck_82M",
  "cheap7_mean": 43.6845918367347,
  "cheap7_range": 0.6778571428571425,
  "peak_minus_neighbor_mean": 0.2314285714285731,
  "weight_signal_rows": [
    {
      "checkpoint": "chck_77M",
      "cheap7": 43.28214285714286,
      "adapter_rms": 0.0661726679263108,
      "adapter_to_stock_norm_ratio": 0.2865164516224629,
      "prev_delta_total_norm": null,
      "prev_delta_stock_norm": null,
      "prev_delta_adapter_norm": null,
      "prev_delta_adapter_norm_fraction": null,
      "cosine_with_previous_delta_total": null,
      "cosine_with_previous_delta_stock": null,
      "cosine_with_previous_delta_adapter": null
    },
    {
      "checkpoint": "chck_78M",
      "cheap7": 43.70214285714286,
      "adapter_rms": 0.06616983738364275,
      "adapter_to_stock_norm_ratio": 0.2864657034168059,
      "prev_delta_total_norm": 2.977625687141779,
      "prev_delta_stock_norm": 2.935764574531231,
      "prev_delta_adapter_norm": 0.49753501952527374,
      "prev_delta_adapter_norm_fraction": 0.1670911900289446,
      "cosine_with_previous_delta_total": null,
      "cosine_with_previous_delta_stock": null,
      "cosine_with_previous_delta_adapter": null
    },
    {
      "checkpoint": "chck_79M",
      "cheap7": 43.57857142857143,
      "adapter_rms": 0.06616674978581248,
      "adapter_to_stock_norm_ratio": 0.2864129868491973,
      "prev_delta_total_norm": 2.7570087629920987,
      "prev_delta_stock_norm": 2.718575960038073,
      "prev_delta_adapter_norm": 0.45873965243729886,
      "prev_delta_adapter_norm_fraction": 0.16639034978599143,
      "cosine_with_previous_delta_total": 0.1978369030516001,
      "cosine_with_previous_delta_stock": 0.19828442522078124,
      "cosine_with_previous_delta_adapter": 0.1821896744451174
    },
    {
      "checkpoint": "chck_80M",
      "cheap7": 43.81214285714286,
      "adapter_rms": 0.066164113314976,
      "adapter_to_stock_norm_ratio": 0.28637176252082563,
      "prev_delta_total_norm": 2.5961844866169326,
      "prev_delta_stock_norm": 2.5598535340885236,
      "prev_delta_adapter_norm": 0.4328091641415686,
      "prev_delta_adapter_norm_fraction": 0.16670971048962657,
      "cosine_with_previous_delta_total": 0.20260509039608487,
      "cosine_with_previous_delta_stock": 0.20310526911935456,
      "cosine_with_previous_delta_adapter": 0.1850739735758263
    },
    {
      "checkpoint": "chck_81M",
      "cheap7": 43.64928571428572,
      "adapter_rms": 0.06616082603017372,
      "adapter_to_stock_norm_ratio": 0.28633607852847326,
      "prev_delta_total_norm": 2.2911268542045313,
      "prev_delta_stock_norm": 2.2588299562481104,
      "prev_delta_adapter_norm": 0.3833399154970858,
      "prev_delta_adapter_norm_fraction": 0.16731501129830703,
      "cosine_with_previous_delta_total": 0.20127184608688323,
      "cosine_with_previous_delta_stock": 0.20177463473490578,
      "cosine_with_previous_delta_adapter": 0.18375040088436265
    },
    {
      "checkpoint": "chck_82M",
      "cheap7": 43.96,
      "adapter_rms": 0.06615895717314245,
      "adapter_to_stock_norm_ratio": 0.28629470990133804,
      "prev_delta_total_norm": 2.0704201791520314,
      "prev_delta_stock_norm": 2.0413781286993946,
      "prev_delta_adapter_norm": 0.34556483314667197,
      "prev_delta_adapter_norm_fraction": 0.1669056535607196,
      "cosine_with_previous_delta_total": 0.20553829899547987,
      "cosine_with_previous_delta_stock": 0.20591447859625525,
      "cosine_with_previous_delta_adapter": 0.19244444138564387
    },
    {
      "checkpoint": "chck_83M",
      "cheap7": 43.807857142857145,
      "adapter_rms": 0.06615716731917727,
      "adapter_to_stock_norm_ratio": 0.2862739706051811,
      "prev_delta_total_norm": 1.8829032293649364,
      "prev_delta_stock_norm": 1.8564577249926646,
      "prev_delta_adapter_norm": 0.31446666988405236,
      "prev_delta_adapter_norm_fraction": 0.16701159410625438,
      "cosine_with_previous_delta_total": 0.20934743601162437,
      "cosine_with_previous_delta_stock": 0.20984569397486347,
      "cosine_with_previous_delta_adapter": 0.19197112724589857
    }
  ]
}
```

This script reports internal weight geometry only. A useful early-stop signal would need to single out chck_82M, or at least warn against 83M-100M degradation, without using official evaluation labels. If the recorded weight rows are monotone/smooth while cheap7 oscillates, then current saved internal traces do not explain the peak.

JSON: `experiments/archive/frontier_consolidation/data/scale1p75_peak_weight_geometry/scale1p75_peak_weight_geometry.json`
