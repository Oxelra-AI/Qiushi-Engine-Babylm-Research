# source conditioned ordering interaction synthesis frozen-model source-use probe (chck100)

Checkpoint: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M`
Total target records: 1500
Dry run: False
Elapsed: 22.0s

## Family summaries
| family | n_pairs | n_valid | mean I_f | se I_f | positive frac | mean NLL(ord|S) | mean NLL(scr|S) | mean NLL(ord) | mean NLL(scr) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact_vs_compact_scrambled | 100 | 500 | 2.691495 | 0.188423 | 0.742000 | 2.222316 | 8.180242 | 5.561303 | 8.827734 |
| prefix_fluent_vs_prefix_scrambled | 100 | 500 | 3.318416 | 0.163228 | 0.824000 | 0.318189 | 7.498153 | 4.422880 | 8.284427 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled | 100 | 500 | 3.379952 | 0.194645 | 0.788000 | 1.299935 | 7.755050 | 5.485884 | 8.561047 |

## Stratified I_f analysis

### By family
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact_vs_compact_scrambled | 500 | 100 | 2.6915 | 0.1884 | 0.7420 | 3.3390 | 0.6475 |
| prefix_fluent_vs_prefix_scrambled | 500 | 100 | 3.3184 | 0.1632 | 0.8240 | 4.1047 | 0.7863 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled | 500 | 100 | 3.3800 | 0.1946 | 0.7880 | 4.1859 | 0.8060 |

### By copy_zone
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| absent | 107 | 87 | 1.1877 | 0.3656 | 0.5607 | 0.9600 | -0.2277 |
| both_prefix_tail | 232 | 168 | 2.2352 | 0.2097 | 0.7629 | 1.9894 | -0.2458 |
| prefix_only | 803 | 296 | 3.3284 | 0.1534 | 0.7833 | 4.4379 | 1.1095 |
| tail_only | 358 | 193 | 3.8453 | 0.1994 | 0.8687 | 4.7121 | 0.8668 |

### By lex_class
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| capitalized_content | 126 | 99 | 4.9308 | 0.4051 | 0.8413 | 5.7676 | 0.8368 |
| content | 715 | 282 | 3.8789 | 0.1720 | 0.8168 | 5.3762 | 1.4973 |
| function | 603 | 274 | 1.9077 | 0.1295 | 0.7413 | 1.7023 | -0.2054 |
| number | 17 | 16 | 0.4512 | 0.4349 | 0.5294 | 2.2508 | 1.7997 |
| punct_or_symbol | 2 | 2 | 9.5857 | 1.1646 | 1.0000 | 7.5819 | -2.0038 |
| short_other | 37 | 36 | 3.3263 | 0.7006 | 0.7838 | 4.4378 | 1.1115 |

### By source_match_bin
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 107 | 87 | 1.1877 | 0.3656 | 0.5607 | 0.9600 | -0.2277 |
| 1_unique | 1083 | 296 | 3.6040 | 0.1308 | 0.8163 | 4.7326 | 1.1286 |
| 2_3 | 286 | 187 | 2.2270 | 0.1998 | 0.7483 | 2.0040 | -0.2230 |
| 4plus | 24 | 23 | 1.1577 | 0.3572 | 0.7917 | 0.5655 | -0.5922 |

### By counterpart_visible
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| False | 107 | 87 | 1.1877 | 0.3656 | 0.5607 | 0.9600 | -0.2277 |
| True | 1393 | 297 | 3.2791 | 0.1085 | 0.8019 | 4.1006 | 0.8214 |

### By complete_bpe_copy
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| False | 178 | 96 | 1.6636 | 0.3246 | 0.6348 | 1.8598 | 0.1962 |
| True | 1322 | 296 | 3.3274 | 0.1130 | 0.8048 | 4.1481 | 0.8207 |

### By contentlike
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| False | 659 | 280 | 1.9731 | 0.1332 | 0.7390 | 1.8879 | -0.0852 |
| True | 841 | 290 | 4.0365 | 0.1553 | 0.8205 | 5.4348 | 1.3984 |

### By family_x_copy_zone
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact_vs_compact_scrambled/absent | 105 | 85 | 1.0278 | 0.3461 | 0.5524 | 0.8339 | -0.1939 |
| compact_vs_compact_scrambled/both_prefix_tail | 55 | 51 | 1.8926 | 0.3770 | 0.7455 | 1.1043 | -0.7883 |
| compact_vs_compact_scrambled/prefix_only | 186 | 99 | 3.0038 | 0.2845 | 0.7527 | 4.1067 | 1.1029 |
| compact_vs_compact_scrambled/tail_only | 154 | 97 | 3.7339 | 0.3141 | 0.8571 | 4.9178 | 1.1839 |
| prefix_fluent_vs_prefix_scrambled/absent | 1 | 1 | 8.4211 | nan | 1.0000 | 3.9350 | -4.4861 |
| prefix_fluent_vs_prefix_scrambled/both_prefix_tail | 106 | 63 | 2.5453 | 0.3291 | 0.8113 | 2.5028 | -0.0424 |
| prefix_fluent_vs_prefix_scrambled/prefix_only | 393 | 100 | 3.5140 | 0.2080 | 0.8270 | 4.5372 | 1.0232 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled/absent | 1 | 1 | 10.7503 | nan | 1.0000 | 11.2288 | 0.4785 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled/both_prefix_tail | 71 | 56 | 2.0377 | 0.3768 | 0.7042 | 1.9084 | -0.1294 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled/prefix_only | 224 | 100 | 3.2722 | 0.2952 | 0.7321 | 4.5387 | 1.2665 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled/tail_only | 204 | 99 | 3.9293 | 0.2478 | 0.8775 | 4.5568 | 0.6275 |

## Interpretation
I_f = [NLL(V_scr|S) - NLL(V_ord|S)] - [NLL(V_scr) - NLL(V_ord)]

Positive I_f: ordered structure specifically helps source-conditioned reconstruction beyond generic fluency preference.
Near-zero I_f: ordering helps equally with and without source; generic fluency, not source-use structure.
Negative I_f: ordering helps *less* when source is present; source makes ordering irrelevant.

Records CSV: `experiments/archive/frontier_consolidation/data/frozen_source_use_probe_chck100_pilot/source_use_probe_records.csv`
JSON: `experiments/archive/frontier_consolidation/data/frozen_source_use_probe_chck100_pilot/frozen_source_use_probe.json`
