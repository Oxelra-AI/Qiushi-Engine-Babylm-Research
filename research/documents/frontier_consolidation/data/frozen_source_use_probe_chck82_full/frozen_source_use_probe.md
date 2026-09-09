# source conditioned ordering interaction synthesis frozen-model source-use probe (chck82)

Checkpoint: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M`
Total target records: 44995
Dry run: False
Elapsed: 137.0s

## Family summaries
| family | n_pairs | n_valid | mean I_f | se I_f | positive frac | mean NLL(ord|S) | mean NLL(scr|S) | mean NLL(ord) | mean NLL(scr) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact_vs_compact_scrambled | 3000 | 14999 | 2.747941 | 0.032087 | 0.762718 | 2.241608 | 8.122892 | 5.573490 | 8.706833 |
| prefix_fluent_vs_prefix_scrambled | 3000 | 14999 | 3.465261 | 0.033272 | 0.823322 | 0.373961 | 7.537016 | 4.557011 | 8.254805 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled | 3000 | 14997 | 3.595358 | 0.033677 | 0.829899 | 1.340585 | 7.948080 | 5.544859 | 8.556995 |

## Stratified I_f analysis

### By family
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact_vs_compact_scrambled | 14999 | 3000 | 2.7479 | 0.0321 | 0.7627 | 3.3319 | 0.5839 |
| prefix_fluent_vs_prefix_scrambled | 14999 | 3000 | 3.4653 | 0.0333 | 0.8233 | 4.1830 | 0.7178 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled | 14997 | 3000 | 3.5954 | 0.0337 | 0.8299 | 4.2043 | 0.6089 |

### By copy_zone
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| absent | 3403 | 2740 | 1.1457 | 0.0552 | 0.6277 | 1.1151 | -0.0305 |
| both_prefix_tail | 6968 | 4016 | 2.2849 | 0.0444 | 0.7714 | 2.0554 | -0.2294 |
| prefix_only | 23841 | 6989 | 3.5695 | 0.0318 | 0.8188 | 4.4869 | 0.9174 |
| tail_only | 10783 | 5250 | 3.9127 | 0.0422 | 0.8535 | 4.6998 | 0.7872 |

### By lex_class
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| capitalized_content | 4248 | 2881 | 4.8216 | 0.0777 | 0.8446 | 6.1786 | 1.3570 |
| content | 20463 | 6749 | 4.1327 | 0.0341 | 0.8416 | 5.4271 | 1.2945 |
| function | 18458 | 6668 | 1.9668 | 0.0251 | 0.7603 | 1.6835 | -0.2832 |
| number | 594 | 502 | 0.7559 | 0.0919 | 0.5606 | 2.4340 | 1.6781 |
| punct_or_symbol | 154 | 142 | 6.2108 | 0.3152 | 0.9416 | 6.2907 | 0.0799 |
| short_other | 1078 | 949 | 4.0391 | 0.1266 | 0.8479 | 4.6164 | 0.5773 |

### By source_match_bin
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3403 | 2740 | 1.1457 | 0.0552 | 0.6277 | 1.1151 | -0.0305 |
| 1_unique | 32588 | 6998 | 3.7855 | 0.0253 | 0.8353 | 4.7278 | 0.9422 |
| 2_3 | 8112 | 4444 | 2.2973 | 0.0418 | 0.7673 | 2.1145 | -0.1828 |
| 4plus | 892 | 544 | 1.3611 | 0.0900 | 0.7343 | 0.8427 | -0.5184 |

### By counterpart_visible
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| False | 3403 | 2740 | 1.1457 | 0.0552 | 0.6277 | 1.1151 | -0.0305 |
| True | 41592 | 7003 | 3.4433 | 0.0215 | 0.8198 | 4.1348 | 0.6915 |

### By complete_bpe_copy
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| False | 5465 | 2973 | 1.6494 | 0.0508 | 0.6750 | 1.8459 | 0.1965 |
| True | 39530 | 6999 | 3.4935 | 0.0226 | 0.8233 | 4.1913 | 0.6978 |

### By contentlike
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| False | 20284 | 6756 | 2.0737 | 0.0251 | 0.7605 | 1.8964 | -0.1773 |
| True | 24711 | 6934 | 4.2511 | 0.0316 | 0.8421 | 5.5563 | 1.3052 |

### By family_x_copy_zone
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact_vs_compact_scrambled/absent | 3251 | 2624 | 0.9084 | 0.0514 | 0.6130 | 0.8724 | -0.0360 |
| compact_vs_compact_scrambled/both_prefix_tail | 1622 | 1460 | 1.8338 | 0.0815 | 0.7226 | 1.6876 | -0.1462 |
| compact_vs_compact_scrambled/prefix_only | 5520 | 2977 | 3.3424 | 0.0593 | 0.8024 | 4.2933 | 0.9509 |
| compact_vs_compact_scrambled/tail_only | 4606 | 2945 | 3.6559 | 0.0620 | 0.8350 | 4.4946 | 0.8388 |
| prefix_fluent_vs_prefix_scrambled/absent | 80 | 78 | 6.1377 | 0.3878 | 0.9375 | 6.3438 | 0.2061 |
| prefix_fluent_vs_prefix_scrambled/both_prefix_tail | 3080 | 1872 | 2.6285 | 0.0651 | 0.8006 | 2.4295 | -0.1990 |
| prefix_fluent_vs_prefix_scrambled/prefix_only | 11839 | 3000 | 3.6649 | 0.0391 | 0.8284 | 4.6247 | 0.9598 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled/absent | 72 | 71 | 6.3134 | 0.5013 | 0.9444 | 6.2648 | -0.0486 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled/both_prefix_tail | 2266 | 1772 | 2.1407 | 0.0714 | 0.7665 | 1.8103 | -0.3304 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled/prefix_only | 6482 | 2995 | 3.5889 | 0.0558 | 0.8152 | 4.4003 | 0.8114 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled/tail_only | 6177 | 2996 | 4.1041 | 0.0539 | 0.8672 | 4.8528 | 0.7487 |

## Interpretation
I_f = [NLL(V_scr|S) - NLL(V_ord|S)] - [NLL(V_scr) - NLL(V_ord)]

Positive I_f: ordered structure specifically helps source-conditioned reconstruction beyond generic fluency preference.
Near-zero I_f: ordering helps equally with and without source; generic fluency, not source-use structure.
Negative I_f: ordering helps *less* when source is present; source makes ordering irrelevant.

Records CSV: `experiments/archive/frontier_consolidation/data/frozen_source_use_probe_chck82_full/source_use_probe_records.csv`
JSON: `experiments/archive/frontier_consolidation/data/frozen_source_use_probe_chck82_full/frozen_source_use_probe.json`
