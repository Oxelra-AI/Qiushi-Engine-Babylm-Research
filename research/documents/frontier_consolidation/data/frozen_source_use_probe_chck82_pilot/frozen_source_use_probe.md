# source conditioned ordering interaction synthesis frozen-model source-use probe (chck82)

Checkpoint: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M`
Total target records: 1500
Dry run: False
Elapsed: 28.2s

## Family summaries
| family | n_pairs | n_valid | mean I_f | se I_f | positive frac | mean NLL(ord|S) | mean NLL(scr|S) | mean NLL(ord) | mean NLL(scr) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact_vs_compact_scrambled | 100 | 500 | 2.692654 | 0.189759 | 0.760000 | 2.254732 | 8.187631 | 5.581732 | 8.821978 |
| prefix_fluent_vs_prefix_scrambled | 100 | 500 | 3.310705 | 0.160712 | 0.832000 | 0.369098 | 7.526142 | 4.453467 | 8.299807 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled | 100 | 500 | 3.400283 | 0.195923 | 0.784000 | 1.322350 | 7.749482 | 5.517557 | 8.544405 |

## Stratified I_f analysis

### By family
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact_vs_compact_scrambled | 500 | 100 | 2.6927 | 0.1898 | 0.7600 | 3.3270 | 0.6343 |
| prefix_fluent_vs_prefix_scrambled | 500 | 100 | 3.3107 | 0.1607 | 0.8320 | 4.0844 | 0.7737 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled | 500 | 100 | 3.4003 | 0.1959 | 0.7840 | 4.1952 | 0.7949 |

### By copy_zone
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| absent | 107 | 87 | 1.1133 | 0.3557 | 0.5701 | 0.9039 | -0.2094 |
| both_prefix_tail | 232 | 168 | 2.2138 | 0.2069 | 0.7845 | 1.9968 | -0.2171 |
| prefix_only | 803 | 296 | 3.3314 | 0.1514 | 0.7945 | 4.4130 | 1.0816 |
| tail_only | 358 | 193 | 3.8938 | 0.2035 | 0.8575 | 4.7477 | 0.8539 |

### By lex_class
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| capitalized_content | 126 | 99 | 5.0113 | 0.4111 | 0.8413 | 5.8248 | 0.8135 |
| content | 715 | 282 | 3.8978 | 0.1712 | 0.8238 | 5.3748 | 1.4769 |
| function | 603 | 274 | 1.8927 | 0.1258 | 0.7479 | 1.6829 | -0.2098 |
| number | 17 | 16 | 0.4659 | 0.4993 | 0.5294 | 2.2691 | 1.8032 |
| punct_or_symbol | 2 | 2 | 9.0413 | 1.6046 | 1.0000 | 7.1563 | -1.8851 |
| short_other | 37 | 36 | 3.1394 | 0.7189 | 0.8378 | 4.2906 | 1.1513 |

### By source_match_bin
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 107 | 87 | 1.1133 | 0.3557 | 0.5701 | 0.9039 | -0.2094 |
| 1_unique | 1083 | 296 | 3.6193 | 0.1305 | 0.8209 | 4.7234 | 1.1040 |
| 2_3 | 286 | 187 | 2.2195 | 0.1977 | 0.7657 | 2.0228 | -0.1966 |
| 4plus | 24 | 23 | 1.1749 | 0.3093 | 0.7917 | 0.5266 | -0.6483 |

### By counterpart_visible
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| False | 107 | 87 | 1.1133 | 0.3557 | 0.5701 | 0.9039 | -0.2094 |
| True | 1393 | 297 | 3.2898 | 0.1083 | 0.8090 | 4.0966 | 0.8068 |

### By complete_bpe_copy
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| False | 178 | 96 | 1.6178 | 0.3189 | 0.6404 | 1.8385 | 0.2206 |
| True | 1322 | 296 | 3.3388 | 0.1122 | 0.8124 | 4.1422 | 0.8035 |

### By contentlike
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| False | 659 | 280 | 1.9476 | 0.1291 | 0.7481 | 1.8610 | -0.0866 |
| True | 841 | 290 | 4.0646 | 0.1560 | 0.8264 | 5.4422 | 1.3775 |

### By family_x_copy_zone
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact_vs_compact_scrambled/absent | 105 | 85 | 0.9623 | 0.3377 | 0.5619 | 0.7848 | -0.1775 |
| compact_vs_compact_scrambled/both_prefix_tail | 55 | 51 | 1.9267 | 0.3748 | 0.7455 | 1.1515 | -0.7752 |
| compact_vs_compact_scrambled/prefix_only | 186 | 99 | 2.9931 | 0.2703 | 0.7903 | 4.0624 | 1.0694 |
| compact_vs_compact_scrambled/tail_only | 154 | 97 | 3.7831 | 0.3213 | 0.8636 | 4.9490 | 1.1659 |
| prefix_fluent_vs_prefix_scrambled/absent | 1 | 1 | 7.4367 | nan | 1.0000 | 3.3053 | -4.1314 |
| prefix_fluent_vs_prefix_scrambled/both_prefix_tail | 106 | 63 | 2.4958 | 0.3233 | 0.8302 | 2.4782 | -0.0175 |
| prefix_fluent_vs_prefix_scrambled/prefix_only | 393 | 100 | 3.5200 | 0.2048 | 0.8321 | 4.5196 | 0.9995 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled/absent | 1 | 1 | 10.6459 | nan | 1.0000 | 11.0072 | 0.3613 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled/both_prefix_tail | 71 | 56 | 2.0154 | 0.3720 | 0.7465 | 1.9327 | -0.0826 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled/prefix_only | 224 | 100 | 3.2814 | 0.3021 | 0.7321 | 4.5172 | 1.2357 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled/tail_only | 204 | 99 | 3.9773 | 0.2543 | 0.8529 | 4.5957 | 0.6184 |

## Interpretation
I_f = [NLL(V_scr|S) - NLL(V_ord|S)] - [NLL(V_scr) - NLL(V_ord)]

Positive I_f: ordered structure specifically helps source-conditioned reconstruction beyond generic fluency preference.
Near-zero I_f: ordering helps equally with and without source; generic fluency, not source-use structure.
Negative I_f: ordering helps *less* when source is present; source makes ordering irrelevant.

Records CSV: `experiments/archive/frontier_consolidation/data/frozen_source_use_probe_chck82_pilot/source_use_probe_records.csv`
JSON: `experiments/archive/frontier_consolidation/data/frozen_source_use_probe_chck82_pilot/frozen_source_use_probe.json`
