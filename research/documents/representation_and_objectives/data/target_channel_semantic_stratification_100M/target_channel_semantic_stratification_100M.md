# earlier analysis target-channel semantic stratification at 100M

JSON: `experiments/archive/representation_and_objectives/data/target_channel_semantic_stratification_100M/target_channel_semantic_stratification_100M.json`

Positive `drop_abs_minus_drop_copied_word` means removing source-absent compact-content labels increased local loss relative to removing matched copied labels.

## Focus table: main contrast

### doc_disjoint_all_accepted
| cell | n_events | pieces | delta | boot p025 | boot p975 | frac_gt0 |
|---|---:|---:|---:|---:|---:|---:|
| source_absent_content|all | 1012 | 1619 | 0.358682 | 0.26786 | 0.448348 | 1.0 |
| source_absent_content|relational_or_event_state | 97 | 103 | 0.634134 | 0.353557 | 0.905597 | 1.0 |
| source_absent_content|not_relational_or_event_state | 915 | 1516 | 0.339967 | 0.249329 | 0.434845 | 1.0 |
| source_absent_content|ordinary_nonrel_nonentity | 837 | 1344 | 0.340165 | 0.226812 | 0.444371 | 1.0 |
| source_absent_content|capitalized_or_number | 60 | 151 | 0.301736 | 0.07179 | 0.543707 | 0.995 |
| retained_content|all | 3199 | 6053 | -0.018743 | -0.049069 | 0.014507 | 0.138 |
| retained_content|relational_or_event_state | 139 | 166 | -0.096515 | -0.279565 | 0.079288 | 0.152 |
| retained_content|not_relational_or_event_state | 3060 | 5887 | -0.01655 | -0.046837 | 0.016358 | 0.161 |
| retained_content|ordinary_nonrel_nonentity | 2256 | 3988 | -0.023458 | -0.062999 | 0.016781 | 0.148 |
| retained_content|capitalized_or_number | 727 | 1796 | -0.007777 | -0.056774 | 0.040695 | 0.371 |
| function_other|all | 2246 | 2329 | -0.003824 | -0.039662 | 0.029894 | 0.413 |
| function_other|relational_or_event_state | 165 | 167 | 0.064004 | -0.079797 | 0.201441 | 0.812 |
| function_other|not_relational_or_event_state | 2081 | 2162 | -0.009064 | -0.042792 | 0.027786 | 0.32 |
| function_other|ordinary_nonrel_nonentity | 1894 | 1953 | -0.001815 | -0.036609 | 0.034257 | 0.463 |
| function_other|capitalized_or_number | 197 | 220 | -0.056369 | -0.188943 | 0.062451 | 0.176 |

Novelty×rel/event: `{"source_absent_rel_event": 0.634134, "source_absent_other": 0.339967, "retained_rel_event": -0.096515, "retained_other": -0.01655, "interaction": 0.374132, "meaning": "Positive means the source-absent-label local effect is especially concentrated in relational/event words beyond retained relational/event movement."}`

### doc_disjoint_quality
| cell | n_events | pieces | delta | boot p025 | boot p975 | frac_gt0 |
|---|---:|---:|---:|---:|---:|---:|
| source_absent_content|all | 71 | 110 | 0.501048 | 0.198636 | 0.805117 | 0.999 |
| source_absent_content|relational_or_event_state | 5 | 5 | 0.678095 | -0.07338 | 1.269304 | 0.971 |
| source_absent_content|not_relational_or_event_state | 66 | 105 | 0.492617 | 0.156491 | 0.819413 | 0.995 |
| source_absent_content|ordinary_nonrel_nonentity | 59 | 88 | 0.429809 | 0.050292 | 0.807219 | 0.988 |
| source_absent_content|capitalized_or_number | 7 | 17 | 0.817741 | 0.236272 | 1.418544 | 0.996 |
| retained_content|all | 949 | 1767 | -0.009735 | -0.054934 | 0.033673 | 0.326 |
| retained_content|relational_or_event_state | 57 | 67 | -0.106882 | -0.294453 | 0.072094 | 0.108 |
| retained_content|not_relational_or_event_state | 892 | 1700 | -0.005906 | -0.056678 | 0.042195 | 0.386 |
| retained_content|ordinary_nonrel_nonentity | 611 | 1051 | -0.002767 | -0.054071 | 0.047945 | 0.454 |
| retained_content|capitalized_or_number | 265 | 629 | -0.005723 | -0.091288 | 0.081865 | 0.468 |
| function_other|all | 579 | 598 | -0.003447 | -0.046935 | 0.042396 | 0.439 |
| function_other|relational_or_event_state | 32 | 33 | 0.108149 | -0.104185 | 0.349407 | 0.831 |
| function_other|not_relational_or_event_state | 547 | 565 | -0.009965 | -0.052647 | 0.033768 | 0.334 |
| function_other|ordinary_nonrel_nonentity | 512 | 524 | -0.007434 | -0.055932 | 0.041559 | 0.389 |
| function_other|capitalized_or_number | 36 | 43 | -0.013548 | -0.242456 | 0.149738 | 0.507 |

Novelty×rel/event: `{"source_absent_rel_event": 0.678095, "source_absent_other": 0.492617, "retained_rel_event": -0.106882, "retained_other": -0.005906, "interaction": 0.286454, "meaning": "Positive means the source-absent-label local effect is especially concentrated in relational/event words beyond retained relational/event movement."}`

### source_disjoint_quality
| cell | n_events | pieces | delta | boot p025 | boot p975 | frac_gt0 |
|---|---:|---:|---:|---:|---:|---:|
| source_absent_content|all | 974 | 1578 | 0.385856 | 0.30695 | 0.474849 | 1.0 |
| source_absent_content|relational_or_event_state | 159 | 175 | 0.504521 | 0.261209 | 0.756178 | 1.0 |
| source_absent_content|not_relational_or_event_state | 815 | 1403 | 0.371055 | 0.271775 | 0.47172 | 1.0 |
| source_absent_content|ordinary_nonrel_nonentity | 732 | 1149 | 0.453358 | 0.33881 | 0.567817 | 1.0 |
| source_absent_content|capitalized_or_number | 79 | 251 | 0.011054 | -0.200702 | 0.224757 | 0.545 |
| retained_content|all | 4096 | 7735 | -0.016759 | -0.037321 | 0.004643 | 0.064 |
| retained_content|relational_or_event_state | 234 | 281 | -0.023676 | -0.129509 | 0.081849 | 0.32 |
| retained_content|not_relational_or_event_state | 3862 | 7454 | -0.016499 | -0.037916 | 0.005871 | 0.082 |
| retained_content|ordinary_nonrel_nonentity | 2714 | 4778 | -0.029431 | -0.05779 | -0.001729 | 0.02 |
| retained_content|capitalized_or_number | 1083 | 2601 | -0.001914 | -0.038861 | 0.035623 | 0.477 |
| function_other|all | 4096 | 4244 | -0.016682 | -0.038857 | 0.005745 | 0.063 |
| function_other|relational_or_event_state | 312 | 319 | 0.049851 | -0.053216 | 0.156722 | 0.843 |
| function_other|not_relational_or_event_state | 3784 | 3925 | -0.022089 | -0.043205 | -0.000411 | 0.023 |
| function_other|ordinary_nonrel_nonentity | 3428 | 3526 | -0.026928 | -0.050601 | -0.004291 | 0.011 |
| function_other|capitalized_or_number | 376 | 420 | 0.027039 | -0.031594 | 0.087218 | 0.79 |

Novelty×rel/event: `{"source_absent_rel_event": 0.504521, "source_absent_other": 0.371055, "retained_rel_event": -0.023676, "retained_other": -0.016499, "interaction": 0.140643, "meaning": "Positive means the source-absent-label local effect is especially concentrated in relational/event words beyond retained relational/event movement."}`

### train_fixed_probe
| cell | n_events | pieces | delta | boot p025 | boot p975 | frac_gt0 |
|---|---:|---:|---:|---:|---:|---:|
| source_absent_content|all | 4096 | 6738 | 0.650196 | 0.605517 | 0.696873 | 1.0 |
| source_absent_content|relational_or_event_state | 533 | 582 | 0.959541 | 0.823267 | 1.098457 | 1.0 |
| source_absent_content|not_relational_or_event_state | 3563 | 6156 | 0.62095 | 0.574872 | 0.667632 | 1.0 |
| source_absent_content|ordinary_nonrel_nonentity | 3238 | 5248 | 0.686232 | 0.632773 | 0.733736 | 1.0 |
| source_absent_content|capitalized_or_number | 282 | 851 | 0.251698 | 0.129552 | 0.385942 | 1.0 |
| retained_content|all | 4096 | 7735 | -0.024666 | -0.052569 | 0.003516 | 0.048 |
| retained_content|relational_or_event_state | 189 | 237 | -0.114589 | -0.245864 | 0.022008 | 0.047 |
| retained_content|not_relational_or_event_state | 3907 | 7498 | -0.021824 | -0.04851 | 0.004469 | 0.054 |
| retained_content|ordinary_nonrel_nonentity | 2953 | 5210 | -0.019698 | -0.051583 | 0.012409 | 0.128 |
| retained_content|capitalized_or_number | 861 | 2172 | -0.029228 | -0.074986 | 0.020533 | 0.119 |
| function_other|all | 4096 | 4244 | 0.003044 | -0.019202 | 0.026437 | 0.591 |
| function_other|relational_or_event_state | 313 | 319 | -0.050233 | -0.155124 | 0.041249 | 0.141 |
| function_other|not_relational_or_event_state | 3783 | 3925 | 0.007374 | -0.016646 | 0.029447 | 0.725 |
| function_other|ordinary_nonrel_nonentity | 3443 | 3545 | 0.003203 | -0.021143 | 0.029885 | 0.593 |
| function_other|capitalized_or_number | 358 | 400 | 0.047767 | -0.008171 | 0.105 | 0.949 |

Novelty×rel/event: `{"source_absent_rel_event": 0.959541, "source_absent_other": 0.62095, "retained_rel_event": -0.114589, "retained_other": -0.021824, "interaction": 0.431356, "meaning": "Positive means the source-absent-label local effect is especially concentrated in relational/event words beyond retained relational/event movement."}`

