# span matcher review and next span discovery review of span matcher construction and design SpanMatcher gauge result

## Reconstructed central metrics from saved predictions

| run | state rows | comp rows | malformed choices | graph_same | graph_margin | direct_same | unchanged | hh_closure | mixed_acc | mixed_margin | mixed pred-true |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| learned_shared_bsplus | 2048 | 640 | 0 | 1.000 | 18.214 | 1.000 | 1.000 | 1.000 | 0.500 | -0.042 | 0.500 |
| learned_shared_bsminus | 2048 | 640 | 0 | 0.000 | -17.449 | 0.000 | 1.000 | 1.000 | 0.500 | -0.399 | 0.500 |
| oracle_shared_bsplus | 2048 | 640 | 0 | 1.000 | 17.986 | 1.000 | 1.000 | 1.000 | 0.750 | 0.001 | 0.500 |
| oracle_shared_bsminus | 2048 | 640 | 0 | 0.000 | -17.449 | 0.000 | 1.000 | 1.000 | 0.500 | -0.304 | 0.500 |
| oracle_untied_bsplus_partial | 2048 | 640 | 0 | 1.000 | 5.284 | 1.000 | 1.000 | 1.000 | 0.250 | -1.509 | 0.500 |
| learned_untied_bsplus | 2048 | 640 | 0 | 0.969 | 5.810 | 1.000 | 1.000 | 1.000 | 0.500 | -0.023 | 0.500 |
| learned_untied_bsminus | 2048 | 640 | 0 | 0.469 | -1.029 | 0.000 | 1.000 | 1.000 | 0.500 | -0.023 | 0.500 |

## Row-paired sign checks

| comparison | paired_n | opposite sign | same sign | zero | same prediction | a_acc | b_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| learned_graph_changed | 256 | 1.000 | 0 | 0 | 0.000 | 1.000 | 0.000 |
| oracle_graph_changed | 256 | 1.000 | 0 | 0 | 0.000 | 1.000 | 0.000 |
| learned_direct_changed | 256 | 1.000 | 0 | 0 | 0.000 | 1.000 | 0.000 |
| oracle_direct_changed | 256 | 1.000 | 0 | 0 | 0.000 | 1.000 | 0.000 |
| learned_unchanged | 512 | 0.000 | 512 | 0 | 1.000 | 1.000 | 1.000 |
| oracle_unchanged | 512 | 0.000 | 512 | 0 | 1.000 | 1.000 | 1.000 |
| learned_all_changed | 512 | 1.000 | 0 | 0 | 0.000 | 1.000 | 0.000 |
| oracle_all_changed | 512 | 1.000 | 0 | 0 | 0.000 | 1.000 | 0.000 |

## Learned versus oracle sign alignment

| comparison | paired_n | same sign fraction | opposite sign n | zero |
|---|---:|---:|---:|---:|
| bsplus_graph_changed | 256 | 1.000 | 0 | 0 |
| bsminus_graph_changed | 256 | 1.000 | 0 | 0 |
| bsplus_direct_changed | 256 | 1.000 | 0 | 0 |
| bsminus_direct_changed | 256 | 1.000 | 0 | 0 |
| bsplus_unchanged | 512 | 1.000 | 0 | 0 |
| bsminus_unchanged | 512 | 1.000 | 0 | 0 |
| bsplus_all_changed | 512 | 1.000 | 0 | 0 |
| bsminus_all_changed | 512 | 1.000 | 0 | 0 |

## Mixed-comparison breakdown

### learned_shared_bsplus

By suite:

| suite | n | acc | pred_true | label_true | signed_margin | abs_logit | product_sign_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldheld_unseen_edge_closure | 128 | 1.000 | 0.500 | 0.500 | 13.794 | 13.794 | 1.000 |
| mixed_held_seen_orientation | 512 | 0.500 | 0.500 | 0.500 | -0.042 | 0.629 | 0.500 |

Mixed relation pairs:

| suite/rels | n | acc | pred_true | label_true | signed_margin | abs_logit | product_sign_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| ('mixed_held_seen_orientation', 'h0_dax', 's_give') | 32 | 0.500 | 0.500 | 0.500 | -0.245 | 0.677 | 0.500 |
| ('mixed_held_seen_orientation', 'h0_dax', 's_receive') | 32 | 0.500 | 0.500 | 0.500 | 0.117 | 0.550 | 0.500 |
| ('mixed_held_seen_orientation', 'h1_mep', 's_give') | 32 | 0.500 | 0.500 | 0.500 | -0.249 | 0.683 | 0.500 |
| ('mixed_held_seen_orientation', 'h1_mep', 's_receive') | 32 | 0.500 | 0.500 | 0.500 | 0.210 | 0.602 | 0.500 |
| ('mixed_held_seen_orientation', 'h2_norp', 's_give') | 32 | 0.500 | 0.500 | 0.500 | -0.249 | 0.683 | 0.500 |
| ('mixed_held_seen_orientation', 'h2_norp', 's_receive') | 32 | 0.500 | 0.500 | 0.500 | 0.210 | 0.602 | 0.500 |
| ('mixed_held_seen_orientation', 'h3_ziv', 's_give') | 32 | 0.500 | 0.500 | 0.500 | -0.245 | 0.677 | 0.500 |
| ('mixed_held_seen_orientation', 'h3_ziv', 's_receive') | 32 | 0.500 | 0.500 | 0.500 | 0.114 | 0.560 | 0.500 |
| ('mixed_held_seen_orientation', 's_give', 'h0_dax') | 32 | 0.500 | 0.500 | 0.500 | -0.246 | 0.680 | 0.500 |
| ('mixed_held_seen_orientation', 's_give', 'h1_mep') | 32 | 0.500 | 0.500 | 0.500 | -0.247 | 0.680 | 0.500 |
| ('mixed_held_seen_orientation', 's_give', 'h2_norp') | 32 | 0.500 | 0.500 | 0.500 | -0.246 | 0.680 | 0.500 |
| ('mixed_held_seen_orientation', 's_give', 'h3_ziv') | 32 | 0.500 | 0.500 | 0.500 | -0.247 | 0.680 | 0.500 |
| ('mixed_held_seen_orientation', 's_receive', 'h0_dax') | 32 | 0.500 | 0.500 | 0.500 | 0.160 | 0.569 | 0.500 |
| ('mixed_held_seen_orientation', 's_receive', 'h1_mep') | 32 | 0.500 | 0.500 | 0.500 | 0.167 | 0.583 | 0.500 |
| ('mixed_held_seen_orientation', 's_receive', 'h2_norp') | 32 | 0.500 | 0.500 | 0.500 | 0.157 | 0.579 | 0.500 |
| ('mixed_held_seen_orientation', 's_receive', 'h3_ziv') | 32 | 0.500 | 0.500 | 0.500 | 0.167 | 0.583 | 0.500 |

### learned_shared_bsminus

By suite:

| suite | n | acc | pred_true | label_true | signed_margin | abs_logit | product_sign_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldheld_unseen_edge_closure | 128 | 1.000 | 0.500 | 0.500 | 13.615 | 13.615 | 1.000 |
| mixed_held_seen_orientation | 512 | 0.500 | 0.500 | 0.500 | -0.399 | 7.875 | 0.500 |

Mixed relation pairs:

| suite/rels | n | acc | pred_true | label_true | signed_margin | abs_logit | product_sign_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| ('mixed_held_seen_orientation', 'h0_dax', 's_give') | 32 | 0.500 | 0.500 | 0.500 | 5.366 | 8.442 | 0.500 |
| ('mixed_held_seen_orientation', 'h0_dax', 's_receive') | 32 | 0.500 | 0.500 | 0.500 | -5.937 | 7.066 | 0.500 |
| ('mixed_held_seen_orientation', 'h1_mep', 's_give') | 32 | 0.500 | 0.500 | 0.500 | 5.369 | 8.439 | 0.500 |
| ('mixed_held_seen_orientation', 'h1_mep', 's_receive') | 32 | 0.500 | 0.500 | 0.500 | -5.938 | 7.053 | 0.500 |
| ('mixed_held_seen_orientation', 'h2_norp', 's_give') | 32 | 0.500 | 0.500 | 0.500 | 5.369 | 8.439 | 0.500 |
| ('mixed_held_seen_orientation', 'h2_norp', 's_receive') | 32 | 0.500 | 0.500 | 0.500 | -6.312 | 7.454 | 0.500 |
| ('mixed_held_seen_orientation', 'h3_ziv', 's_give') | 32 | 0.500 | 0.500 | 0.500 | 5.366 | 8.442 | 0.500 |
| ('mixed_held_seen_orientation', 'h3_ziv', 's_receive') | 32 | 0.500 | 0.500 | 0.500 | -6.320 | 7.489 | 0.500 |
| ('mixed_held_seen_orientation', 's_give', 'h0_dax') | 32 | 0.500 | 0.500 | 0.500 | 5.367 | 8.441 | 0.500 |
| ('mixed_held_seen_orientation', 's_give', 'h1_mep') | 32 | 0.500 | 0.500 | 0.500 | 5.368 | 8.441 | 0.500 |
| ('mixed_held_seen_orientation', 's_give', 'h2_norp') | 32 | 0.500 | 0.500 | 0.500 | 5.367 | 8.441 | 0.500 |
| ('mixed_held_seen_orientation', 's_give', 'h3_ziv') | 32 | 0.500 | 0.500 | 0.500 | 5.368 | 8.441 | 0.500 |
| ('mixed_held_seen_orientation', 's_receive', 'h0_dax') | 32 | 0.500 | 0.500 | 0.500 | -5.882 | 6.999 | 0.500 |
| ('mixed_held_seen_orientation', 's_receive', 'h1_mep') | 32 | 0.500 | 0.500 | 0.500 | -6.306 | 7.460 | 0.500 |
| ('mixed_held_seen_orientation', 's_receive', 'h2_norp') | 32 | 0.500 | 0.500 | 0.500 | -6.326 | 7.483 | 0.500 |
| ('mixed_held_seen_orientation', 's_receive', 'h3_ziv') | 32 | 0.500 | 0.500 | 0.500 | -6.306 | 7.460 | 0.500 |

### oracle_shared_bsplus

By suite:

| suite | n | acc | pred_true | label_true | signed_margin | abs_logit | product_sign_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldheld_unseen_edge_closure | 128 | 1.000 | 0.500 | 0.500 | 13.809 | 13.809 | 1.000 |
| mixed_held_seen_orientation | 512 | 0.750 | 0.500 | 0.500 | 0.001 | 0.034 | 0.750 |

Mixed relation pairs:

| suite/rels | n | acc | pred_true | label_true | signed_margin | abs_logit | product_sign_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| ('mixed_held_seen_orientation', 'h0_dax', 's_give') | 32 | 1.000 | 0.500 | 0.500 | 0.035 | 0.035 | 1.000 |
| ('mixed_held_seen_orientation', 'h0_dax', 's_receive') | 32 | 0.500 | 0.500 | 0.500 | -0.033 | 0.034 | 0.500 |
| ('mixed_held_seen_orientation', 'h1_mep', 's_give') | 32 | 1.000 | 0.500 | 0.500 | 0.035 | 0.035 | 1.000 |
| ('mixed_held_seen_orientation', 'h1_mep', 's_receive') | 32 | 0.500 | 0.500 | 0.500 | -0.033 | 0.034 | 0.500 |
| ('mixed_held_seen_orientation', 'h2_norp', 's_give') | 32 | 1.000 | 0.500 | 0.500 | 0.035 | 0.035 | 1.000 |
| ('mixed_held_seen_orientation', 'h2_norp', 's_receive') | 32 | 0.500 | 0.500 | 0.500 | -0.033 | 0.034 | 0.500 |
| ('mixed_held_seen_orientation', 'h3_ziv', 's_give') | 32 | 1.000 | 0.500 | 0.500 | 0.035 | 0.035 | 1.000 |
| ('mixed_held_seen_orientation', 'h3_ziv', 's_receive') | 32 | 0.500 | 0.500 | 0.500 | -0.033 | 0.034 | 0.500 |
| ('mixed_held_seen_orientation', 's_give', 'h0_dax') | 32 | 1.000 | 0.500 | 0.500 | 0.035 | 0.035 | 1.000 |
| ('mixed_held_seen_orientation', 's_give', 'h1_mep') | 32 | 1.000 | 0.500 | 0.500 | 0.035 | 0.035 | 1.000 |
| ('mixed_held_seen_orientation', 's_give', 'h2_norp') | 32 | 1.000 | 0.500 | 0.500 | 0.035 | 0.035 | 1.000 |
| ('mixed_held_seen_orientation', 's_give', 'h3_ziv') | 32 | 1.000 | 0.500 | 0.500 | 0.035 | 0.035 | 1.000 |
| ('mixed_held_seen_orientation', 's_receive', 'h0_dax') | 32 | 0.500 | 0.500 | 0.500 | -0.033 | 0.034 | 0.500 |
| ('mixed_held_seen_orientation', 's_receive', 'h1_mep') | 32 | 0.500 | 0.500 | 0.500 | -0.033 | 0.034 | 0.500 |
| ('mixed_held_seen_orientation', 's_receive', 'h2_norp') | 32 | 0.500 | 0.500 | 0.500 | -0.033 | 0.034 | 0.500 |
| ('mixed_held_seen_orientation', 's_receive', 'h3_ziv') | 32 | 0.500 | 0.500 | 0.500 | -0.033 | 0.034 | 0.500 |

### oracle_shared_bsminus

By suite:

| suite | n | acc | pred_true | label_true | signed_margin | abs_logit | product_sign_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldheld_unseen_edge_closure | 128 | 1.000 | 0.500 | 0.500 | 13.473 | 13.473 | 1.000 |
| mixed_held_seen_orientation | 512 | 0.500 | 0.500 | 0.500 | -0.304 | 1.528 | 0.500 |

Mixed relation pairs:

| suite/rels | n | acc | pred_true | label_true | signed_margin | abs_logit | product_sign_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| ('mixed_held_seen_orientation', 'h0_dax', 's_give') | 32 | 0.500 | 0.500 | 0.500 | 0.691 | 1.225 | 0.500 |
| ('mixed_held_seen_orientation', 'h0_dax', 's_receive') | 32 | 0.500 | 0.500 | 0.500 | -1.299 | 1.832 | 0.500 |
| ('mixed_held_seen_orientation', 'h1_mep', 's_give') | 32 | 0.500 | 0.500 | 0.500 | 0.691 | 1.225 | 0.500 |
| ('mixed_held_seen_orientation', 'h1_mep', 's_receive') | 32 | 0.500 | 0.500 | 0.500 | -1.299 | 1.832 | 0.500 |
| ('mixed_held_seen_orientation', 'h2_norp', 's_give') | 32 | 0.500 | 0.500 | 0.500 | 0.691 | 1.225 | 0.500 |
| ('mixed_held_seen_orientation', 'h2_norp', 's_receive') | 32 | 0.500 | 0.500 | 0.500 | -1.299 | 1.832 | 0.500 |
| ('mixed_held_seen_orientation', 'h3_ziv', 's_give') | 32 | 0.500 | 0.500 | 0.500 | 0.691 | 1.225 | 0.500 |
| ('mixed_held_seen_orientation', 'h3_ziv', 's_receive') | 32 | 0.500 | 0.500 | 0.500 | -1.299 | 1.832 | 0.500 |
| ('mixed_held_seen_orientation', 's_give', 'h0_dax') | 32 | 0.500 | 0.500 | 0.500 | 0.691 | 1.225 | 0.500 |
| ('mixed_held_seen_orientation', 's_give', 'h1_mep') | 32 | 0.500 | 0.500 | 0.500 | 0.691 | 1.225 | 0.500 |
| ('mixed_held_seen_orientation', 's_give', 'h2_norp') | 32 | 0.500 | 0.500 | 0.500 | 0.691 | 1.225 | 0.500 |
| ('mixed_held_seen_orientation', 's_give', 'h3_ziv') | 32 | 0.500 | 0.500 | 0.500 | 0.691 | 1.225 | 0.500 |
| ('mixed_held_seen_orientation', 's_receive', 'h0_dax') | 32 | 0.500 | 0.500 | 0.500 | -1.299 | 1.832 | 0.500 |
| ('mixed_held_seen_orientation', 's_receive', 'h1_mep') | 32 | 0.500 | 0.500 | 0.500 | -1.299 | 1.832 | 0.500 |
| ('mixed_held_seen_orientation', 's_receive', 'h2_norp') | 32 | 0.500 | 0.500 | 0.500 | -1.299 | 1.832 | 0.500 |
| ('mixed_held_seen_orientation', 's_receive', 'h3_ziv') | 32 | 0.500 | 0.500 | 0.500 | -1.299 | 1.832 | 0.500 |

## Reviewer interpretation from this analysis

Saved predictions support the state-level sign fingerprint: graph-transfer changed rows reverse between learned bs+ and bs- while unchanged rows remain accurate. The mixed comparison surface remains weak: held-held closure is perfect, but mixed held-seen comparison accuracy is only 0.5 in learned cells and not a full causal gauge experimental design replication. This should be retained as a boundary on the comparison pathway, not ignored.
