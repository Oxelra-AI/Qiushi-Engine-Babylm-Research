# topology phase1 result topology-pair analysis

## full_graph_e60

### Central metrics

| bridge sign | train_state | train_cmp_modified | train_cmp_original | direct_same | graph_same | pair_both_graph_same | hh_closure | mixed_acc | unchanged |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 0.500 |
| -1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.500 |

### Hash comparison

- init_event_trunk_equal: `True`
- pretrain_event_trunk_equal: `True`
- final_event_trunk_equal: `False`
- final_event_state_head_equal: `False`

### Per-relation paired sign analysis

| relation | matched | acc + | acc - | opposite frac | same frac | mean d_e + | mean d_e - | abs d_e + | abs d_e - | anti-corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h0_dax | 96 | 1.000 | 0.000 | 1.000 | 0.000 | 0.872 | -0.386 | 11.963 | 19.955 | 1.000 |
| h1_mep | 96 | 1.000 | 0.000 | 1.000 | 0.000 | -0.281 | -0.362 | 11.938 | 20.470 | 1.000 |
| h2_norp | 96 | 1.000 | 0.000 | 1.000 | 0.000 | 1.360 | -0.656 | 12.605 | 20.848 | 1.000 |
| h3_ziv | 96 | 1.000 | 0.000 | 1.000 | 0.000 | -0.399 | -2.958 | 15.912 | 17.009 | 1.000 |

### Per-relation changed-state accuracy by sign

#### bridge sign +1

| relation | subset | n | acc | mean d_e | mean abs d_e |
|---|---|---:|---:|---:|---:|
| h0_dax | all_changed | 96 | 1.000 | 0.872 | 11.963 |
| h0_dax | paired_same | 32 | 1.000 | 0.872 | 11.963 |
| h0_dax | cross_template | 32 | 1.000 | 0.872 | 11.963 |
| h1_mep | all_changed | 96 | 1.000 | -0.281 | 11.938 |
| h1_mep | paired_same | 32 | 1.000 | -0.281 | 11.938 |
| h1_mep | cross_template | 32 | 1.000 | -0.281 | 11.938 |
| h2_norp | all_changed | 96 | 1.000 | 1.360 | 12.605 |
| h2_norp | paired_same | 32 | 1.000 | 1.360 | 12.605 |
| h2_norp | cross_template | 32 | 1.000 | 1.360 | 12.605 |
| h3_ziv | all_changed | 96 | 1.000 | -0.399 | 15.912 |
| h3_ziv | paired_same | 32 | 1.000 | -0.399 | 15.912 |
| h3_ziv | cross_template | 32 | 1.000 | -0.399 | 15.912 |

#### bridge sign -1

| relation | subset | n | acc | mean d_e | mean abs d_e |
|---|---|---:|---:|---:|---:|
| h0_dax | all_changed | 96 | 0.000 | -0.386 | 19.955 |
| h0_dax | paired_same | 32 | 0.000 | -0.386 | 19.955 |
| h0_dax | cross_template | 32 | 0.000 | -0.386 | 19.955 |
| h1_mep | all_changed | 96 | 0.000 | -0.362 | 20.470 |
| h1_mep | paired_same | 32 | 0.000 | -0.362 | 20.470 |
| h1_mep | cross_template | 32 | 0.000 | -0.362 | 20.470 |
| h2_norp | all_changed | 96 | 0.000 | -0.656 | 20.848 |
| h2_norp | paired_same | 32 | 0.000 | -0.656 | 20.848 |
| h2_norp | cross_template | 32 | 0.000 | -0.656 | 20.848 |
| h3_ziv | all_changed | 96 | 0.000 | -2.958 | 17.009 |
| h3_ziv | paired_same | 32 | 0.000 | -2.958 | 17.009 |
| h3_ziv | cross_template | 32 | 0.000 | -2.958 | 17.009 |


## detach_full_graph_e60

### Central metrics

| bridge sign | train_state | train_cmp_modified | train_cmp_original | direct_same | graph_same | pair_both_graph_same | hh_closure | mixed_acc | unchanged |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +1 | 0.958 | 0.443 | 0.443 | 0.250 | 0.500 | 0.250 | 0.469 | 0.516 | 0.500 |
| -1 | 0.986 | 0.443 | 0.443 | 0.250 | 0.500 | 0.250 | 0.469 | 0.516 | 0.500 |

### Hash comparison

- init_event_trunk_equal: `True`
- pretrain_event_trunk_equal: `True`
- final_event_trunk_equal: `True`
- final_event_state_head_equal: `False`

### Per-relation paired sign analysis

| relation | matched | acc + | acc - | opposite frac | same frac | mean d_e + | mean d_e - | abs d_e + | abs d_e - | anti-corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h0_dax | 96 | 0.000 | 0.000 | 0.000 | 1.000 | -0.008 | 0.006 | 0.026 | 0.037 | -1.000 |
| h1_mep | 96 | 0.500 | 0.500 | 0.000 | 1.000 | -0.034 | -0.035 | 0.034 | 0.035 | -1.000 |
| h2_norp | 96 | 0.500 | 0.500 | 0.000 | 1.000 | -0.029 | -0.032 | 0.029 | 0.032 | -1.000 |
| h3_ziv | 96 | 0.500 | 0.500 | 0.000 | 1.000 | -0.051 | -0.042 | 0.051 | 0.042 | -1.000 |

### Per-relation changed-state accuracy by sign

#### bridge sign +1

| relation | subset | n | acc | mean d_e | mean abs d_e |
|---|---|---:|---:|---:|---:|
| h0_dax | all_changed | 96 | 0.000 | -0.008 | 0.026 |
| h0_dax | paired_same | 32 | 0.000 | -0.008 | 0.026 |
| h0_dax | cross_template | 32 | 0.000 | -0.008 | 0.026 |
| h1_mep | all_changed | 96 | 0.500 | -0.034 | 0.034 |
| h1_mep | paired_same | 32 | 0.500 | -0.034 | 0.034 |
| h1_mep | cross_template | 32 | 0.500 | -0.034 | 0.034 |
| h2_norp | all_changed | 96 | 0.500 | -0.029 | 0.029 |
| h2_norp | paired_same | 32 | 0.500 | -0.029 | 0.029 |
| h2_norp | cross_template | 32 | 0.500 | -0.029 | 0.029 |
| h3_ziv | all_changed | 96 | 0.500 | -0.051 | 0.051 |
| h3_ziv | paired_same | 32 | 0.500 | -0.051 | 0.051 |
| h3_ziv | cross_template | 32 | 0.500 | -0.051 | 0.051 |

#### bridge sign -1

| relation | subset | n | acc | mean d_e | mean abs d_e |
|---|---|---:|---:|---:|---:|
| h0_dax | all_changed | 96 | 0.000 | 0.006 | 0.037 |
| h0_dax | paired_same | 32 | 0.000 | 0.006 | 0.037 |
| h0_dax | cross_template | 32 | 0.000 | 0.006 | 0.037 |
| h1_mep | all_changed | 96 | 0.500 | -0.035 | 0.035 |
| h1_mep | paired_same | 32 | 0.500 | -0.035 | 0.035 |
| h1_mep | cross_template | 32 | 0.500 | -0.035 | 0.035 |
| h2_norp | all_changed | 96 | 0.500 | -0.032 | 0.032 |
| h2_norp | paired_same | 32 | 0.500 | -0.032 | 0.032 |
| h2_norp | cross_template | 32 | 0.500 | -0.032 | 0.032 |
| h3_ziv | all_changed | 96 | 0.500 | -0.042 | 0.042 |
| h3_ziv | paired_same | 32 | 0.500 | -0.042 | 0.042 |
| h3_ziv | cross_template | 32 | 0.500 | -0.042 | 0.042 |


## disconn_delete_e60

### Central metrics

| bridge sign | train_state | train_cmp_modified | train_cmp_original | direct_same | graph_same | pair_both_graph_same | hh_closure | mixed_acc | unchanged |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 0.250 | 1.000 | 1.000 | 0.500 |
| -1 | 1.000 | 1.000 | 0.750 | 0.000 | 0.250 | 0.125 | 0.750 | 0.125 | 0.500 |

### Hash comparison

- init_event_trunk_equal: `True`
- pretrain_event_trunk_equal: `True`
- final_event_trunk_equal: `False`
- final_event_state_head_equal: `False`

### Per-relation paired sign analysis

| relation | matched | acc + | acc - | opposite frac | same frac | mean d_e + | mean d_e - | abs d_e + | abs d_e - | anti-corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h0_dax | 96 | 1.000 | 0.000 | 1.000 | 0.000 | 2.926 | 0.109 | 17.863 | 18.458 | 1.000 |
| h1_mep | 96 | 1.000 | 0.500 | 0.500 | 0.500 | 7.492 | -4.801 | 11.967 | 4.801 | 1.000 |
| h2_norp | 96 | 1.000 | 0.000 | 1.000 | 0.000 | 0.372 | -3.202 | 20.952 | 16.654 | 1.000 |
| h3_ziv | 96 | 0.000 | 0.000 | 0.000 | 1.000 | 3.491 | -2.061 | 6.947 | 16.845 | -1.000 |

### Per-relation changed-state accuracy by sign

#### bridge sign +1

| relation | subset | n | acc | mean d_e | mean abs d_e |
|---|---|---:|---:|---:|---:|
| h0_dax | all_changed | 96 | 1.000 | 2.926 | 17.863 |
| h0_dax | paired_same | 32 | 1.000 | 2.926 | 17.863 |
| h0_dax | cross_template | 32 | 1.000 | 2.926 | 17.863 |
| h1_mep | all_changed | 96 | 1.000 | 7.492 | 11.967 |
| h1_mep | paired_same | 32 | 1.000 | 7.492 | 11.967 |
| h1_mep | cross_template | 32 | 1.000 | 7.492 | 11.967 |
| h2_norp | all_changed | 96 | 1.000 | 0.372 | 20.952 |
| h2_norp | paired_same | 32 | 1.000 | 0.372 | 20.952 |
| h2_norp | cross_template | 32 | 1.000 | 0.372 | 20.952 |
| h3_ziv | all_changed | 96 | 0.000 | 3.491 | 6.947 |
| h3_ziv | paired_same | 32 | 0.000 | 3.491 | 6.947 |
| h3_ziv | cross_template | 32 | 0.000 | 3.491 | 6.947 |

#### bridge sign -1

| relation | subset | n | acc | mean d_e | mean abs d_e |
|---|---|---:|---:|---:|---:|
| h0_dax | all_changed | 96 | 0.000 | 0.109 | 18.458 |
| h0_dax | paired_same | 32 | 0.000 | 0.109 | 18.458 |
| h0_dax | cross_template | 32 | 0.000 | 0.109 | 18.458 |
| h1_mep | all_changed | 96 | 0.500 | -4.801 | 4.801 |
| h1_mep | paired_same | 32 | 0.500 | -4.801 | 4.801 |
| h1_mep | cross_template | 32 | 0.500 | -4.801 | 4.801 |
| h2_norp | all_changed | 96 | 0.000 | -3.202 | 16.654 |
| h2_norp | paired_same | 32 | 0.000 | -3.202 | 16.654 |
| h2_norp | cross_template | 32 | 0.000 | -3.202 | 16.654 |
| h3_ziv | all_changed | 96 | 0.000 | -2.061 | 16.845 |
| h3_ziv | paired_same | 32 | 0.000 | -2.061 | 16.845 |
| h3_ziv | cross_template | 32 | 0.000 | -2.061 | 16.845 |


## disconn_adversarial_e60

### Central metrics

| bridge sign | train_state | train_cmp_modified | train_cmp_original | direct_same | graph_same | pair_both_graph_same | hh_closure | mixed_acc | unchanged |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +1 | 1.000 | 1.000 | 0.500 | 1.000 | 0.500 | 0.250 | 0.500 | 0.750 | 0.500 |
| -1 | 1.000 | 1.000 | 0.500 | 0.000 | 0.500 | 0.250 | 0.500 | 0.250 | 0.500 |

### Hash comparison

- init_event_trunk_equal: `True`
- pretrain_event_trunk_equal: `True`
- final_event_trunk_equal: `False`
- final_event_state_head_equal: `False`

### Per-relation paired sign analysis

| relation | matched | acc + | acc - | opposite frac | same frac | mean d_e + | mean d_e - | abs d_e + | abs d_e - | anti-corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h0_dax | 96 | 1.000 | 0.000 | 1.000 | 0.000 | -0.858 | 0.204 | 12.020 | 10.199 | 1.000 |
| h1_mep | 96 | 0.000 | 1.000 | 1.000 | 0.000 | -0.469 | -1.629 | 20.804 | 10.322 | 1.000 |
| h2_norp | 96 | 1.000 | 0.000 | 1.000 | 0.000 | -0.880 | -1.660 | 20.841 | 8.436 | 1.000 |
| h3_ziv | 96 | 1.000 | 0.000 | 1.000 | 0.000 | -0.744 | -1.619 | 0.917 | 8.592 | 1.000 |

### Per-relation changed-state accuracy by sign

#### bridge sign +1

| relation | subset | n | acc | mean d_e | mean abs d_e |
|---|---|---:|---:|---:|---:|
| h0_dax | all_changed | 96 | 1.000 | -0.858 | 12.020 |
| h0_dax | paired_same | 32 | 1.000 | -0.858 | 12.020 |
| h0_dax | cross_template | 32 | 1.000 | -0.858 | 12.020 |
| h1_mep | all_changed | 96 | 0.000 | -0.469 | 20.804 |
| h1_mep | paired_same | 32 | 0.000 | -0.469 | 20.804 |
| h1_mep | cross_template | 32 | 0.000 | -0.469 | 20.804 |
| h2_norp | all_changed | 96 | 1.000 | -0.880 | 20.841 |
| h2_norp | paired_same | 32 | 1.000 | -0.880 | 20.841 |
| h2_norp | cross_template | 32 | 1.000 | -0.880 | 20.841 |
| h3_ziv | all_changed | 96 | 1.000 | -0.744 | 0.917 |
| h3_ziv | paired_same | 32 | 1.000 | -0.744 | 0.917 |
| h3_ziv | cross_template | 32 | 1.000 | -0.744 | 0.917 |

#### bridge sign -1

| relation | subset | n | acc | mean d_e | mean abs d_e |
|---|---|---:|---:|---:|---:|
| h0_dax | all_changed | 96 | 0.000 | 0.204 | 10.199 |
| h0_dax | paired_same | 32 | 0.000 | 0.204 | 10.199 |
| h0_dax | cross_template | 32 | 0.000 | 0.204 | 10.199 |
| h1_mep | all_changed | 96 | 1.000 | -1.629 | 10.322 |
| h1_mep | paired_same | 32 | 1.000 | -1.629 | 10.322 |
| h1_mep | cross_template | 32 | 1.000 | -1.629 | 10.322 |
| h2_norp | all_changed | 96 | 0.000 | -1.660 | 8.436 |
| h2_norp | paired_same | 32 | 0.000 | -1.660 | 8.436 |
| h2_norp | cross_template | 32 | 0.000 | -1.660 | 8.436 |
| h3_ziv | all_changed | 96 | 0.000 | -1.619 | 8.592 |
| h3_ziv | paired_same | 32 | 0.000 | -1.619 | 8.592 |
| h3_ziv | cross_template | 32 | 0.000 | -1.619 | 8.592 |


