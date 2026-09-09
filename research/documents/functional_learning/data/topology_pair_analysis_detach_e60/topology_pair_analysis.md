# topology phase1 result topology-pair analysis

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


