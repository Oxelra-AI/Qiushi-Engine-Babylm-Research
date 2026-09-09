# topology phase1 result topology-pair analysis

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


