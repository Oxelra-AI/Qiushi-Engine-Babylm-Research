# comparison channel design row-paired sign analysis

| family | n | metric | value |
|---|---:|---|---:|
| direct_anchor | 384 | opposite_sign_frac | 0.5000 |
| direct_anchor | 384 | same_sign_frac | 0.5000 |
| direct_anchor | 384 | mean_d_e_plus | -1.8988 |
| direct_anchor | 384 | mean_d_e_minus | -0.0499 |
| graph_transfer | 384 | opposite_sign_frac | 0.1250 |
| graph_transfer | 384 | same_sign_frac | 0.8750 |
| graph_transfer | 384 | mean_d_e_plus | -4.7357 |
| graph_transfer | 384 | mean_d_e_minus | -1.4621 |
| unchanged | 384 | same_sign_frac | 1.0000 |
| comp_product_stability | 640 | same_sign_frac | 0.3750 |

## Central eval metrics

### Bridge sign +1
- direct_same: 1.0
- graph_mean_de: -9.471410006284714
- graph_same: 0.25
- graph_same_margin: -3.62380912899971
- hh_closure: 0.25
- hh_closure_margin: -0.08341683987858917
- mixed_acc: 0.625
- mixed_acc_margin: 0.12096694012582312
- pair_both_graph_same: 0.125
- same_init_changed: 0.625
- unchanged: 0.5
### Bridge sign -1
- direct_same: 0.0
- graph_mean_de: -2.924140304327011
- graph_same: 0.5
- graph_same_margin: -2.138093739748001
- hh_closure: 0.5
- hh_closure_margin: 0.007245954346315848
- mixed_acc: 0.25
- mixed_acc_margin: -0.08509558166536636
- pair_both_graph_same: 0.25
- same_init_changed: 0.25
- unchanged: 0.5
