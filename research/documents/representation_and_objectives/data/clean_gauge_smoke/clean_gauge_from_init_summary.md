# clean init and attention execution plan clean-init supplied-harness gauge recheck

Each bridge-sign cell is trained from a fresh deep-copy of an untouched condition-specific initialization.

## Initialization audit

| seed | condition | init hash prefix | untouched after all runs |
|---:|---|---|---:|
| 29300 | shared_trunk | d8c14930cfc889a8 | True |

## Central readout

| condition | bridge_sign | seed | train_state_acc | train_cmp_acc | direct_same | graph_same | pair_both_graph_same | unchanged | mixed_acc | mixed_margin | hh_closure | graph_same_margin | graph_mean_de |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shared_trunk | 1 | 29300 | 0.9722 | 0.5677 | 0.2500 | 0.7500 | 0.7500 | 1.0000 | 0.7500 | 0.0000 | 0.5938 | 0.0009 | -0.0028 |
| shared_trunk | -1 | 29300 | 0.9861 | 0.5000 | 0.2500 | 0.7500 | 0.7500 | 1.0000 | 0.6250 | 0.0000 | 0.5312 | 0.0011 | -0.0031 |

## Bridge-sign pairs

### shared_trunk|seed29300|_cleaninit
- bs+1_direct_de: -0.0021204352378845215
- bs+1_graph_de: -0.002781590446829796
- bs+1_graph_margin: 0.0009279977530241013
- bs+1_graph_same: 0.75
- bs+1_hh_closure: 0.59375
- bs+1_mixed_acc: 0.75
- bs+1_mixed_margin: 2.3972243070866357e-06
- bs-1_direct_de: -0.002802673727273941
- bs-1_graph_de: -0.0030854512006044388
- bs-1_graph_margin: 0.0011002589017152786
- bs-1_graph_same: 0.75
- bs-1_hh_closure: 0.53125
- bs-1_mixed_acc: 0.625
- bs-1_mixed_margin: 1.5478581190325424e-06
- de_sign_flip: False
- margin_sign_flip: False

