# clean init and attention execution plan clean-init supplied-harness gauge recheck

Each bridge-sign cell is trained from a fresh deep-copy of an untouched condition-specific initialization.

## Initialization audit

| seed | condition | init hash prefix | untouched after all runs |
|---:|---|---|---:|
| 29000 | shared_trunk | ade1a77b34fadeeb | True |
| 29000 | untied | ade1a77b34fadeeb | True |

## Central readout

| condition | bridge_sign | seed | train_state_acc | train_cmp_acc | direct_same | graph_same | pair_both_graph_same | unchanged | mixed_acc | mixed_margin | hh_closure | graph_same_margin | graph_mean_de |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shared_trunk | 1 | 29000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 13.8089 | 1.0000 | 12.3692 | 2.5601 |
| shared_trunk | -1 | 29000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | -13.8089 | 1.0000 | -14.0324 | -0.1046 |
| untied | 1 | 29000 | 1.0000 | 1.0000 | 1.0000 | 0.7500 | 0.7500 | 1.0000 | 0.5000 | 2.8329 | 1.0000 | 4.6806 | 6.1101 |
| untied | -1 | 29000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 0.5000 | 1.0000 | 0.5000 | 2.8329 | 1.0000 | -0.1785 | 7.7031 |

## Bridge-sign pairs

### shared_trunk|seed29000|_cleaninit
- bs+1_direct_de: 2.02057945728302
- bs+1_graph_de: 2.5601202845573425
- bs+1_graph_margin: 12.369208872318268
- bs+1_graph_same: 1.0
- bs+1_hh_closure: 1.0
- bs+1_mixed_acc: 1.0
- bs+1_mixed_margin: 13.808913768295765
- bs-1_direct_de: -0.38374781608581543
- bs-1_graph_de: -0.10459649562835693
- bs-1_graph_margin: -14.032434105873108
- bs-1_graph_same: 0.0
- bs-1_hh_closure: 1.0
- bs-1_mixed_acc: 0.0
- bs-1_mixed_margin: -13.808913768295765
- de_sign_flip: True
- margin_sign_flip: True

### untied|seed29000|_cleaninit
- bs+1_direct_de: 0.5588822364807129
- bs+1_graph_de: 6.110071629285812
- bs+1_graph_margin: 4.680613547563553
- bs+1_graph_same: 0.75
- bs+1_hh_closure: 1.0
- bs+1_mixed_acc: 0.5
- bs+1_mixed_margin: 2.8329405908562326
- bs-1_direct_de: -0.48979413509368896
- bs-1_graph_de: 7.703144609928131
- bs-1_graph_margin: -0.17853015661239624
- bs-1_graph_same: 0.5
- bs-1_hh_closure: 1.0
- bs-1_mixed_acc: 0.5
- bs-1_mixed_margin: 2.8329405908562326
- de_sign_flip: False
- margin_sign_flip: True

