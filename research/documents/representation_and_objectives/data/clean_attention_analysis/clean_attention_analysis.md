# clean init and attention execution plan clean-init and raw-name attention analysis

## Clean supplied-harness status

- init_untouched: True
- shared_trunk_full_fit: True
- shared_trunk_clean_transport: True
- untied_full_fit: True
- untied_lacks_shared_transport: True
- clean_mechanism_reestablished: True

### Clean supplied-harness pairs

#### shared_trunk|seed29000
- train_fit_plus: {'state': 1.0, 'cmp': 1.0}
- train_fit_minus: {'state': 1.0, 'cmp': 1.0}
- graph_same_plus: 1.0
- graph_same_minus: 0.0
- direct_same_plus: 1.0
- direct_same_minus: 0.0
- pair_both_plus: 1.0
- pair_both_minus: 0.0
- unchanged_plus: 1.0
- unchanged_minus: 1.0
- hh_closure_plus: 1.0
- hh_closure_minus: 1.0
- mixed_acc_plus: 1.0
- mixed_acc_minus: 0.0
- mixed_margin_plus: 13.808913768295765
- mixed_margin_minus: -13.808913768295765
- graph_same_margin_plus: 12.369208872318268
- graph_same_margin_minus: -14.032434105873108
- graph_mean_de_plus: 2.5601202845573425
- graph_mean_de_minus: -0.10459649562835693
- graph_margin_sign_flip: True
- graph_de_sign_flip: True
- mixed_margin_sign_flip: True

#### untied|seed29000
- train_fit_plus: {'state': 1.0, 'cmp': 1.0}
- train_fit_minus: {'state': 1.0, 'cmp': 1.0}
- graph_same_plus: 0.75
- graph_same_minus: 0.5
- direct_same_plus: 1.0
- direct_same_minus: 0.0
- pair_both_plus: 0.75
- pair_both_minus: 0.5
- unchanged_plus: 1.0
- unchanged_minus: 1.0
- hh_closure_plus: 1.0
- hh_closure_minus: 1.0
- mixed_acc_plus: 0.5
- mixed_acc_minus: 0.5
- mixed_margin_plus: 2.8329405908562326
- mixed_margin_minus: 2.8329405908562326
- graph_same_margin_plus: 4.680613547563553
- graph_same_margin_minus: -0.17853015661239624
- graph_mean_de_plus: 6.110071629285812
- graph_mean_de_minus: 7.703144609928131
- graph_margin_sign_flip: True
- graph_de_sign_flip: False
- mixed_margin_sign_flip: False


## Raw-name query-attention pairs

#### shared_trunk|seed29300
- binding_fit: False
- transport_like: False
- train_fit_plus: {'state': 1.0, 'cmp': 1.0}
- train_fit_minus: {'state': 1.0, 'cmp': 0.9479166666666666}
- graph_same_plus: 0.6875
- graph_same_minus: 0.53125
- pair_both_plus: 0.59375
- pair_both_minus: 0.46875
- unchanged_plus: 0.8125
- unchanged_minus: 0.8125
- hh_closure_plus: 0.546875
- hh_closure_minus: 0.421875
- mixed_acc_plus: 0.5390625
- mixed_acc_minus: 0.4921875
- mixed_margin_plus: 1.8559329045624144
- mixed_margin_minus: 0.2807204157449372
- graph_de_sign_flip: None

#### tied|seed29300
- binding_fit: False
- transport_like: False
- train_fit_plus: {'state': 1.0, 'cmp': 1.0}
- train_fit_minus: {'state': 1.0, 'cmp': 0.9791666666666666}
- graph_same_plus: 0.84375
- graph_same_minus: 0.3125
- pair_both_plus: 0.71875
- pair_both_minus: 0.265625
- unchanged_plus: 0.8125
- unchanged_minus: 0.8125
- hh_closure_plus: 0.75
- hh_closure_minus: 0.578125
- mixed_acc_plus: 0.77734375
- mixed_acc_minus: 0.3046875
- mixed_margin_plus: 3.676011604332886
- mixed_margin_minus: -3.2082199451621083
- graph_de_sign_flip: None


## Row-paired state d_e sign reversal

### clean_supplied_harness
#### shared_trunk|seed29000
- graph_transfer_changed_same: n=96, opposite_frac=1.0, mean_plus=2.5601202845573425, mean_minus=-0.10459649562835693
- graph_transfer_changed_all: n=192, opposite_frac=1.0, mean_plus=2.5601202845573425, mean_minus=-0.10459649562835693
- direct_anchor_changed_same: n=96, opposite_frac=1.0, mean_plus=2.02057945728302, mean_minus=-0.38374781608581543
- unchanged_all: n=384, opposite_frac=0.0, mean_plus=0.0, mean_minus=0.0

#### untied|seed29000
- graph_transfer_changed_same: n=96, opposite_frac=0.25, mean_plus=6.110071629285812, mean_minus=7.703144609928131
- graph_transfer_changed_all: n=192, opposite_frac=0.25, mean_plus=6.110071629285812, mean_minus=7.703144609928131
- direct_anchor_changed_same: n=96, opposite_frac=1.0, mean_plus=0.5588822364807129, mean_minus=-0.48979413509368896
- unchanged_all: n=384, opposite_frac=0.0, mean_plus=0.0, mean_minus=0.0

### raw_name_query_attention
#### shared_trunk|seed29300
- graph_transfer_changed_same: n=128, opposite_frac=0.59375, mean_plus=0.4736597165465355, mean_minus=-2.5867250841110945
- graph_transfer_changed_all: n=256, opposite_frac=0.59375, mean_plus=0.4736597165465355, mean_minus=-2.5867250841110945
- direct_anchor_changed_same: n=128, opposite_frac=0.65625, mean_plus=0.7655275471042842, mean_minus=-0.28506451239809394
- unchanged_all: n=512, opposite_frac=0.0, mean_plus=0.0, mean_minus=0.0

#### tied|seed29300
- graph_transfer_changed_same: n=128, opposite_frac=0.71875, mean_plus=-1.8741222922690213, mean_minus=1.4641926866024733
- graph_transfer_changed_all: n=256, opposite_frac=0.71875, mean_plus=-1.8741222922690213, mean_minus=1.4641926866024733
- direct_anchor_changed_same: n=128, opposite_frac=0.75, mean_plus=-1.3172772033140063, mean_minus=0.4513103652279824
- unchanged_all: n=512, opposite_frac=0.0, mean_plus=0.0, mean_minus=0.0


## Scientific reading

The clean supplied-harness recheck is the load-bearing result: it must show untouched initializations, full local fit, shared_trunk sign-controlled h1/h3 reversal, and untied non-transport. The attention result first tests whether candidate matching is learnable in raw names; any transport-like pattern is provisional unless the clean recheck has reestablished the original mechanism.
