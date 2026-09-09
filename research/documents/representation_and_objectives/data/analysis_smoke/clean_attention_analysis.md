# clean init and attention execution plan clean-init and raw-name attention analysis

## Clean supplied-harness status

- init_untouched: True
- shared_trunk_full_fit: False
- shared_trunk_clean_transport: False
- untied_full_fit: False
- untied_lacks_shared_transport: True
- clean_mechanism_reestablished: False

### Clean supplied-harness pairs

#### shared_trunk|seed29300
- train_fit_plus: {'state': 0.9722222222222222, 'cmp': 0.5677083333333334}
- train_fit_minus: {'state': 0.9861111111111112, 'cmp': 0.5}
- graph_same_plus: 0.75
- graph_same_minus: 0.75
- direct_same_plus: 0.25
- direct_same_minus: 0.25
- pair_both_plus: 0.75
- pair_both_minus: 0.75
- unchanged_plus: 1.0
- unchanged_minus: 1.0
- hh_closure_plus: 0.59375
- hh_closure_minus: 0.53125
- mixed_acc_plus: 0.75
- mixed_acc_minus: 0.625
- mixed_margin_plus: 2.3972243070866357e-06
- mixed_margin_minus: 1.5478581190325424e-06
- graph_same_margin_plus: 0.0009279977530241013
- graph_same_margin_minus: 0.0011002589017152786
- graph_mean_de_plus: -0.002781590446829796
- graph_mean_de_minus: -0.0030854512006044388
- graph_margin_sign_flip: False
- graph_de_sign_flip: False
- mixed_margin_sign_flip: False


## Raw-name query-attention pairs

No completed attention rows found.

## Row-paired state d_e sign reversal

### clean_supplied_harness
#### shared_trunk|seed29300
- graph_transfer_changed_same: n=96, opposite_frac=0.0, mean_plus=-0.002781590446829796, mean_minus=-0.0030854512006044388
- graph_transfer_changed_all: n=192, opposite_frac=0.0, mean_plus=-0.002781590446829796, mean_minus=-0.0030854512006044388
- direct_anchor_changed_same: n=96, opposite_frac=0.0, mean_plus=-0.0021204352378845215, mean_minus=-0.002802673727273941
- unchanged_all: n=384, opposite_frac=0.0, mean_plus=0.0, mean_minus=0.0

### raw_name_query_attention
No row-paired predictions found.

## Scientific reading

The clean supplied-harness recheck is the load-bearing result: it must show untouched initializations, full local fit, shared_trunk sign-controlled h1/h3 reversal, and untied non-transport. The attention result first tests whether candidate matching is learnable in raw names; any transport-like pattern is provisional unless the clean recheck has reestablished the original mechanism.
