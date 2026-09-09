# corrected multiseed gauge affine and next raw binding multi-seed gauge and affine synthesis

## Per-pair causal gauge readout

| cond | seed | train state +/- | train cmp +/- | graph same +/- | graph oppfrac | mixed acc +/- | mixed margin +/- | hh closure +/- | unchanged +/- |
|---|---:|---|---|---|---:|---|---|---|---|
| shared_trunk | 29000 | 1.000/1.000 | 1.000/1.000 | 1.000/0.000 | 1.000 | 1.000/0.000 | 13.809/-13.809 | 1.000/1.000 | 1.000/1.000 |
| shared_trunk | 29001 | 1.000/1.000 | 1.000/1.000 | 1.000/0.000 | 1.000 | 1.000/0.000 | 12.401/-13.809 | 1.000/1.000 | 1.000/1.000 |
| shared_trunk | 29002 | 1.000/1.000 | 1.000/1.000 | 1.000/0.000 | 1.000 | 1.000/0.250 | 13.809/-10.239 | 1.000/1.000 | 1.000/1.000 |
| tied | 29000 | 1.000/1.000 | 1.000/1.000 | 1.000/0.000 | 1.000 | 1.000/0.000 | 13.747/-13.018 | 1.000/1.000 | 1.000/1.000 |
| tied | 29001 | 1.000/1.000 | 1.000/1.000 | 1.000/0.000 | 1.000 | 1.000/0.000 | 12.899/-13.809 | 1.000/1.000 | 1.000/1.000 |
| tied | 29002 | 1.000/1.000 | 1.000/0.875 | 1.000/0.250 | 0.750 | 1.000/0.125 | 13.809/-6.942 | 1.000/0.625 | 1.000/1.000 |
| untied | 29000 | 1.000/1.000 | 1.000/1.000 | 0.750/0.500 | 0.250 | 0.500/0.500 | 2.833/2.833 | 1.000/1.000 | 1.000/1.000 |
| untied | 29001 | 1.000/1.000 | 1.000/1.000 | 0.750/0.500 | 0.250 | 0.500/0.500 | 0.407/0.407 | 1.000/1.000 | 1.000/1.000 |
| untied | 29002 | 1.000/1.000 | 1.000/1.000 | 0.500/0.750 | 0.750 | 0.750/0.750 | 3.767/3.767 | 1.000/1.000 | 1.000/1.000 |

## By-condition means

### shared_trunk seeds=[29000, 29001, 29002]
- graph_same_plus: mean=1.000 sd=0.000 values=[1.0, 1.0, 1.0]
- graph_same_minus: mean=0.000 sd=0.000 values=[0.0, 0.0, 0.0]
- pair_both_plus: mean=1.000 sd=0.000 values=[1.0, 1.0, 1.0]
- pair_both_minus: mean=0.000 sd=0.000 values=[0.0, 0.0, 0.0]
- mixed_acc_plus: mean=1.000 sd=0.000 values=[1.0, 1.0, 1.0]
- mixed_acc_minus: mean=0.083 sd=0.144 values=[0.0, 0.0, 0.25]
- hh_closure_plus: mean=1.000 sd=0.000 values=[1.0, 1.0, 1.0]
- hh_closure_minus: mean=1.000 sd=0.000 values=[1.0, 1.0, 1.0]
- graph_changed_opposite_frac: mean=1.000 sd=0.000 values=[1.0, 1.0, 1.0]
- unchanged_opposite_frac: mean=0.000 sd=0.000 values=[0.0, 0.0, 0.0]
- mixed_held_seen_orientation_product_same_frac: mean=0.083 sd=0.144 values=[0.0, 0.0, 0.25]
- heldheld_unseen_edge_closure_product_same_frac: mean=1.000 sd=0.000 values=[1.0, 1.0, 1.0]

### tied seeds=[29000, 29001, 29002]
- graph_same_plus: mean=1.000 sd=0.000 values=[1.0, 1.0, 1.0]
- graph_same_minus: mean=0.083 sd=0.144 values=[0.0, 0.0, 0.25]
- pair_both_plus: mean=1.000 sd=0.000 values=[1.0, 1.0, 1.0]
- pair_both_minus: mean=0.083 sd=0.144 values=[0.0, 0.0, 0.25]
- mixed_acc_plus: mean=1.000 sd=0.000 values=[1.0, 1.0, 1.0]
- mixed_acc_minus: mean=0.042 sd=0.072 values=[0.0, 0.0, 0.125]
- hh_closure_plus: mean=1.000 sd=0.000 values=[1.0, 1.0, 1.0]
- hh_closure_minus: mean=0.875 sd=0.217 values=[1.0, 1.0, 0.625]
- graph_changed_opposite_frac: mean=0.917 sd=0.144 values=[1.0, 1.0, 0.75]
- unchanged_opposite_frac: mean=0.000 sd=0.000 values=[0.0, 0.0, 0.0]
- mixed_held_seen_orientation_product_same_frac: mean=0.042 sd=0.072 values=[0.0, 0.0, 0.125]
- heldheld_unseen_edge_closure_product_same_frac: mean=0.875 sd=0.217 values=[1.0, 1.0, 0.625]

### untied seeds=[29000, 29001, 29002]
- graph_same_plus: mean=0.667 sd=0.144 values=[0.75, 0.75, 0.5]
- graph_same_minus: mean=0.583 sd=0.144 values=[0.5, 0.5, 0.75]
- pair_both_plus: mean=0.667 sd=0.144 values=[0.75, 0.75, 0.5]
- pair_both_minus: mean=0.583 sd=0.144 values=[0.5, 0.5, 0.75]
- mixed_acc_plus: mean=0.583 sd=0.144 values=[0.5, 0.5, 0.75]
- mixed_acc_minus: mean=0.583 sd=0.144 values=[0.5, 0.5, 0.75]
- hh_closure_plus: mean=1.000 sd=0.000 values=[1.0, 1.0, 1.0]
- hh_closure_minus: mean=1.000 sd=0.000 values=[1.0, 1.0, 1.0]
- graph_changed_opposite_frac: mean=0.417 sd=0.289 values=[0.25, 0.25, 0.75]
- unchanged_opposite_frac: mean=0.000 sd=0.000 values=[0.0, 0.0, 0.0]
- mixed_held_seen_orientation_product_same_frac: mean=1.000 sd=0.000 values=[1.0, 1.0, 1.0]
- heldheld_unseen_edge_closure_product_same_frac: mean=1.000 sd=0.000 values=[1.0, 1.0, 1.0]

## Frozen-affine scalar evidence

- Included affine summary JSON: `experiments/archive/representation_and_objectives/data/frozen_affine_saved_outputs/frozen_affine_saved_outputs.json` with n_runs=2
  - tied_heldheld_only_seed28801: train_cmp=1.000, hh_closure=1.000, affine graph_same=1.000, graph_all=1.000
  - tied_bs+1_noanchor_seed29000: train_cmp=0.875, hh_closure=0.625, affine graph_same=0.750, graph_all=0.750

## Scientific reading
Tied/shared_trunk stability across seeds supports causal gauge transport by shared representation. Untied stability near partial/local behavior supports that fitting local state anchors and comparisons separately is insufficient. Frozen-affine success should be read as scalar representational sufficiency only: it shows a one-dimensional coordinate can carry the sign once calibrated, not that the unanchored model learned to use it for state updating during training.
