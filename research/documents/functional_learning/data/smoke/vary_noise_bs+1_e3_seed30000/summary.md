# mechanism macro convergence: vary_noise bs+1 e3

- model: shared_trunk
- seed: 30000
- epochs: 3
- comparison events per epoch: 192
- unique comparison pairs seen: 574
- final model hash: fca998fe0fc289c1

## Per-relation eval accuracy (changed state, held-out names)

| relation | n | accuracy | mean_margin |
|---|---:|---:|---:|
| h0_dax | 96 | 0.5833 | 0.0004 |
| h1_mep | 96 | 0.6250 | 0.0013 |
| h2_norp | 96 | 0.3333 | -0.0009 |
| h3_ziv | 96 | 0.3750 | -0.0007 |

## Central eval

```json
{
  "direct_same": 0.40625,
  "graph_mean_de": -0.0012122206389904022,
  "graph_same": 0.5625,
  "graph_same_margin": 0.0004763845354318619,
  "hh_closure": 0.375,
  "hh_closure_margin": -2.8870999813081475e-08,
  "mixed_acc": 0.470703125,
  "mixed_acc_margin": 2.1886080503433374e-08,
  "pair_both_graph_same": 0.328125,
  "same_init_changed": 0.484375,
  "unchanged": 0.59375
}
```

## Train last epoch

```json
{
  "epoch": 3,
  "loss": 0.5605788230895996,
  "train_changed": 0.8055555555555556,
  "train_cmp_fixed": 0.46875,
  "train_state": 0.7291666666666666,
  "train_unchanged": 0.6527777777777778
}
```
