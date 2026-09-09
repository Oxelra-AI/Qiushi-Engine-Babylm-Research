# mechanism macro convergence: vary_context bs+1 e3

- model: shared_trunk
- seed: 30000
- epochs: 3
- comparison events per epoch: 192
- unique comparison pairs seen: 575
- final model hash: df3cf10ab994a7e4

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
  "graph_mean_de": -0.00121227465569973,
  "graph_same": 0.5625,
  "graph_same_margin": 0.00047637708485126495,
  "hh_closure": 0.53125,
  "hh_closure_margin": 8.38190317153758e-09,
  "mixed_acc": 0.513671875,
  "mixed_acc_margin": 3.119930624958378e-08,
  "pair_both_graph_same": 0.328125,
  "same_init_changed": 0.484375,
  "unchanged": 0.59375
}
```

## Train last epoch

```json
{
  "epoch": 3,
  "loss": 0.5605787634849548,
  "train_changed": 0.8055555555555556,
  "train_cmp_fixed": 0.4635416666666667,
  "train_state": 0.7291666666666666,
  "train_unchanged": 0.6527777777777778
}
```
