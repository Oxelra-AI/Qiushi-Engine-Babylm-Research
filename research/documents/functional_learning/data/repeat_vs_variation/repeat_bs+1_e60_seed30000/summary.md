# mechanism macro convergence: repeat bs+1 e60

- model: shared_trunk
- seed: 30000
- epochs: 60
- comparison events per epoch: 192
- unique comparison pairs seen: 192
- final model hash: 6f36407c6e14cf83

## Per-relation eval accuracy (changed state, held-out names)

| relation | n | accuracy | mean_margin |
|---|---:|---:|---:|
| h0_dax | 96 | 1.0000 | 15.6297 |
| h1_mep | 96 | 1.0000 | 17.4389 |
| h2_norp | 96 | 1.0000 | 19.5983 |
| h3_ziv | 96 | 1.0000 | 18.0361 |

## Central eval

```json
{
  "direct_same": 1.0,
  "graph_mean_de": 1.3090559262782335,
  "graph_same": 1.0,
  "graph_same_margin": 17.565678058192134,
  "hh_closure": 1.0,
  "hh_closure_margin": 3.554767237169305,
  "mixed_acc": 1.0,
  "mixed_acc_margin": 2.6845681311574263,
  "pair_both_graph_same": 0.75,
  "same_init_changed": 1.0,
  "unchanged": 0.828125
}
```

## Train last epoch

```json
{
  "epoch": 60,
  "loss": 2.4882044726837194e-06,
  "train_changed": 1.0,
  "train_cmp_fixed": 1.0,
  "train_state": 1.0,
  "train_unchanged": 1.0
}
```
