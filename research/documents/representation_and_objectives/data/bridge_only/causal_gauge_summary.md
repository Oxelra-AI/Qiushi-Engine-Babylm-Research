# causal gauge experimental design causal gauge-transport probe

Fixed aligned arm, train-only vocabulary, dropout=0, paired initialization.

## Per-run central readout

| condition | bridge_sign | cell_tag | seed | train_state_acc | train_cmp_acc | direct_same | graph_same | pair_both_graph_same | unchanged | mixed_acc | mixed_margin | hh_closure | graph_same_margin | graph_mean_de |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tied | 1 | bs+1_nocmp | 29000 | 1.0000 | nan | 1.0000 | 0.5000 | 0.5000 | 1.0000 | 0.7500 | 6.5013 | 0.5000 | 0.0000 | -0.2297 |
| untied | 1 | bs+1_nocmp | 29000 | 1.0000 | nan | 1.0000 | 0.5000 | 0.5000 | 1.0000 | 0.5000 | -0.0000 | 0.5000 | 0.0000 | -0.2297 |
| tied | -1 | bs-1_nocmp | 29000 | 1.0000 | nan | 0.0000 | 0.5000 | 0.5000 | 1.0000 | 0.2500 | -6.7018 | 0.2500 | 0.0000 | -0.1350 |
| untied | -1 | bs-1_nocmp | 29000 | 1.0000 | nan | 0.0000 | 0.5000 | 0.5000 | 1.0000 | 0.5000 | -0.0000 | 0.5000 | 0.0000 | -0.1350 |

## Bridge-sign paired comparisons

### tied|seed29000|_nocmp
  bs+1_direct_de: 0.04787302017211914
  bs+1_graph_de: -0.22970378398895264
  bs+1_graph_margin: 0.0
  bs+1_graph_same: 0.5
  bs+1_hh_closure: 0.5
  bs+1_mixed_acc: 0.75
  bs+1_mixed_margin: 6.501273801347762
  bs-1_direct_de: -0.18776607513427734
  bs-1_graph_de: -0.13496613502502441
  bs-1_graph_margin: 0.0
  bs-1_graph_same: 0.5
  bs-1_hh_closure: 0.25
  bs-1_mixed_acc: 0.25
  bs-1_mixed_margin: -6.701804472044353
  de_sign_flip: False
  margin_sign_flip: False

### untied|seed29000|_nocmp
  bs+1_direct_de: 0.04787302017211914
  bs+1_graph_de: -0.22970378398895264
  bs+1_graph_margin: 0.0
  bs+1_graph_same: 0.5
  bs+1_hh_closure: 0.5
  bs+1_mixed_acc: 0.5
  bs+1_mixed_margin: -5.010515451984546e-07
  bs-1_direct_de: -0.18776607513427734
  bs-1_graph_de: -0.13496613502502441
  bs-1_graph_margin: 0.0
  bs-1_graph_same: 0.5
  bs-1_hh_closure: 0.5
  bs-1_mixed_acc: 0.5
  bs-1_mixed_margin: -5.010515451984546e-07
  de_sign_flip: False
  margin_sign_flip: False

## Scientific reading

If the tied model shows graph_same_margin and graph_mean_de reversing between bs+1 and bs-1 while hh_closure and seen references remain stable, this is evidence for absolute gauge transport through the shared coordinate.  shared_trunk and untied models should not show the same coherent reversal.  anchor-only and comparison-only cells provide the factorial interaction.
