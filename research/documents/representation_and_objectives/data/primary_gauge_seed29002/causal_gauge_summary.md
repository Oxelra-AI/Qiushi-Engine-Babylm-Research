# causal gauge experimental design causal gauge-transport probe

Fixed aligned arm, train-only vocabulary, dropout=0, paired initialization.

## Per-run central readout

| condition | bridge_sign | cell_tag | seed | train_state_acc | train_cmp_acc | direct_same | graph_same | pair_both_graph_same | unchanged | mixed_acc | mixed_margin | hh_closure | graph_same_margin | graph_mean_de |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tied | 1 | bs+1 | 29002 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 13.8089 | 1.0000 | 17.4258 | -0.2533 |
| shared_trunk | 1 | bs+1 | 29002 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 13.8089 | 1.0000 | 15.3059 | -0.0389 |
| untied | 1 | bs+1 | 29002 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 0.5000 | 1.0000 | 0.7500 | 3.7671 | 1.0000 | 1.5901 | 2.3379 |
| tied | -1 | bs-1 | 29002 | 1.0000 | 0.8750 | 0.0000 | 0.2500 | 0.2500 | 1.0000 | 0.1250 | -6.9420 | 0.6250 | -5.9289 | 6.4704 |
| shared_trunk | -1 | bs-1 | 29002 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2500 | -10.2389 | 1.0000 | -12.6601 | -0.6908 |
| untied | -1 | bs-1 | 29002 | 1.0000 | 1.0000 | 0.0000 | 0.7500 | 0.7500 | 1.0000 | 0.7500 | 3.7671 | 1.0000 | 5.4388 | 4.1308 |

## Bridge-sign paired comparisons

### shared_trunk|seed29002|full
  bs+1_direct_de: 0.17617928981781006
  bs+1_graph_de: -0.038921356201171875
  bs+1_graph_margin: 15.305894374847412
  bs+1_graph_same: 1.0
  bs+1_hh_closure: 1.0
  bs+1_mixed_acc: 1.0
  bs+1_mixed_margin: 13.808913768295765
  bs-1_direct_de: -1.0471851825714111
  bs-1_graph_de: -0.6908173561096191
  bs-1_graph_margin: -12.660142421722412
  bs-1_graph_same: 0.0
  bs-1_hh_closure: 1.0
  bs-1_mixed_acc: 0.25
  bs-1_mixed_margin: -10.238922101740416
  de_sign_flip: False
  margin_sign_flip: True

### tied|seed29002|full
  bs+1_direct_de: -0.1197504997253418
  bs+1_graph_de: -0.253324031829834
  bs+1_graph_margin: 17.425797939300537
  bs+1_graph_same: 1.0
  bs+1_hh_closure: 1.0
  bs+1_mixed_acc: 1.0
  bs+1_mixed_margin: 13.808913768295765
  bs-1_direct_de: 0.3209172487258911
  bs-1_graph_de: 6.470368504524231
  bs-1_graph_margin: -5.928922057151794
  bs-1_graph_same: 0.25
  bs-1_hh_closure: 0.625
  bs-1_mixed_acc: 0.125
  bs-1_mixed_margin: -6.9419788509473594
  de_sign_flip: True
  margin_sign_flip: True

### untied|seed29002|full
  bs+1_direct_de: 0.3257182836532593
  bs+1_graph_de: 2.33793181180954
  bs+1_graph_margin: 1.590078055858612
  bs+1_graph_same: 0.5
  bs+1_hh_closure: 1.0
  bs+1_mixed_acc: 0.75
  bs+1_mixed_margin: 3.7671274871457485
  bs-1_direct_de: -0.16981840133666992
  bs-1_graph_de: 4.130797415971756
  bs-1_graph_margin: 5.4387610256671906
  bs-1_graph_same: 0.75
  bs-1_hh_closure: 1.0
  bs-1_mixed_acc: 0.75
  bs-1_mixed_margin: 3.7671274871457485
  de_sign_flip: False
  margin_sign_flip: False

## Scientific reading

If the tied model shows graph_same_margin and graph_mean_de reversing between bs+1 and bs-1 while hh_closure and seen references remain stable, this is evidence for absolute gauge transport through the shared coordinate.  shared_trunk and untied models should not show the same coherent reversal.  anchor-only and comparison-only cells provide the factorial interaction.
