# causal gauge experimental design causal gauge-transport probe

Fixed aligned arm, train-only vocabulary, dropout=0, paired initialization.

## Per-run central readout

| condition | bridge_sign | cell_tag | seed | train_state_acc | train_cmp_acc | direct_same | graph_same | pair_both_graph_same | unchanged | mixed_acc | mixed_margin | hh_closure | graph_same_margin | graph_mean_de |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tied | 1 | bs+1 | 29001 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 12.8988 | 1.0000 | 15.1642 | -1.5702 |
| shared_trunk | 1 | bs+1 | 29001 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 12.4013 | 1.0000 | 14.2060 | -0.9007 |
| untied | 1 | bs+1 | 29001 | 1.0000 | 1.0000 | 1.0000 | 0.7500 | 0.7500 | 1.0000 | 0.5000 | 0.4067 | 1.0000 | 4.0659 | 5.6239 |
| tied | -1 | bs-1 | 29001 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | -13.8089 | 1.0000 | -16.0197 | 0.2375 |
| shared_trunk | -1 | bs-1 | 29001 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | -13.8089 | 1.0000 | -13.8859 | -0.5247 |
| untied | -1 | bs-1 | 29001 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 0.5000 | 1.0000 | 0.5000 | 0.4067 | 1.0000 | 1.2190 | 1.8681 |

## Bridge-sign paired comparisons

### shared_trunk|seed29001|full
  bs+1_direct_de: -0.6887079477310181
  bs+1_graph_de: -0.90070641040802
  bs+1_graph_margin: 14.206043362617493
  bs+1_graph_same: 1.0
  bs+1_hh_closure: 1.0
  bs+1_mixed_acc: 1.0
  bs+1_mixed_margin: 12.401334088552359
  bs-1_direct_de: 0.04229116439819336
  bs-1_graph_de: -0.5247339010238647
  bs-1_graph_margin: -13.885851502418518
  bs-1_graph_same: 0.0
  bs-1_hh_closure: 1.0
  bs-1_mixed_acc: 0.0
  bs-1_mixed_margin: -13.808913768295765
  de_sign_flip: False
  margin_sign_flip: True

### tied|seed29001|full
  bs+1_direct_de: -1.022830605506897
  bs+1_graph_de: -1.5701909065246582
  bs+1_graph_margin: 15.164186239242554
  bs+1_graph_same: 1.0
  bs+1_hh_closure: 1.0
  bs+1_mixed_acc: 1.0
  bs+1_mixed_margin: 12.898774208048387
  bs-1_direct_de: 0.020315051078796387
  bs-1_graph_de: 0.23747313022613525
  bs-1_graph_margin: -16.019729733467102
  bs-1_graph_same: 0.0
  bs-1_hh_closure: 1.0
  bs-1_mixed_acc: 0.0
  bs-1_mixed_margin: -13.808913768295765
  de_sign_flip: True
  margin_sign_flip: True

### untied|seed29001|full
  bs+1_direct_de: -0.08073782920837402
  bs+1_graph_de: 5.623885542154312
  bs+1_graph_margin: 4.065877288579941
  bs+1_graph_same: 0.75
  bs+1_hh_closure: 1.0
  bs+1_mixed_acc: 0.5
  bs+1_mixed_margin: 0.4067493650044461
  bs-1_direct_de: -0.5770552158355713
  bs-1_graph_de: 1.8680820018053055
  bs-1_graph_margin: 1.2190286964178085
  bs-1_graph_same: 0.5
  bs-1_hh_closure: 1.0
  bs-1_mixed_acc: 0.5
  bs-1_mixed_margin: 0.4067493650044461
  de_sign_flip: False
  margin_sign_flip: False

## Scientific reading

If the tied model shows graph_same_margin and graph_mean_de reversing between bs+1 and bs-1 while hh_closure and seen references remain stable, this is evidence for absolute gauge transport through the shared coordinate.  shared_trunk and untied models should not show the same coherent reversal.  anchor-only and comparison-only cells provide the factorial interaction.
