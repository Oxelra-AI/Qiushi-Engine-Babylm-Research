# causal gauge experimental design causal gauge-transport probe

Fixed aligned arm, train-only vocabulary, dropout=0, paired initialization.

## Per-run central readout

| condition | bridge_sign | cell_tag | seed | train_state_acc | train_cmp_acc | direct_same | graph_same | pair_both_graph_same | unchanged | mixed_acc | mixed_margin | hh_closure | graph_same_margin | graph_mean_de |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tied | 1 | bs+1_noanchor | 29000 | 1.0000 | 0.8750 | 0.0000 | 0.2500 | 0.2500 | 1.0000 | 0.1250 | -3.9004 | 0.6250 | -2.6918 | 6.5436 |
| tied | -1 | bs-1_noanchor | 29000 | 1.0000 | 0.8750 | 0.0000 | 0.2500 | 0.2500 | 1.0000 | 0.1250 | -3.9004 | 0.6250 | -2.6918 | 6.5436 |

## Bridge-sign paired comparisons

### tied|seed29000|_noanchor
  bs+1_direct_de: 0.16165964305400848
  bs+1_graph_de: 6.5435825958848
  bs+1_graph_margin: -2.6918041929602623
  bs+1_graph_same: 0.25
  bs+1_hh_closure: 0.625
  bs+1_mixed_acc: 0.125
  bs+1_mixed_margin: -3.9003819618975286
  bs-1_direct_de: 0.16165964305400848
  bs-1_graph_de: 6.5435825958848
  bs-1_graph_margin: -2.6918041929602623
  bs-1_graph_same: 0.25
  bs-1_hh_closure: 0.625
  bs-1_mixed_acc: 0.125
  bs-1_mixed_margin: -3.9003819618975286
  de_sign_flip: False
  margin_sign_flip: False

## Scientific reading

If the tied model shows graph_same_margin and graph_mean_de reversing between bs+1 and bs-1 while hh_closure and seen references remain stable, this is evidence for absolute gauge transport through the shared coordinate.  shared_trunk and untied models should not show the same coherent reversal.  anchor-only and comparison-only cells provide the factorial interaction.
