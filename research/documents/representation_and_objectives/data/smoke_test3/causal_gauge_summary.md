# causal gauge experimental design causal gauge-transport probe

Fixed aligned arm, train-only vocabulary, dropout=0, paired initialization.

## Per-run central readout

| condition | bridge_sign | cell_tag | seed | train_state_acc | train_cmp_acc | direct_same | graph_same | pair_both_graph_same | unchanged | mixed_acc | mixed_margin | hh_closure | graph_same_margin | graph_mean_de |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tied | 1 | bs+1 | 29000 | 0.9722 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 1.0000 | 0.5000 | -0.0000 | 0.5000 | -0.0011 | -0.0011 |
| shared_trunk | 1 | bs+1 | 29000 | 0.9722 | 0.6250 | 0.5000 | 0.5000 | 0.5000 | 1.0000 | 0.3750 | -0.0001 | 0.3750 | -0.0011 | -0.0011 |
| untied | 1 | bs+1 | 29000 | 0.9722 | 0.6250 | 0.5000 | 0.5000 | 0.5000 | 1.0000 | 0.3750 | -0.0000 | 0.6250 | -0.0011 | -0.0011 |
| tied | -1 | bs-1 | 29000 | 0.9861 | 0.6250 | 0.2500 | 0.5000 | 0.5000 | 1.0000 | 0.3750 | -0.0001 | 0.3750 | -0.0014 | -0.0010 |
| shared_trunk | -1 | bs-1 | 29000 | 0.9861 | 0.6250 | 0.2500 | 0.5000 | 0.5000 | 1.0000 | 0.2500 | -0.0001 | 0.6250 | -0.0015 | -0.0010 |
| untied | -1 | bs-1 | 29000 | 0.9861 | 0.6250 | 0.2500 | 0.5000 | 0.5000 | 1.0000 | 0.3750 | -0.0000 | 0.6250 | -0.0014 | -0.0010 |

## Bridge-sign paired comparisons

### shared_trunk|seed29000|full
  bs+1_direct_de: -0.0053506698459386826
  bs+1_graph_de: -0.0010635508224368095
  bs+1_graph_margin: -0.0010935897007584572
  bs+1_graph_same: 0.5
  bs+1_hh_closure: 0.375
  bs+1_mixed_acc: 0.375
  bs+1_mixed_margin: -6.744265338888707e-05
  bs-1_direct_de: -0.0073342351242899895
  bs-1_graph_de: -0.0009584305807948112
  bs-1_graph_margin: -0.001451767049729824
  bs-1_graph_same: 0.5
  bs-1_hh_closure: 0.625
  bs-1_mixed_acc: 0.25
  bs-1_mixed_margin: -9.214505373426367e-05
  de_sign_flip: False
  margin_sign_flip: False

### tied|seed29000|full
  bs+1_direct_de: -0.005343849770724773
  bs+1_graph_de: -0.001053420826792717
  bs+1_graph_margin: -0.0010867193341255188
  bs+1_graph_same: 0.5
  bs+1_hh_closure: 0.5
  bs+1_mixed_acc: 0.5
  bs+1_mixed_margin: -1.627299965945541e-05
  bs-1_direct_de: -0.007339730858802795
  bs-1_graph_de: -0.0009507378563284874
  bs-1_graph_margin: -0.00144937913864851
  bs-1_graph_same: 0.5
  bs-1_hh_closure: 0.375
  bs-1_mixed_acc: 0.375
  bs-1_mixed_margin: -6.659235808006717e-05
  de_sign_flip: False
  margin_sign_flip: False

### untied|seed29000|full
  bs+1_direct_de: -0.005334554240107536
  bs+1_graph_de: -0.0011072345077991486
  bs+1_graph_margin: -0.0010825376957654953
  bs+1_graph_same: 0.5
  bs+1_hh_closure: 0.625
  bs+1_mixed_acc: 0.375
  bs+1_mixed_margin: -3.603287130331033e-05
  bs-1_direct_de: -0.007326873019337654
  bs-1_graph_de: -0.0009960709139704704
  bs-1_graph_margin: -0.0014341427013278008
  bs-1_graph_same: 0.5
  bs-1_hh_closure: 0.625
  bs-1_mixed_acc: 0.375
  bs-1_mixed_margin: -3.603287130331033e-05
  de_sign_flip: False
  margin_sign_flip: False

## Scientific reading

If the tied model shows graph_same_margin and graph_mean_de reversing between bs+1 and bs-1 while hh_closure and seen references remain stable, this is evidence for absolute gauge transport through the shared coordinate.  shared_trunk and untied models should not show the same coherent reversal.  anchor-only and comparison-only cells provide the factorial interaction.
