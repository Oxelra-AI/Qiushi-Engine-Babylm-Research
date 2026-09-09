# causal gauge experimental design causal gauge-transport probe

Fixed aligned arm, train-only vocabulary, dropout=0, paired initialization.

## Per-run central readout

| condition | bridge_sign | cell_tag | seed | train_state_acc | train_cmp_acc | direct_same | graph_same | pair_both_graph_same | unchanged | mixed_acc | mixed_margin | hh_closure | graph_same_margin | graph_mean_de |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tied | -1 | bs-1 | 29002 | 1.0000 | 0.8750 | 0.0000 | 0.2500 | 0.2500 | 1.0000 | 0.1250 | -6.7419 | 0.6250 | -5.7293 | 6.3148 |

## Bridge-sign paired comparisons

## Scientific reading

If the tied model shows graph_same_margin and graph_mean_de reversing between bs+1 and bs-1 while hh_closure and seen references remain stable, this is evidence for absolute gauge transport through the shared coordinate.  shared_trunk and untied models should not show the same coherent reversal.  anchor-only and comparison-only cells provide the factorial interaction.
