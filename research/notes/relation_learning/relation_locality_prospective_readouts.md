# Relation Locality: Predictions and Readout Design

Scientific status: historical measurement design with explicitly prospective transfer predictions.

R denotes local exact recurrence, V local nonidentical restatement, and C the no-companion control. RS and VS preserve the corresponding relation material in the corpus but separate source and companion into different records. T, U, and N denote true-source, unrelated-source, and neutral-anchor conditions.

The main design separates source-specific behavior from a generic change in prediction quality. It reports both loss contrasts Delta(T-N) and Delta(U-N), each relative to CLEAN. Positive loss differences mean worse prediction; advantage-based AT uses the opposite sign. The original three-seed loss contrasts were R-C: +0.8138 +/- 0.1729 for Delta(T-N), -0.0679 +/- 0.0340 for Delta(U-N); V-C: -0.7970 +/- 0.0758 and +0.0085 +/- 0.0536. The error values are across-seed standard deviations, not 95% confidence intervals.

The locality prediction was attenuation after splitting. Original R/V/C means used seeds 43022, 43122, and 43222, while the split comparison used 43022 and 43122. Individual seed points and the T/U decomposition are necessary: a small aggregate change can conceal simultaneous improvements in both T and U rather than retained source-specific structure.

The Entity prediction connected recurrence to unchanged-state retrieval and a cost after relevant state updates. Relevant-update depth is the number of changes to the queried entity before evaluation, not optimizer progress. The zero-update split-V anomaly was a hypothesis requiring replication, not an established general rule.

The source-output analysis distinguished probability mass assigned to source content from probability assigned to the correct transformed target. Increased source recognition need not improve that target. The budget interpretation also distinguished a compact neutral-anchor local/split gap of about 0.30 nats from an ordinary-text gap of about 0.02 nats; this motivates a narrow redistribution claim, not broad degradation.

## Prospective Natural-Transfer Predictions

On Wikipedia simplification pairs and nonoverlap targets, V-C Delta(T-N) below zero would support transfer beyond the compact format; a value near zero would restrict the restatement gain to the practiced format. R-C above zero would support transfer of the recurrence cost; a value near zero would restrict that cost. These were alternative outcomes specified before the natural-transfer readout, not guaranteed conclusions.

Evidence: [three-seed anchor measurements](../../../experiments/archive/relation_learning/data/original_threeseed_neutral_anchor/rewrite_TUN_across_seed_contrasts.csv), [split seed-level contrasts](../../../experiments/archive/relation_learning/data/split_seed_replication/rewrite_late_contrasts.csv), [two-seed neutral-anchor synthesis](twoseed_N_anchored_locality_summary.md), [ordinary-price probe](ordinary_heldout_price_probe.md).
