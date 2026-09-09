# Relation Learning: Matching Training and Testing

The same information can be repeated verbatim or restated in different words. The experiments distinguish the source-target relation during training from the relation needed at test time. Measurements compare target prediction loss with correct, neutral and unrelated sources. Source advantage is measured in nats, not leaderboard percentage points.

In the [core results](../results/relation_context_use.csv), verbatim repetition and aligned restatement change source advantage in different directions for compact-restatement targets whose token IDs do not occur in the true source. This is a tokenizer-level definition, not a test of whether a whole word or its meaning is new. Error bars show standard deviations across three training seeds. Each condition has a split-window control using the same material to test whether the related content must be jointly available during prediction.

Tests on English Wikipedia and Simple English restatements use the same token-ID distinction. Aligned restatement improves source use for targets recurring in the source under a different sentence form; targets absent from the source show no equally stable improvement over the reference. Relation selectivity means that training beneficial for one target class need not help another.

Source interventions on the final model establish prediction changes only under the specified source conditions. Greater source sensitivity does not automatically imply more correct reasoning, let alone a general solution to in-context learning.

See [Chapter 3 of the report](../reports/zh/chapters/04_principles.tex) for detailed conditions, natural-text results and direct prior work. A separate source-disjoint experiment is recorded in the [target-loss table](../results/source_disjoint_target_loss.csv); the measurement sets must not be combined.
