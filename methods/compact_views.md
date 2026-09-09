# Compact Restatements and Budget Reallocation

Pair a source passage with a shorter restatement of similar meaning. A "view" here is another textual expression of the same content, not an additional image input. Under a fixed word budget, shorter restatements allow more source material to be included. Source words, generated words, pair counts and cumulative training presentations must therefore be recorded separately.

The designated compact block contains 12,155 source-restatement pairs totaling 423,511 words, with another 9 words completing the block budget. It accommodates 2,061 more pairs than the longer-restatement condition. Early and late performance rankings differ, so data selection cannot rely solely on short-run loss rankings.

The [budget table](../results/compact_budget.csv) retains the numerical results; the [fixed data manifest](../models/frontier/DATA_MANIFEST.json) records the full corpus composition and fingerprints; the [data guide](../data/README.md) distinguishes source text, the generation teacher, selected text and presentation budgets.

These fixed-budget comparisons test specific compression and reallocation strategies. They do not establish that stronger compression is always better or provide an efficiency multiplier valid at arbitrary scales.
