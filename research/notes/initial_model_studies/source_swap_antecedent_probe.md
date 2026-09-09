# source swap antecedent probe — source-swap antecedent-dependence probe

Model: protected 100M DeBERTa (`chck_100M`). Probe examples: 540.

Target tokens are identical across all three conditions; only the earlier antecedent changes.

| condition | mean target LL |
|---|---:|
| A correct antecedent visible | -2.3757 |
| B antecedent swapped | -4.8201 |
| C antecedent masked | -4.4782 |

A - B: mean 2.4445, median 0.8653, positive fraction 0.9315
A - C: mean 2.1025, median 0.7551, positive fraction 0.9463

If A - B and A - C are near zero, the later target does not depend on the correct antecedent, and anchored-masking training would not localize cross-mention binding. If clearly positive, a target-identical anchored/control training is justified.
