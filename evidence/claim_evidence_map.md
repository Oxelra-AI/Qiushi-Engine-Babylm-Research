# Main Findings and Experimental Evidence

This page connects research conclusions to specific materials. Detailed experimental interpretation remains in the report.

| Finding | Direct evidence | Supported scope |
| --- | --- | --- |
| Compact restatements allow more sources within a fixed budget | [Budget results](../results/compact_budget.csv), [method](../methods/compact_views.md) | Text and pair counts in the specified replacement block; not a claim that all compression is more effective |
| Repetition and restatement change source use for different targets | [Relation experiments](../results/relation_context_use.csv), [Chapter 3](../reports/zh/chapters/04_principles.tex) | Specified relations, targets and windows; each same-material comparison contrasts shared and split windows |
| Recovery on familiar inputs can dissociate from functional access on unseen queries | [Functional interventions](../results/interface_reach.csv), [interpretation](../methods/functional_access.md) | Specific controlled tasks and internal signals; not a complete causal account of the final language model |
| Changing the learning strategy can outperform ordinary continuation | [Full nine-component controls](../results/training_strategy_comparison.csv), [training design](../methods/principle_guided_training.md) | Two continuation seeds from the same parent; acquisition-only comparisons have equal cumulative exposure |
| The full strategy with ordinary-input preservation further improves Overall | The same [full nine-component table](../results/training_strategy_comparison.csv) | Preservation adds presentations and computation, so the gain cannot be attributed entirely to KL |
| The effective strength of preservation must be measured | [Common-target gradient diagnostic](../results/preservation_gradient_diagnostic.csv) | Gradient strengths differ at equal coefficients; the comparison does not isolate the input condition alone |

## Corrected Interpretations

An early small-scale entity-memory experiment was affected by an answer shortcut independent of the query. Its compositional-binding interpretation has been withdrawn; see catalog entry R36. Other entity read/write experiments retain their own data and task conditions. The withdrawn 98–99% figures are not reused as new evidence.

The first generation already used output-preservation constraints. The second generation contributes a new configuration of inputs, supervision and preservation, not the first proposal of knowledge preservation. The main principle is a design principle developed through continuing research and tested in part through actual training. It is not retroactively presented as a universal law fully formulated before the first generation.
