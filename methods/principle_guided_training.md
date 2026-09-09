# Separating Input, Supervision and Preservation

The method uses existing source-restatement text and keeps the source visible. It increases masking within the restatement while holding selected sparse target positions fixed. Masking more words can reduce local cues without making every masked word a prediction-loss target. Ordinary text continues to use whole-word masking.

The acquisition objective combines two separately averaged losses: focused-target cross-entropy has weight 0.15, and ordinary-target cross-entropy has weight 0.85. It does not pool all targets into a single mean, and 0.15 is not a fraction of the word budget.

The full strategy adds an ordinarily masked presentation of paired rows and constrains the corresponding target distributions using the frozen first-generation model. The divergence is KL(parent || student), with coefficient and temperature both equal to 1. Preservation computation disables dropout and restores the random-number state to avoid changing the acquisition branch's random sequence.

Only the added residual branch is updated. The 80 updates consume 3,162,742 acquisition words; the full strategy adds 517,332 preservation word presentations. Including the parent's 86,005,295 words, the total is 89,685,369.

## What the Comparisons Establish

Ordinary continuation, dense masking with sparse supervision, dense masking with dense supervision, and the full strategy all have nine-component endpoint evaluations. Sparse masking with sparse supervision has separate Fast and mechanism measurements but is absent from the current full nine-component table. Ordinary continuation cannot substitute for that single-factor control.

A common-target diagnostic finds an approximately 11.835-fold difference between preservation gradient norms on densely masked and ordinary inputs. Equal coefficients therefore do not imply equal effective constraints, and the current comparisons cannot attribute the entire effect to the input condition used for preservation. Full preservation also adds presentations and computation.

- [Experiment configuration](../experiments/configs/stage3.json)
- [Full nine-component evaluation](../results/training_strategy_comparison.csv)
- [Gradient diagnostic](../results/preservation_gradient_diagnostic.csv)
- [Model package method description](../models/principle_guided/METHOD.md)
- [Detailed discussion and equations](../reports/zh/chapters/05_guided_improvement.tex)
