# Model Lineage

```text
Jointly pretrained encoder and first residual path
                |
Add a zero-output branch, freeze mature parameters and continue learning
                |
Qiushi-Engine-Frontier-Advancement
                |-- Ordinary continuation (two seeds)
                |-- Dense masking, dense supervision (two seeds)
                |-- Dense masking, sparse supervision (two seeds)
                +-- Sparse supervision + ordinary-input preservation (two seeds)
                         |
Qiushi-Engine-Principle-Guided-Frontier-Advancement
```

The two seeds share the same first-generation parent. Control experiments in the archive are not additional released model generations. The first generation's complete training design was not deduced in advance from principles developed later; it supplied the model, data, observations and starting point for mechanism studies.

The first generation accumulated 86,005,295 word presentations. Acquisition-only continuation adds 3,162,742 words, and the full strategy adds another 517,332 words. See [models](../models/README.md) for architecture, input budgets and evaluation identities, and [results](../results/training_strategy_comparison.csv) for the complete comparison.

Target budgets and actual save boundaries for intermediate checkpoints are recorded in each model directory's `CHECKPOINTS.json`. AoA uses the shared ancestral trajectory and each model's own endpoint; later branches must not substitute for these models' actual training histories.
