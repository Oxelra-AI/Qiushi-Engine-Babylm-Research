# Method and Research Connection

## Frontier Advancement

The corpus pairs some source passages with concise rewrites that preserve their
meaning. Shorter rewrites leave room for more source passages within the fixed word
budget. A randomly initialized DeBERTa-v2 encoder is trained with 15% whole-word
masking, which selects words and masks their constituent tokens together.

Small residual adapters are added to the encoder. Each adapter projects a hidden
representation to a lower-dimensional space and back, then adds its output to the
original representation. A second adapter is initialized with zero output, leaving
the model's predictions unchanged at attachment. Training this adapter produces
the Stage I model.

## Principle Discovery

Controlled studies examined what models learn from paired texts, which contextual
information they use, and how later training affects previously learned behavior.
They motivated two design choices: separating the words hidden from the model from
the positions used to compute prediction loss, and combining the new training
objective with a loss that preserves the parent model's predictions.

## Principle-Guided Frontier Advancement

For selected paired rows, more words are masked in the rewrite, while only a subset
of positions contribute to the prediction loss. The source passage remains visible.
This reduces clues available within the rewrite and encourages use of the source.
Other rows use ordinary whole-word masking.

The acquisition loss combines the mean cross entropy of paired-view targets with
the mean cross entropy of ordinary targets, weighted 0.15 and 0.85 respectively.
Paired rows also receive an ordinary-mask presentation. On these inputs, the
Kullback-Leibler divergence KL(parent || student) penalizes changes from the frozen
parent's output probabilities, with coefficient 1 and temperature 1. Dropout is
disabled for this preservation calculation. Random-number-generator state is
restored around teacher setup and preservation calculations to keep the acquisition
dropout sequence unchanged. Only the second adapter's 995,584 parameters are updated.

## Interpretation

Both continuation seeds improve Overall over ordinary continuation from the same
parent. The preservation comparison changes two things together: ordinary-input
presentations and KL matching. It therefore measures their combined effect;
the contribution of KL alone remains unresolved. The tested setting is English
BabyLM Strict-Small with the shared parent model. The companion research report
provides the detailed experiments and analysis.
