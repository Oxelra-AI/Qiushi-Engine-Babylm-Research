# compact mixture model and predictions locality × residual-prediction-work readout

This analysis reuses the scored held-out copy and compact-rewrite rows from probe results interpretation and relation practice principle. No model forward passes are run. It asks what REPEAT_SPLIT means relative to original REPEAT, not only relative to CLEAN.

## 1. In-window recurrence reduces learning from the repeated content itself

Seed43022 token-nonoverlap compact-rewrite terms, averaged over 80M/90M/100M:

| contrast | gain Δ (U−T) | true-source NLL Δ | unrelated-source NLL Δ | reading |
|---|---:|---:|---:|---|
| original R−C | -0.7517 | +0.4480 | -0.3037 | original in-window exact recurrence: true source becomes worse while unrelated source improves |
| REPEAT_SPLIT−C | -0.0313 | -0.5469 | -0.5782 | same repeated tokens across rows: no source-specific cost, both terms improve |
| REPEAT_SPLIT−original R | +0.7204 | -0.9949 | -0.2745 | same selected tokens/budget but no in-window copy: true-source NLL is about one nat lower and unrelated-source NLL is also lower |

The RS−R true-source improvement is the new residual-work signal. The exact same selected source and companion tokens are more useful when they are spaced into separate rows than when an exact companion sits in the same masked-LM window. In-window identity supplies an easy copy route for masked tokens, so it exhausts residual prediction work in those rows and weakens ordinary content learning; cross-row repetition preserves exposure without creating the local shortcut.

## 2. Copy gain should be normalized by each arm's own control level

Held-out natural-copy terms over never-trained rows, averaged over 80M/90M/100M:

| arm | raw copy gain | normalized gain = gain/control NLL | repeated-source NLL | unrepeated-control NLL |
|---|---:|---:|---:|---:|
| C | +3.6869 | +0.9040 | +0.3913 | +4.0782 |
| R | +4.1864 | +0.9537 | +0.2032 | +4.3896 |
| RS | +3.4863 | +0.8734 | +0.5054 | +3.9918 |
| V | +4.0140 | +0.9131 | +0.3820 | +4.3960 |

Contrasts:

| contrast | raw gain Δ | normalized gain Δ | repeated NLL Δ | control NLL Δ |
|---|---:|---:|---:|---:|
| RminusC | +0.4996 | +0.0497 | -0.1881 | +0.3115 |
| RSminusC | -0.2005 | -0.0307 | +0.1141 | -0.0864 |
| RSminusR | -0.7001 | -0.0804 | +0.3022 | -0.3979 |

The raw RS−C copy-gain contrast is negative, but this should not be read as lower copy ability alone: RS has a lower unrepeated-control NLL than C, leaving less room for a nat-scale gain. After normalization RS remains below original R by about 0.080 of its control level, while RS is only slightly below C (about −0.031). Thus the strong statement is not that split repetition destroys copy; it is that the original REPEAT copy gain requires in-window co-occurrence, while cross-row repetition primarily improves ordinary control fit rather than installing a large local copy advantage.

## 3. Mechanistic implication

Exact recurrence inside a window has two separable costs under a fixed budget: (i) it installs recognized-source identity routing, which creates the T/U sign reversal and source-token misfire on nonidentical targets; and (ii) it reduces residual prediction work available for ordinary content learning, because the same-window exact copy lets masked tokens be solved locally. Spaced cross-row repetition of the same tokens at this dose preserves the content-exposure benefit and removes the in-window shortcut. This reconnects the current BabyLM mechanism to the inherited frontier_consolidation residual-work principle in a concrete training-window form.

Data outputs: `experiments/archive/relation_learning/data/locality_residual_work`.
