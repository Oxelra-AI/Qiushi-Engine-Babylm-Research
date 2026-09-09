# compact mixture model and predictions compact mixture model for relation-practice learning

This note formulates the smallest mechanism that currently fits the measured BabyLM evidence and gives exact predictions for the remaining split and natural-variation tests.

## Measured ingredients now fixed

1. **Original exact in-window recurrence installs a source-specific cost.** On held-out compact-rewrite token-nonoverlap targets, original REPEAT vs CLEAN has negative gain with an isolating T/U sign reversal across DeBERTa seeds: true-source NLL is worse but unrelated-source NLL is better. At seed43022, R−C gain is −0.7517, T delta +0.4480, U delta −0.3037.

2. **Removing same-window co-occurrence removes the source-specific cost.** REPEAT_SPLIT preserves the selected source/repeat tokens and 100M budget but places source and exact companion in separate rows. At seed43022, RS−C token-nonoverlap gain is −0.0313, with both terms improved (T −0.5469, U −0.5782). Pair-level and strict-word analyses keep RS−C near zero rather than original-R-like: pair-level gain −0.0470, excess true-source cost +0.0470; strict-word gain −0.0964, excess +0.0964.

3. **The source-token mass shift is specific to the recognized true source.** The content-token 2×2 source-specificity control gives R−C true-source content-mass deltas of +0.0727, +0.0972, +0.0550 under true-source windows, but only +0.0054, +0.0083, +0.0050 under unrelated-source windows. Target probability is suppressed only under the true-source condition. This rules out a pure frequency-hedging explanation.

4. **Spaced repetition improves ordinary content fit relative to in-window exact recurrence.** Comparing REPEAT_SPLIT directly with original REPEAT at seed43022, with the same selected tokens and budget, RS is much better on held-out nonidentical compact rewrites: true-source NLL lower by 0.9949 and unrelated-source NLL lower by 0.2745. On the held-out natural-copy probe's unrepeated-control term, RS has lower NLL than original R (3.9918 vs 4.3896). This is the inherited residual-prediction-work idea in a concrete window form: an exact same-window copy makes repeated-row masks too easy and reduces the ordinary content gradient; spaced repetition preserves exposure without the local shortcut.

5. **Raw copy gain must be normalized.** Original R−C copy gain is +0.4996 raw and +0.0497 after gain/control normalization. RS−C is −0.2005 raw but only −0.0307 normalized because RS has lower unrepeated-control NLL. The safe reading is that the large original REPEAT copy advantage requires in-window co-occurrence; not that split repetition intrinsically has worse copy ability than CLEAN.

6. **VIEW uses the source differently.** VIEW increases true-source content mass under the true-source condition but also increases target probability. Original REPEAT routes mass toward source tokens and away from the correct nonidentical target; VIEW routes through recognized source content to predict the nonidentical target. VIEW also retains intermediate natural-copy gain.

## Compact hypothesis: recognized-span readout is a mixture

For a masked target in a window containing a related span `s`, write the model's context-dependent readout as a mixture of three components:

\[
P_\theta(y\mid s, c) \approx \alpha(o,\tau) P_{\mathrm{id}}(y\mid s) + \beta(o,\tau) P_{\mathrm{content}}(y\mid s,c) + \gamma P_{\mathrm{base}}(y\mid c),
\]

where:

- `o` is the surface overlap of the practiced source--companion relation inside training windows;
- `\tau` summarizes whether the evaluation context is recognized as the same relation type;
- `P_id` places mass on tokens or surface forms in the source span;
- `P_content` uses the source span as evidence for a nonidentical but semantically/correspondentially related target;
- `P_base` is ordinary context/language modeling not specifically routed through the related span.

Training changes the mixture weights and the quality of the components:

- **Exact in-window recurrence (`o≈1`)** increases `\alpha` strongly and gives little pressure to improve `P_content` for transformed/nonidentical targets. At test time this helps exact copying and unchanged-state retrieval, but on token-nonoverlap targets `P_id` steals probability mass from the correct target.
- **Partial-overlap restatement (`0<o<1`, with some changed content)** increases `\beta` while retaining a moderate `\alpha` on overlapping tokens. This explains why VIEW can exceed CLEAN on nonoverlap source use and still exceed CLEAN on copy gain.
- **No same-window related span** leaves `\alpha,\beta` near their baseline triggered levels; tokens may still improve `P_base` if repeated across rows, especially when residual prediction work is preserved.

The model is intentionally phenomenological. The current evidence establishes the output-level components and their source-specific trigger; it does not yet identify the internal circuit (attention head, value path, MLP/readout coupling, or representation geometry).

## Predictions and readings for remaining work

### VIEW_SPLIT seed43022

- If VS−C token-nonoverlap rewrite gain moves near zero and true-source advantage disappears, then both identity and content readouts require in-window relation practice. The window is the unit of reusable relation learning.
- If VS−C remains in the original +0.684 to +0.894 band, then content readout can be learned from cross-row paraphrase exposure, while identity readout/cost is specifically local.
- If intermediate, the mixture weights have both corpus-level and window-level components.

### Second split seeds

REPEAT_SPLIT seed43122 should reproduce the elimination of the R−C sign reversal. The most important quantity is not the raw gain alone but the term pattern: RS−C should not show original-R-like T worse and U better. VIEW_SPLIT seed43122 should be launched when a GPU is free.

### Natural variation-set probe

For adjacent caregiver/spoken utterances binned by surface overlap between source utterance and target utterance:

- On exact or near-exact overlap targets, identity readout should help; REPEAT-like arms should be strong.
- On nonoverlap content targets inside partial-overlap pairs, content readout should help; VIEW should outperform REPEAT and often CLEAN.
- The largest REPEAT deficit should occur at high enough overlap to trigger source recognition but with target tokens not present in the source: this is where `\alpha` fires while the correct answer requires `\beta`.
- At very low overlap, neither readout is strongly triggered, so arm differences should shrink toward ordinary LM fit.

This is the natural-language, register-free version of the principle: data efficiency comes from arranging related experiences within the learner's context window at an overlap that makes identity insufficient but correspondence learnable.

## Training principle implied if VIEW_SPLIT is local

Under a finite data and compute budget, repeated exposure is not a scalar good. Exact same-window repetition can be worse than spaced repetition because it both installs a copy shortcut and removes residual prediction work from repeated rows. To form reusable, compositional, transferable knowledge, training data should place related spans close enough for the learner to compare them, but with enough controlled difference that the lowest-loss relation is content correspondence rather than identity copying. In practice: prefer in-window restatement or variation sets with partial overlap over exact in-window duplication.

## Evidence files

- `research/notes/relation_learning/locality_residual_work.md`
- `research/notes/relation_learning/repeat_split_robustness.md`
- `research/notes/relation_learning/source_specificity_misfire.md`
- `research/notes/relation_learning/relation_practice_principle.md`
- `research/notes/relation_learning/split_control_numeric_prestatement.md`
