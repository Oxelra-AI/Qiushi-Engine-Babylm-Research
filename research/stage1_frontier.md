# Stage 1: Frontier Model Development

[Three stages](README.md) | [Original notes](notes/README.md) | [Experimental plans](plans/README.md) | [Glossary](terms.md)

This stage began with a practical limited-data training problem: within a fixed corpus pool and
cumulative word budget, how can a small language model improve its overall capabilities while
producing models and materials suitable for further investigation?
The result was the first-generation **Qiushi-Engine-Frontier-Advancement**. Its development was
not a sequence in which every added component necessarily improved the score. Data organization,
representation, and late-stage training each had independent comparisons. Some informed the
model's methods, some yielded counterexamples, and others opened questions for the next stage.

## From a Text Budget to Learnable Experience

A **source** is a selected passage of original text. A **rewrite** is a corresponding alternative
wording intended to preserve relevant content; it is also called a textual view, not an image.
Verbatim repetition presents the source again and is not equivalent to a rewrite. Correspondence
and semantic fidelity require checking: shorter text is not necessarily better information.
The **corpus-pool word count** also differs from **cumulative words presented**. Reusing the same
text in training still consumes the presentation budget. This research uses a 10M-word corpus
pool and a cumulative ceiling of 100M words; the final model may stop training before reaching
that ceiling. See the [data reconstruction guide](../data/RECONSTRUCTION.md) for the construction details.

Early representation and training comparisons examined characters, subwords, masked prediction,
autoregressive prediction, and model capacity. Whole-word masking selects a word's subword pieces
together, preventing unmasked pieces of that word from readily revealing the answer.
Compared with token-level masking, it changes the prediction task the model practices, not merely
the input format. These comparisons ultimately supported a particular encoder and whole-word
masking combination, without implying that every task benefited. The
[models and experience notes](notes/models_and_experience.md) retain these questions and results,
including those not incorporated into the representative model.

The compact-rewrite line asked a further question: could the budget spent on a second wording
also provide exposure to more sources? The method does not add free text beyond the corpus budget.
It shortens rewrites within a fixed-size replacement block, then reinvests the saved words in
additional source-rewrite pairs. The
[original reinvestment record](notes/frontier_consolidation/density_core_reinvestment_medium_riskhard.md)
distinguishes the common source core from the added pairs. The counts below come from the
[published budget table](../results/compact_budget.csv).

| Quantity within the replacement block | Actual count | What the count establishes |
| --- | ---: | --- |
| Original core / added paragraph pairs | 10,094 / 2,061 | Which materials are shared and which are added using the saved budget |
| Final paragraph pairs / source documents | 12,155 / 4,529 | Paragraph coverage is not the number of distinct documents |
| Source words / compact-rewrite words | 261,803 / 161,708 | How much of the budget each wording uses |
| Padding words / total words in the replacement block | 9 / 423,520 | Whether all text fits within the same block budget |

The approximately 20.42% increase refers to paragraph pairs within this block, not to the number
of documents in the entire corpus. The implementation must also preserve coherent text outside
the replacement; otherwise, an apparent compression benefit is confounded with damage caused by
fragmenting the background corpus. The
[row-level replacement correction](notes/frontier_consolidation/density_cleanqwen_overlay_medium_riskhard.md)
explains why an early word-stream overlay was abandoned. The
[original construction program](../experiments/archive/frontier_consolidation/scripts/materialize_density_on_cleanqwen_base_rowholdout.py)
preserves existing paired rows and complete background rows that are not removed.
Equal word counts are only a starting point: source selection, pair packing, and actual order also matter.

## Why Might Shorter Rewrites Help?

There are at least three competing explanations: compression reduces redundancy and covers more
content within the same budget; presenting a source with its own rewrite gives practice in
cross-expression correspondence; or the main benefit comes from additional repetition and source
coverage. Developed after the compact condition had shown a benefit, the
[original three-way comparison plan](plans/representation_and_objectives/compact_view_triangle_protocol.md)
proposed comparing compact rewrites, reinvestment in verbatim repetitions, and disrupted adjacency
between sources and their own rewrites. This was a mechanistic plan to explain an observed
phenomenon, not an advance prediction of every choice in the first-generation model.

A key criterion in that plan is that if disrupting adjacency also broadly harms language
performance, the gap alone cannot establish cross-expression learning. Preserving the source
and rewrite contents while removing their local correspondence provides a more targeted test
of the relationship. A fixed budget also entails removing other material, so "rewrites are
better" must be distinguished from "the displaced material was less valuable."
The [compact-view method](../methods/compact_views.md) and
[subsequent research synthesis](notes/frontier_methods.md) keep compression, coverage, adjacency,
and opportunity cost distinct.

Training stage also changes the interpretation: in matched comparisons, the compact condition
lagged early in training and gained an advantage only later. Short-run loss or a single screening
result therefore cannot assign a permanent ranking to data. The useful outcome of this line is
a reusable budget-reallocation method and its learning trajectory, not a law that shorter text
is always better.

## Continuing to Learn from a Mature Model

The representative model uses a DeBERTa-v2-style encoder with bottleneck residual branches in
each layer. A residual branch adds a learned correction to an existing representation.
Zero-output initialization leaves the original function unchanged when the branch is attached.
The first branch is trained jointly with the main model, so the base model is still changing.
Only after 82,012,495 words of exposure are the existing 35,463,008 parameters frozen and a second,
zero-output branch attached, with 995,584 new trainable parameters.
The [residual-learning method](../methods/residual_learning.md) and
[actual model code](../models/frontier/modeling_frozen_slow_private_debertav2.py) distinguish these
two stages. A small fraction of trainable parameters does not imply the same fraction of computation.

The [initial frozen-base plan](notes/frontier_consolidation/chck82_frozen_private_tail_design.md)
considered sparse auxiliary learning on paired text. The
[coherent replay trainer ultimately used](../experiments/archive/frontier_consolidation/scripts/frozen82_fastpath_replay_trainer.py)
explicitly includes an ordinary masked-prediction loss, correcting an earlier interpretation
that had treated a preservation-only objective as ordinary learning. The first generation already
used a KL preservation term to limit divergence between old and new predictive distributions;
Stage 3 did not introduce KL for the first time. Disabling the added branch can recover the
frozen base model, but whether old capabilities are preserved with the branch enabled remains
an empirical question.

The closest control for text organization fixes the starting point, trainable parameters,
exposure, and update count, changing only the coherence of within-row spans by shuffling them.
The following research seven-metric means come from the
[late-training control table](../results/late_consolidation_controls.csv). They are not the final,
complete nine-metric Overall: they include Reading but exclude SuperGLUE and AoA.

| Late-training condition | Seven-metric mean | Interpretation |
| --- | ---: | --- |
| Mature model before freezing | 43.9594 | Starting point for continued learning |
| Ordinary joint continued training | 43.7707 | Also changes the updated parameters, optimizer, and objective; not a pure text-organization control |
| Coherent residual-branch replay, branch scale 1.0 | 44.1064 | Uses 3,992,800 additional words, as does the next row |
| Residual-branch replay with shuffled within-row spans | 43.1214 | Compares coherent organization with span disruption under the same budget |

The coherent condition outperforms shuffled spans, but not every capability improves.
The [original item-level analysis](notes/frontier_consolidation/frozen_anchor_coherent_replay_item_reading.md)
records gains on some tasks and losses on others, rejecting the strong interpretation of
"universal capability gains without interference." This mechanistic limitation is compatible
with the final model being a useful endpoint.

## Outcomes and Questions for the Next Stage

The added branch was trained at scale 1.0; evaluation subsequently selected 0.75 as the release
scale. This is model selection, not an independent training replication.
The [scale comparison](../results/private_scale_sweep.csv) retains historical measurements.
The [subsequent scoring correction](notes/functional_learning/endpoint_policy_interpretation_corrections.md)
makes clear that the historical loading procedure and evaluation with the complete branch
retained are different conditions; gains must not be calculated across them.
In the current [consistent nine-metric table](../results/training_strategy_comparison.csv),
the first-generation Overall is 42.023968, displayed publicly as 42.02.
The [model package](../models/frontier/README.md) and
[training record](../models/frontier/TRAINING.md) describe the actual model after
86,005,295 cumulative words of exposure.

This stage produced a model, a paired-text construction method, and a controllable setting for
incremental learning. It also left three concrete questions: why do the same words have different
effects under different relational organization; which predictions actually use the source;
and, when performance on old examples is preserved, can new inputs still access the learned
computation? [Stage 2](stage2_principles.md) tests these questions separately.
[Stage 3](stage3_improvement.md) then uses this model as a shared parent to test training improvements.
Further architecture, objective, and curriculum branches are linked through the
[research catalog](catalog.md) and [scientific guide](scientific_guide.md); they need not all be
cast as explanations for the final model's score gains.
