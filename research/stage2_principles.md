# Stage 2: Investigating Learning Principles

[Three stages](README.md) | [Original notes](notes/README.md) | [Experimental plans](plans/README.md) | [Glossary](terms.md)

[Stage 1](stage1_frontier.md) produced a usable model, but scores alone did not explain why the
data worked. This stage separates the problem into two complementary research lines: how
relational organization in natural text changes source use, and why a learned computation can
remain available on familiar examples yet become inaccessible through new symbols.
The first varies textual relationships in training and testing; the second uses controlled
tasks that permit internal interventions. Each has scientific value independently of whether
it establishes a mechanism mediating the final model's score gains.

## What Does a Prediction Actually Gain from the Source?

A **source** is the original text providing context; a **rewrite** is its corresponding alternative
wording. Here, a **target** is a word or token position that is scored or contributes to prediction
loss, not the entire rewrite. Whether a target is "present in the source" or "absent from the source"
depends on the experiment's word- or token-matching rule. Absence does not automatically mean new
knowledge or new meaning. Verbatim repetition, aligned rewrites, and mismatched pairs therefore
cannot be treated as interchangeable forms of "more text."

The [original relation-comparison specification](notes/relation_learning/paired_context_relation_design_prestate.md)
sets out competing explanations: gains might reflect general text fitting, practice in relating
sources and targets within a shared window, or simply greater sensitivity to adjacent text.
Mismatched pairing, in particular, is not necessarily a neutral reference: it may train the
model to ignore the adjacent source. Outperforming a mismatched condition is therefore not
equivalent to outperforming ordinary training.

The measurement fixes target positions and supplies the true source T, an unrelated source U,
or length-matched neutral text N. If their target losses are L_T, L_U, and L_N, the true-source
advantage is **L_N - L_T**. A positive value means that the true source helps more than neutral
text. L_N - L_U and absolute losses in each condition must also be checked, so that a worsening
comparison condition is not mistaken for an improvement under the true source.
The unit, nats, measures natural-log loss, not leaderboard percentage points.
The [original neutral-reference analysis](notes/relation_learning/original_threeseed_neutral_anchor.md)
and [original scoring program](../experiments/archive/relation_learning/scripts/original_threeseed_neutral_anchor.py)
retain this decomposition; the [relation-learning method](../methods/relation_learning.md)
explains what the measurement means.

| Relationship practiced during training | Change in source advantage on the specified compact-rewrite targets | Standard deviation across training seeds |
| --- | ---: | ---: |
| Verbatim repetition | -0.814 nats | 0.173 |
| Aligned rewrites | +0.797 nats | 0.076 |

This is a [summary across three training seeds](../results/relation_context_use.csv).
Both changes are relative to that experiment's ordinary reference, not claims about all language
tasks. Original notes using L_T - L_N have the opposite sign; the direction must be aligned before
comparing values. The [original aggregate data](../experiments/archive/relation_learning/data/original_threeseed_neutral_anchor/rewrite_TUN_across_seed_contrasts.csv)
also retain the T, U, and N components. The standard deviation describes variation across training
replicates, not a confidence interval for an arbitrary test population.

## Exposure to Content Is Not Joint Practice of Its Relationship

If these differences arose only from exposure to repeated or rewritten text, placing the two
parts in separate windows should retain the main effects. The split-window comparison therefore
preserves each repetition or rewrite condition's own text collection and budget, while removing
co-occurrence between the source and its corresponding text within a prediction window.
The [split-window construction program](../experiments/archive/relation_learning/scripts/materialize_split_inwindow_controls.py)
and [original split-window analysis](notes/relation_learning/view_split_integration.md) document
this comparison. Content is matched between same-window and split-window versions of each
condition; repetition and rewriting do not have identical text to each other.

The later [two-seed neutral-reference summary](notes/relation_learning/twoseed_N_anchored_locality_summary.md)
adds the replication missing from the early split-window notes. Both large source effects
weaken substantially, even when general fitting of the target text can still improve.
This supports a role for relational practice within a shared prediction window.
Splitting windows also changes position and format, however, so it does not establish a law
about distance alone.

Natural-text comparisons further test the scope of the learned relationships.
The [original aligned, mismatched, repeated, and separately presented results](notes/relation_learning/paired_context_relation_design_probe.md)
show that aligned rewrites better help the model use source-present content after a change in
sentence structure, but do not broadly improve replacement targets absent from the source.
Verbatim repetition can also be useful when the same surface form recurs.
The result is therefore selective alignment between the relationship practiced in training
and the relationship required at test time, not a dichotomy in which "repetition is harmful
and rewriting is beneficial."

These measurements also require an exposure-scope check.
The [subsequent overlap correction](notes/functional_learning/overlap_scope_and_model_identity_corrections.md)
found that some nominally held-out material had already appeared in the training stream,
narrowing the affected generalization claims to fitting familiar text.
This is neither evidence that all official evaluations were contaminated nor a record of
cleaning completed before model training. Measurements from different target categories and
probe sets cannot simply be combined into one transfer score.

## Can Unseen Symbols Still Access the Computation Used on Familiar Examples?

The second research line uses four-choice entity-attribute tasks to distinguish "knowing which
attributes occur in the context" from "selecting the queried entity's attribute."
The [initial permutation-task plan](notes/functional_learning/orbit_binding_design.md) preserves
the same word collection while swapping entity-attribute bindings, ruling out a bag-of-words
solution. The [original results and subsequent interpretation correction](notes/functional_learning/orbit_binding_result.md)
found that, under ordinary objectives, the model mainly learned the attribute set. This alone does not establish
insufficient model capacity or a particular optimization failure mechanism.

The [follow-up supervision-allocation results](notes/functional_learning/loss_allocation_result.md)
showed that simply increasing the answer weight did not automatically produce binding either,
motivating a design that places the query before the context.
The [query-first experiment](../experiments/archive/functional_learning/scripts/query_first_binding.py)
and [original per-seed results](notes/functional_learning/query_first_binding_compact_summary.md)
then established successful learning under answer-focused training, while retaining seed-level
differences in transfer to unseen symbols. Subsequent studies could therefore ask how an acquired
capability is preserved, rather than mistaking failure to learn for forgetting.

**Functional reach** is the range of objects, queries, and symbols that can access a computation,
not accuracy on familiar examples. Starting from two training seeds with strong transfer,
the follow-up compares ordinary all-target continued training, static supervision allocation
that limits non-answer learning pressure, and alternation between answer-only and all-target
training. The [functional-access method](../methods/functional_access.md) describes the comparison.
The table below gives two-seed means on the controlled task from the
[consistent results table](../results/interface_reach.csv).

| State or continued-training strategy | Familiar-symbol accuracy | Unseen-query-symbol accuracy |
| --- | ---: | ---: |
| Starting point after capability acquisition | 100.0% | 87.5% |
| Ordinary all-target continued training | 87.2% | 40.2% |
| Static relational supervision allocation | 100.0% | 75.5% |
| Alternating answer-only / all-target training | 100.0% | 83.3% |

Behavioral gaps still require functional evidence. The study replaces an internal matching
signal with one produced by a different query and tests whether the answer changes in the
intended direction. It then erases, centers, or rotates that signal across attribute positions
during an ordinary forward pass to test whether the original answer depends on it.
The [original intervention notes](notes/functional_learning/clean_d_component_ablation.md) and
[intervention program](../experiments/archive/functional_learning/scripts/clean_d_component_ablation.py)
show that this is not merely correlated information decodable by a probe: ordinary selection
does depend on the positional matching pattern. The rate of selecting the designated option
after rotation is an intervention response, not accuracy on the original question.

The [subsequent functional-reach synthesis](notes/functional_learning/interface_reach_synthesis.md)
also separates signal formation from downstream use, limiting the interpretation that all
damage occurs at a single site. The static weight 1/17 matches coefficients under this task's
layout; it is not a universally optimal value. Nor do these interventions establish that the
final language model necessarily uses exactly the same internal computation.

## What Carries Forward?

Together, the two research lines pose actionable questions: which evidence is visible during
prediction, which targets apply learning pressure to its use, and which new inputs can still
access the acquired function after continued learning? They support designing input,
supervision, and preservation separately, and measuring both gains and costs.
Neither adding more targets nor recovering familiar-example performance is sufficient on its own.

A separate early small-scale entity-memory experiment withdrew its binding interpretation because
the answer did not depend on the queried entity. The
[correction on the original record](notes/representation_and_objectives/entity_memory_miniscreen_synthesis.md)
still applies: correcting the data and program is not the same as completing successful retraining.
This does not invalidate the independent given-address experiments or the functional interventions above.

[Stage 3](stage3_improvement.md) applies only the implementable input, supervision, and preservation
designs to the first-generation model, then tests them on real text and complete evaluation.
It does not combine every mechanistic branch into one training method.
Independent results and unfinished directions in target deletion, sparse anchoring, addressing,
representation, and data substitution remain linked through the
[research catalog](catalog.md) and [scientific guide](scientific_guide.md).
