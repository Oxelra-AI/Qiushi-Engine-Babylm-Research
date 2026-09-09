# BabyLM Scientific Research Guide

[Selected original notes and plans](reading_paths.md) · [All research notes](notes/README.md) · [Experiment plans](plans/README.md) · [Terminology](terms.md) · [Topic evidence](materials.md)

This research asks how a language model can learn more from a limited amount of text: which representations make experience usable, which relationships training actually rewards, and which acquired functions survive further learning. The program connects three scientific stages: building a competitive model, investigating the conditions behind its behavior, and using those findings to improve the same model through a specified continuation policy.

The [research catalog](catalog.md) organizes 74 topics with different evidential status: completed experiments, conditional findings, unsuccessful constructions, a withdrawn interpretation and untested proposals. Controlled tasks, data constructors and evaluation corrections have scientific value even when they do not improve the final model. The [topic index](topics.json) connects each question to its methods and evidence.

## Questions, Methods, and Evidence

The links below identify representative comparisons for each research family. Each result concerns its stated training conditions, evaluation population and measurement unit.

| Topics and scientific question | Method or analysis | Example evidence |
| --- | --- | --- |
| R01-R09: how does the text/model interface affect learning? | [Backbone, tokenizer and masking comparisons](notes/models_and_experience.md#establishing-the-baseline) | [Paired masking results](../experiments/archive/initial_model_studies/data/masked_1m_grid_pos512_profile.json) |
| R10-R14: what does a second expression contribute at fixed budget? | [Compact views and reinvestment](../methods/compact_views.md) | [Pair and word counts](../results/compact_budget.csv) |
| R15-R23: which auxiliary targets improve the complete model? | [Objective-combination tests](notes/models_and_experience.md#objective-and-optimization-boundaries) | [Completed causal-auxiliary evaluation](../experiments/archive/compact_experience/data/mlm_mntp_full_eval/per_target/mlm_mntp_aux015_100M.json) |
| R24-R30: how do effects depend on learning stage? | [Ordering, masking and visible exposure](notes/models_and_experience.md#selection-ordering-and-visible-exposure) | [Masking trajectories](../experiments/archive/compact_experience/data/curriculum_100M_eval/trajectory_compact_table.csv) |
| R31-R34: what can a frozen-parent increment acquire and retain? | [Residual learning](../methods/residual_learning.md) | [Late continuation controls](../results/late_consolidation_controls.csv), [scale effects](../results/private_scale_sweep.csv) |
| R35-R38: does supplied-address computation imply learned routing? | [Memory and causal use](notes/models_and_experience.md#relations-memory-and-causal-use) | [Supplied versus predicted addressing](../experiments/archive/initial_model_studies/data/unlabeled_address_decisive.json) |
| R39-R40: is the difficulty reading a format or selecting a record? | [Selection and reading](#selection-and-reading) | [Frozen reader and group selector](../experiments/archive/representation_and_objectives/data/selector_reader_groupsoftmax_seed27000/selector_reader_summary.json) |
| R41-R44: when can identity and sparse anchors support relation transfer? | [Graphs, memory and interfaces](notes/frontier_methods.md#graphs-memory-and-interfaces) | [Multiseed graph and affine readouts](../experiments/archive/representation_and_objectives/data/multiseed_gauge_affine_full/multiseed_gauge_and_affine_summary.json) |
| R45-R49: how does displaced material change data value? | [Budget and displacement controls](notes/frontier_methods.md#mechanism-tests-and-measurement) | [Register contrasts across seeds](../experiments/archive/relation_learning/data/frontier_consolidation_readout/register_seed_contrasts.csv) |
| R50-R52: what is learned from source-absent versus copied targets? | [Target and source interventions](#targets-and-sources) | [Disjoint-target losses](../results/source_disjoint_target_loss.csv), [target deletion](../results/target_deletion_100m.csv) |
| R53-R59: is an accessible state used by prediction? | [Functional mechanism tests](#functional-mechanisms) | [Counterfactual micro-world readout](../experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3_scores/micro_world_v3_deeper_interpretation.json) |
| R60-R66: which measurement conditions change attribution? | [Initialization and measurement controls](notes/frontier_methods.md#mechanism-tests-and-measurement) | [Common-initialization interaction](../experiments/archive/frontier_consolidation/data/commoncopy_architecture_interaction_four_cell_bootstrap/architecture_interaction_four_cell_bootstrap.json) |
| R67-R71: what relation is acquired, and which inputs can use it? | [Relation learning](../methods/relation_learning.md), [functional access](../methods/functional_access.md) | [Source-use contrasts](../results/relation_context_use.csv), [intervention results](../results/interface_reach.csv) |
| R72-R74: can new source use coexist with ordinary prediction? | [Principle-guided training](../methods/principle_guided_training.md) | [Complete policy comparison](../results/training_strategy_comparison.csv), [preservation gradients](../results/preservation_gradient_diagnostic.csv) |

## Models and Representations

[R01-R09](catalog.md) examine the interface between text and the model. Sparse routing and prefix memory were compared with dense capacity references; character and n-gram pathways were compared with lookup controls. Neither line established a stable, broad advantage in the tested settings. These results constrain particular implementations, not the usefulness of sparsity, memory or morphological structure in general.

Whole-word masking changed the task profile relative to token masking. BERT/DeBERTa-style comparisons, with additional batch controls, supported a stronger combined recipe but did not isolate the contributions of individual positional-attention terms. Recursive autoregressive training supplied a separate reproduction and evaluation study. Token coverage alone did not explain its remaining performance difference, and a Reading token-boundary repair did not eliminate the public-versus-local discrepancy.

Parameter count, compression and actual exposure define the conditions for comparing vocabulary size, depth, width and optimization. A tokenizer fitted on the wrong source collection cannot establish an admissible final result; repaired tokenizers and retraining define new comparison conditions. The frequency-gated compositional representation also has a useful implementation result: its zero-gate limit was checked, while the trained nonzero-gate variant did not surpass the reference. Its residual formula retains the ordinary embedding term at the other limit. GEGLU, attention-gating and additional layer-weighting proposals remain unrun candidates, not empirical contributions.

## Constructing Experience

[R10-R14](catalog.md) study what paired text contributes beyond simply adding more text. Aligned reformulations were compared with wrong correspondence, matched sources, exact repetition and separated presentation. The comparisons show task-dependent effects, and a damaging wrong-pair control must not be mistaken for evidence that the positive treatment beats ordinary training.

Compact-view reinvestment combines two operations: shortening a second expression of the source, and using the released word budget for additional pairs. Within the 423,520-word replacement block, the pair count rises from 10,094 to 12,155. That is about 20.4% more passage pairs in that block, not 20.4% more documents in the entire corpus. Compression and increased coverage are distinct possible contributors. The [construction ledger](../results/compact_budget.csv) records counts, not verified semantic equivalence or information gain.

Adjacency-breaking, repeated content, block layouts and interleaved layouts reveal different task trade-offs. Extractive alternatives test whether source coverage is sufficient; fluent bridges add lexical and semantic checks. Lexical closure does not guarantee faithful meaning, and the bridge prototypes did not yield a validated large-scale replacement stream. A four-condition deterministic rule-rewrite screen did not show joint gains in the intended abilities. Equal rows and absence of truncation did not make every arm's visible-token exposure equal.

## Objectives and Supervision

[R15-R23](catalog.md) ask which relationships receive an effective learning signal. Entity-mention consistency, counterfactual propagation and cross-span losses separate existing source sensitivity from newly acquired dependencies. True correspondence can outperform a destructive mismatch without outperforming ordinary whole-word masking. Local familiar cues can also solve an auxiliary task without the intended cross-sentence computation.

State-coherent text and relation-directed masking sometimes improved a target behavior while harming broader language performance. Shifted-token causal auxiliaries, including batch replacement and gradient-calibrated auxiliary learning, did not establish a complete-model improvement. Self-sampled replaced-token detection with gradient decoupling completed a short training screen, with both benefits and costs. Its earlier majority-class shortcut diagnosis and the later detector measurements are separate pieces of evidence; a proposed mature-model tail is not a completed result.

Word-mean losses, support floors, source-absent target weighting and directed cross-view masking show why supervision coverage is not a sufficient objective. Improving the chosen target class may worsen the wider task profile. An exactly budget-matched target-swap construction remains a tested implementation candidate without a demonstrated training gain. Pivot and spatial-witness studies further distinguish relation selection from direct retrieval of visible answers.

Centered exchange gradients and projected alignment are useful measurement methods, but a favorable direction or small cosine does not establish long-term benefit. The coherent-context margin pilot did not produce the intended coherent-versus-broken likelihood separation. Its missing zero-coefficient continuation control prevents attributing the full endpoint difference to the margin term alone.

## Learning Dynamics

[R24-R30](catalog.md) make training progress part of the scientific comparison. Adaptive masking can lead at an intermediate checkpoint and lose that advantage later. Switching masking granularity can change the direction of a task effect when the surrounding sequence and batch settings change. A short schedule that has already annealed its learning rate is not equivalent to the same exposure point inside a longer schedule.

Same-content order controls, developmental curricula and cadence variants did not establish a universal ordering advantage. Prefix-truncation repairs also changed what the model actually saw, so they cannot be interpreted as pure order interventions. Context-cap results require particular care: a favorable historical Overall contrast was largely driven by the reference's AoA component rather than simultaneous improvement across the other tasks.

Optimizer effects depended on learning stage, switching conditions and configuration: early gains could reverse in mature models. A contaminated LAMB comparison changed positional settings and other implementation details as well as the optimizer. Local trust-ratio reconstructions do not recover an entire optimizer history. Checkpoint averaging and restart studies likewise distinguish a selected peak from a reliable stabilization method.

Concentrated binding experience could teach a controlled task to a mature base, while weaker or sparse natural-stream conditions did not automatically acquire the same behavior. Extra diagnostic exposure is counted separately. Continuing relation losses, local/full-context objectives and remention curricula remain proposals where training was not completed. A later acquisition-timing study has independent value: occurrence scheduling altered the age-of-acquisition readout in a bounded shorter-budget experiment, but this is not the AoA score of the released continuation models, and reduced word coverage and broader loss costs limit the interpretation.

## Residual Learning

[R31-R34](catalog.md) distinguish three residual-learning constructions with different parameter-update rules. A zero-output primary bottleneck preserves the initial function and permits a disabled-path control, but joint training still updates the base. Dual-view and main/auxiliary separation experiments subsequently tested which parameters receive each gradient. Early recovery of difficult items did not guarantee a broad or mature-model benefit.

Frozen-parent continuation instead trains a dedicated increment while keeping the parent weights fixed. Ordinary prediction and parent-distribution matching are combined with coherent or disrupted text. The coherent/span-broken pair is the closest matched comparison; the ordinary and shuffled alternatives differ in additional aspects of training. Disabling the increment restores the parent function by construction, whereas retaining old abilities with the increment enabled is an empirical question.

The model lineage contains branches, not an additive ablation ladder. Continuous packing is not automatically an ancestor of every later model. A chosen inference scale is not an independent training replicate. Historical and repaired full-model scores represent different evaluation conditions. The [tail controls](../results/late_consolidation_controls.csv) and [scale sweep](../results/private_scale_sweep.csv) distinguish training interventions from changes to the dedicated branch's inference scale.

## Memory and Addressing

[R35-R38](catalog.md) separate storing a state from finding its address in language. Supplied-address experiments showed functional effects of writes, overwrites, slot exchanges and address corruption. A consistent renaming of addresses preserves the addressing relation and is therefore a symmetry control, not an adversarial corruption. Gold-address success does not establish that a text-driven router has learned the query address. In one router, high overall token accuracy concealed zero recall on the critical query tokens.

**The early tiny entity-memory binding interpretation is withdrawn.** Its answers followed the action outcome regardless of which entity was queried. High accuracy and sensitivity to write permutations therefore did not demonstrate query-conditioned binding or compositional recombination. A corrected paired-query corpus and a revised training script were constructed, but those artifacts do not establish a successful corrected-model experiment. This withdrawal does not invalidate the separate supplied-address experiments or the final BabyLM model scores.

Raw, lexical, recurrent and discrete address-learning interfaces retained their own failures against supplied-coordinate references. Pointer and copy initialization supplied another instructive boundary: after correcting candidate-token readout, the copy pathway was active but did not choose a property according to the queried entity. Copying, addressing and state selection require different tests.

## Selection and Reading

[R39-R40](catalog.md) investigate whether the difficulty lies in obtaining a usable reading format or choosing the correct record. A model could tolerate some format changes after learning even when training directly in that format was difficult. However, aligned, permuted and disjoint-label preparation all rescued a later task, weakening the claim that a specific learned address correspondence was essential. That comparison does not establish a universal head-only explanation.

A frozen reader combined with a four-candidate selector approached its supplied-address reference on exact names and role swaps. Prior-versus-revised state selection remained substantially weaker, and stable examples inflated pooled accuracy. Explicit candidate sets, tags and direct supervision remain part of the successful construction, not capabilities that were discovered from unrestricted text.

## Relations and Identity

[R41-R44](catalog.md) provide independent controlled results about rule identification and representation sharing. Initial-owner counterfactuals exposed an anti-copy shortcut compatible with high training accuracy. Alias balance, independent event/ranking reversals and counterbalanced examples make the intended rule more identifiable.

Binary relation graphs with sparse orientation anchors then showed that an absolute direction can propagate through a shared representation while relative relations remain intact. Shared and untied models, reversed anchors, relation closure and multiple seeds delimit this result. A readout that recovers a direction establishes representational sufficiency, not necessarily the model's ordinary use of that direction. Some edge-removal controls also change the number of training examples.

Finite character-equality supervision could repair identity matching and reconnect new name combinations to the relation graph. The construction still supplies alphabet coverage, candidate structure, a frozen matcher and hard assignment. It does not solve aliases, repeated mentions, pronouns or unrestricted semantic role induction. Neutral, reversed and program-structured task generators remain useful prototypes, with candidate-yield, balance and leakage checks, rather than evidence of general reasoning ability.

## Conditional Data Value

[R45-R49](catalog.md) replace a single notion of "good data" with a comparison that specifies the added material, removed material, model, budget and learning stage. Structure-density scores were strongly entangled with source and surface properties; residualized selections did not turn the original proxy into a reliable benefit predictor. Shared-anchor clusters outperformed some mismatches but not every repetition or untouched-tail reference.

Fixed-budget view, breadth and repeat substitutions retained different late effects, and those effects varied across architectures. Expanding sentence coverage does not necessarily increase independent document coverage. Register replacement showed that the opportunity cost of removing child-directed or adult-prose material matters even when the admitted block is fixed.

Prediction records require equal care. Word-distribution and profile-distribution predictors are different models, and their reported windows and task sets were not always the same as the eventual comparison. The result is not a universal scalar law of data value. Explicit relation filters and natural state-text selectors also had limited usable yield; a short screening result or a handful of accepted examples is not a mature relation-training corpus.

## Targets and Sources

[R50-R52](catalog.md) hold input fixed while changing which targets are learned, or hold target tokens fixed while changing source context. Source-absent targets are defined by their relationship to the source, not by a guarantee of novel meaning.

Deleting source-absent rather than copied targets increased a local target-loss measure under source-pair separation. The reported contrast is 0.0796 nats, with an interval of [0.031, 0.125]; stricter document-disjoint intervals cross zero. These are reported pair-bootstrap intervals, not variation across independently trained models. In the 100M target-deletion comparison, full training has a seven-family mean of 43.896, versus 43.437 and 42.792 for the two deletion conditions. That ranking does not identify a causal fraction of the complete-model gain. The [disjoint-target results](../results/source_disjoint_target_loss.csv) and [deletion results](../results/target_deletion_100m.csv) use different populations and endpoints.

Large source-conditioned reformulation effects can also remain when a target channel's direct supervision is absent. This does not prove input-only learning or general transfer. Source existence, source order, gaps, prefixes and relative window position are separate interventions. Text-only coverage measurements survive some loader-identity uncertainties, but unverified model readouts cannot establish a mechanism merely because the corresponding geometry is known.

## Functional Mechanisms

[R53-R59](catalog.md) distinguish information in a representation from computation that the model actually performs. Ordered dynamic tasks could expose local intermediate states while final composition failed. Same-entity history could perturb hidden states without improving the relevant content prediction. Micro-worlds separated explicit state statements, action inference, contradictory evidence and overwriting, but did not become reliable selectors of natural-benchmark performance.

Gradient-conflict and learning-signal estimators were tested with cross-pair, same-bag, blocked-permutation and held-object controls. Local association did not establish prospective learning value. Contrastive prediction and auxiliary readouts could fit or saturate without producing the intended context-specific behavior. Layer mixing, bilinear/ridge readouts, rotated nulls and limited position edits similarly preserve a distinction between fit, held transfer and intervention.

State-update and literal-binding investigations identify further alternative explanations: recency, position and format can solve a supposed update task; mention-conditioned summary reading is not literal state assignment; and learning assignment in a controlled construction need not transfer the intended variable to natural evaluation. Antisymmetric paired scores and target swaps provide diagnostic checks, but an algebraic cancellation in the score is not evidence that the model used the correspondence.

## Measurement and Computation

[R60-R66](catalog.md) are substantive methodological results. The same random seed did not give shared parameters the same initialization after a module was removed. Explicitly copying common tensors markedly weakened the previously inferred architecture-by-data interaction. Absolute model ability and the relative effect of a data intervention remain different questions.

Macro averages can improve while more discrete items are lost than gained, because the metrics weight tasks and examples differently. Oracle complementarity is not a working selector. Percentage points, fractions, nats, task means and Overall have different units or aggregation rules. Cross-architecture correlations and principal-component radius changes did not establish a capability-conservation law, and two observed seed differences are not a population noise estimate.

Evaluation depends on the complete loaded model. An encoder-loading path that omits residual parameters changes the comparison, even if a separate masked-language-model path loads correctly. Missing measurements, measured zero, rounded public displays and repaired local values have distinct meanings. AoA needs an identified checkpoint trajectory; Reading needs the correct token boundary and scoring interface. Target-count-weighted microbatch reduction and activation-checkpoint guards are reusable engineering contributions, with deterministic or short-horizon validation rather than a claim of identical stochastic training throughout.

Reconstruction records bind weights, configuration and custom model code. Recreating tensors with one fixed seed is different from scientific replication with another seed, and neither implies arbitrary exact resume without optimizer, random-state and data-position records. Parameter fraction, words, tokens, updates and trainer time are separate costs. Frozen parameters still participate in computation; unknown total search or evaluation cost is not zero.

## Selective Source Use and Functional Reach

[R67-R71](catalog.md) sharpen the link between the earlier constructions and later training design. On compact reformulation targets without token overlap with the source, exact recurrence and aligned reformulation produced opposite changes in true-source advantage relative to a neutral context. Across three training seeds, the changes were -0.8138 and +0.7970 nats, with seed standard deviations 0.1729 and 0.0758. Unrelated-source changes were much smaller. The sign convention matters: greater source advantage means lower target loss with the true source. See the [source-use data](../results/relation_context_use.csv).

Moving each pair's two parts into different training windows strongly attenuated these source effects while retaining that arm's material. The matched comparison is local versus split within each pairing type; the repeat and reformulation corpora are not themselves identical. Natural reformulation tests further separated reuse of source-recurring content from source-absent substitutions. Exact recurrence can help copying while imposing costs on changed-form targets; wrong correspondence can also teach a nonneutral response. Related autoregressive and alternative-architecture tests establish bounded transfer conditions, not objective-independent universality.

Controlled binding studies show a complementary distinction: familiar performance can recover without recovering use by unseen query symbols. Ordinary full-objective continuation, static supervision weights and interleaving left different functional reach. Donor activation replacement, clean-run centering/zeroing and rotation made a particular attribute-position signal causally consequential. Refitting linear directions on held symbols did not restore the lost behavior, but does not exclude every possible distributed recoding. The [functional-reach table](../results/interface_reach.csv) is a controlled-task result, not a general natural-language comprehension score.

## Principle-Guided Continuation

[R72-R74](catalog.md) translate the findings into a concrete training policy on the released parent. Existing source/reformulation text is retained. Input corruption and focused supervision are separated: dense masking removes many local hints, while only a sparse subset of those positions receives focused prediction loss. Ordinary prediction remains part of the acquisition objective. Here `(M,S)` means dense input masking with sparse focused targets; `(M,M)` means dense masking and dense focused targets.

The complete policy adds deterministic KL from the frozen parent's output distribution to the updated model on ordinary whole-word-masked input. This input still contains the source; it is not an evidence-absent condition. The KL direction is teacher-to-student. Correcting dropout and random-state interference was necessary to distinguish learned functional drift from stochastic disagreement.

The [complete evaluation](../results/training_strategy_comparison.csv) contains one parent and four policies with two continuation seeds:

| Policy | Seed 62064 Overall | Seed 62065 Overall | Cumulative word exposure |
| --- | ---: | ---: | ---: |
| Frozen parent | 42.02397 | Same parent | 86,005,295 |
| Ordinary continuation | 42.09261 | 42.11592 | 89,168,037 |
| Dense-input, sparse-target acquisition | 42.20254 | 42.17887 | 89,168,037 |
| Dense-input, dense-target acquisition | 42.14909 | 42.16842 | 89,168,037 |
| Acquisition plus ordinary-state preservation | 42.24641 | 42.23173 | 89,685,369 |

Both sparse-target continuations exceed their matched ordinary continuations at equal exposure. The full policy improves further, with 517,332 additional preservation words and extra computation. These are two continuations from the same pretrained parent, not two independently pretrained bases. The sparse-input/sparse-target control has mechanism and fast-evaluation evidence but is not in this complete nine-task table. Consequently, the full difference from ordinary continuation is not a single-factor masking effect.

Preservation changes the learning trade-off, not just the amount of parameter movement. On 1,003 common target positions from 169 familiar examples, ordinary-state preservation retains much of the new predictive improvement; dense-state preservation suppresses much of it. At the same positions, the dense-state preservation gradient is 11.835 times larger despite an equal coefficient. Full preservation comparisons additionally change target support, so input choice, support and effective strength are not independently isolated. Rollback and scale-reduction controls show that attenuation accounts for part of the recovery; the evidence does not assign every benefit uniquely to teacher-specific preservation. See the [state readouts](../results/state_preservation_readouts.csv) and [gradient diagnostic](../results/preservation_gradient_diagnostic.csv).

The Overall gains are not uniform: GlobalPIQA supplies most of the increase, from a small fixed evaluation set, while other components include both gains and losses. Increased source sensitivity can also carry costs when sources are wrong or missing. The practical result is a measured improvement by a specified combined policy, accompanied by a more precise account of what it learns and what it fails to preserve. It does not demonstrate that general context learning or state tracking has been solved.

## Report Data Map

The report's numerical figures and tables use the following data. Each table identifies a particular comparison rather than an additional independent experiment.

| Report material | Public data and scientific role |
| --- | --- |
| Compact-view construction | [compact_budget.csv](../results/compact_budget.csv): pairs and word-budget accounting |
| Late continuation and scale selection | [late_consolidation_controls.csv](../results/late_consolidation_controls.csv), [private_scale_sweep.csv](../results/private_scale_sweep.csv): matched versus near-matched controls and post-hoc scale choice |
| Training relation and source use | [relation_context_use.csv](../results/relation_context_use.csv): three-seed source-advantage changes and seed dispersion |
| Familiar/held reach and signal interventions | [interface_reach.csv](../results/interface_reach.csv): behavior and functional intervention, not a training chronology |
| Source-disjoint and target-deletion tables | [source_disjoint_target_loss.csv](../results/source_disjoint_target_loss.csv), [target_deletion_100m.csv](../results/target_deletion_100m.csv): different populations and separate target-loss/task-score comparisons |
| Strategy comparison and task contributions | [training_strategy_comparison.csv](../results/training_strategy_comparison.csv): complete nine-task endpoints; task contributions use component changes divided by nine |
| Acquisition and preservation diagnostics | [state_preservation_readouts.csv](../results/state_preservation_readouts.csv), [preservation_gradient_diagnostic.csv](../results/preservation_gradient_diagnostic.csv): distinct probe populations and a matched-position local gradient comparison |
| Historical public comparison and fixed model versions | [leaderboard_comparison.csv](../results/leaderboard_comparison.csv), [public_model_revisions.csv](../results/public_model_revisions.csv): the report's dated display snapshot, not a live ranking or a replacement for local full evaluation |

The input/supervision diagram defines the manipulated factors. The research-lineage diagram distinguishes model branches and controlled comparisons; neither diagram adds a new measurement.
