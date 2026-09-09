# Research catalogue

Scientific topics, interventions, findings and their current status. Identifiers are navigation aids, not a count of independent innovations.

[Three research stages](README.md) | [Chinese report](../reports/zh/qiushi-engine-babylm-report-zh.pdf)

## Models and representations

### R01 - Sparse routing and prefix memory

**Conditional result.** Compare sparse access, capacity, and shared initialization. Local changes did not yield a stable broad advantage; fewer active parameters are not a substitute for comparing capability.

### R02 - Character, subword, and morphological pathways

**Conditional result.** Distinguish auxiliary word-form structure from capacity effects and lookup controls. Surface-information gains in limited settings have not become a stable broad advantage across seeds.

### R03 - Whole-word and token masking

**Experimental result.** Compare masking granularity against a fixed training reference, examining paired seeds and individual task scores. Granularity changes task trade-offs, supporting specific training choices but not a uniform ranking across all capabilities.

### R04 - Backbone architecture and batch-size controls

**Experimental result.** Compare BERT- and DeBERTa-style configurations and check batch-size differences. The late-training advantage of the combined configuration cannot be explained by batch size alone; the contributions of position-related components have not been independently isolated.

### R05 - Recursive autoregressive training

**Conditional result.** Reproduce the recursive training scheme, complete the corresponding evaluation, and check word coverage. This setting did not match the leading configuration; word-coverage differences cannot explain the entire performance gap.

### R06 - Vocabulary, depth, width, and budget

**Experimental result.** Compare vocabulary sizes and depth-width configurations, distinguishing prototypes from budget-compliant retraining. Higher tokenization compression ratios did not automatically yield stable gains; the actual training budget must also be accounted for.

### R07 - Frequency gating and fine-grained representations

**Exploratory construction.** Let coarse-grained word representations draw on fine-grained units and check the zero-gate limit. The limiting implementation behaves as expected, but trained models were not stronger; other gates and seeds remain candidates.

### R08 - Components of a mature training recipe

**Experimental result.** Examine depth, width, vocabulary, LAMB, and sequence-length and masking curricula. No single component reproduced the full configuration's performance; interactions among components must also be studied.

### R09 - Gated feed-forward and layer-weighting candidates

**Not yet run.** Retain proposed designs for GEGLU, attention gating, and layer weighting. These are candidates, not validated methods, and do not count as empirical model contributions.

## Experience construction and compression

### R10 - Aligned reformulation and correspondence

**Experimental result.** Compare correct and incorrect correspondence, matched sources, repetition, and split windows. Relational organization changes learning, but benefits depend on the task; follow-up studies in the main text further distinguish target types.

### R11 - Compact views and budget reinvestment

**Modeling method.** Compress the second view and use the freed word budget for more pairs. Experience coverage increased, yielding a practical frontier-model method; the effects of compression and expanded coverage must be distinguished.

### R12 - Adjacency, repetition, and presentation layout

**Experimental result.** Compare content and adjacency effects, examining block and interleaved presentation. Effects differ across capabilities; tokenizers and window settings must be matched.

### R13 - Extractive compression and bridging text

**Exploratory construction.** Construct bridging texts with balanced coverage, broader coverage, and controlled lexical scope. Word coverage can be checked, but it is not a substitute for semantic relations or gains in the complete model.

### R14 - Rule-based rewriting

**Conditional result.** Compare true pairs, same-entity mismatches, source-only controls, and shuffled controls. Short-run results showed capability trade-offs, not joint gains; visible token counts still need to be matched.

## Learning objectives and supervision allocation

### R15 - Entity-mention consistency

**Conditional result.** Compare correct correspondence, incorrect correspondence, and ordinary whole-word masking. Local advantages from correct correspondence did not reliably translate into broad capability gains.

### R16 - Counterfactual and cross-span supervision

**Experimental result.** Compare true-relation, wrong-relation, and no-relation conditions, checking existing capabilities. An identifiable relation effect is a different conclusion from learning a relation from scratch.

### R17 - State text and directed masking

**Conditional result.** Construct training texts and target-position controls around state selection. Target behavior can improve while broad language performance degrades, motivating joint tests of new learning and retention.

### R18 - Auxiliary causal objectives

**Conditional result.** Match shifted targets and gradient norms, comparing batch replacement with auxiliary learning. Similarly directed gradients did not guarantee full-model improvement; local gradient relationships do not establish a general rule for training outcomes.

### R19 - Self-sampled replaced-token detection

**Conditional result.** Add a detection objective and decompose its gradient effects. Short-run aggregate scores showed both gains and costs; overall effectiveness in late training remains unconfirmed.

### R20 - Target weighting and supervision coverage

**Conditional result.** Compare per-word averaging, coverage floors, and source-absent content weighting. Local fit can improve at the cost of broader performance; replacement under an equal target budget still needs separate testing.

### R21 - Relational pivots and counterfactual cues

**Conditional result.** Match supervision quantity and distinguish relational selection from direct retrieval. If retrieval still solves the task, high scores cannot establish a new computation for state selection.

### R22 - Gradient exchange and projection

**Exploratory construction.** Use centered four-condition comparisons, directional projection, and negative examples. Local gradient differences have been observed, but long-term training gains have not been established.

### R23 - Coherent-context margin loss

**Conditional result.** Compare against negatives with disrupted context and examine late-stage continuation. The expected behavioral separation was not obtained; the zero-coefficient comparison was not completed, so the result cannot be fully attributed to a single factor.

## Curricula and learning dynamics

### R24 - Adaptive masking granularity

**Experimental result.** Compare data-objective combinations along training trajectories. Early rankings can reverse; the learning stage must be considered when judging a training scheme.

### R25 - Data order and developmental curricula

**Experimental result.** Use random and developmental orderings of the same content and correct truncation. An effective combined scheme does not establish a universal advantage for curriculum ordering.

### R26 - Context-length caps

**Conditional result.** Hold paired-example exposure fixed while comparing length caps and evaluation lengths. The observed Overall change is affected by the AoA reference; the seven task families did not improve simultaneously.

### R27 - Optimizers and mature model states

**Experimental result.** Compare AdamW, LAMB, optimizer switches at maturity, and trust ratios. Local numerical changes during optimization have not yet been shown to predict late-training model outcomes.

### R28 - Checkpoint averaging and restarts

**Experimental result.** Compare averaging adjacent checkpoints, restarts, and another seed. Averaging did not reliably improve on individual high-scoring models; model selection and methodological attribution must be separated.

### R29 - Relation dose and learning maturity

**Experimental result.** Compare concentrated, sparse, and role-changing binding experience. Constructed tasks can be learned, but natural-text training with a weak base and sparse exposure did not automatically reproduce the capability; additional budget is listed separately.

### R30 - Continued relation-training candidates

**Not yet run.** Retain proposals for continued relation losses, switching between local and full context, and remention masking. These remain routes to test, not achieved results.

## Residual structures and incremental learning

### R31 - Zero-initialized bottleneck branch

**Modeling method.** Check shared parameters, disabled-branch controls, and training trajectories. The branch preserves the base function at initialization and when disabled; joint training still updates the base.

### R32 - Main and auxiliary path separation

**Experimental result.** Move from broad to sparse auxiliary learning, including comparisons with the auxiliary branch disabled on the main path. Early gains did not carry over directly to mature models, motivating ordinary coherent replay.

### R33 - Dedicated increments on a frozen base

**Modeling method.** Train only newly added parameters, using ordinary prediction and KL; compare coherent and disrupted inputs. This produced the first-generation model; disabling the branch restores the base, but retention with it enabled still requires evaluation.

### R34 - Model lineage and scale selection

**Methods and diagnostics.** Distinguish continuous-packing branches, the main model, incremental branches, and the inference scale used for release. Actual weight inheritance does not mean that every research route entered the final model in sequence.

## Entity memory and addressing

### R35 - Writing and reading with supplied addresses

**Experimental result.** Compare consistent address permutations, wrong addresses, overwrites, and slot swaps. Stored signals have a functional role; results using gold-standard addresses do not establish natural-language addressing.

### R36 - Answer shortcuts in a tiny binding task

**Withdrawn.** Subsequent checks found that answers did not change with the queried entity. The original high scores no longer support compositional binding; correcting the corpus is not equivalent to verification by retraining.

### R37 - Routing from natural text to addresses

**Experimental result.** Compare raw-text, lexical, recurrent, and discrete routing with correct supplied addresses. High aggregate token accuracy concealed zero recall on critical queries, identifying an addressing bottleneck.

### R38 - Pointer and copy initialization

**Exploratory construction.** Test initialization for small pointer models, embeddings, and copying behavior. Being able to copy does not mean being able to select the correct entity state.

## Selection and readout

### R39 - Format preadaptation and label alignment

**Experimental result.** Compare preparatory training with aligned, shuffled, and unrelated labels. Several forms of preparation can help subsequent learning, so the benefit cannot be attributed solely to one particular correspondence signal.

### R40 - Frozen readers and within-group selection

**Experimental result.** Train selection using a four-candidate within-group softmax, measuring lexical, role, and temporal semantics separately. Lexical matching and role swaps were nearly fully successful, but selecting prior versus updated states remained weak.

## Relational structure and identity

### R41 - Identifiability of relational rules

**Experimental result.** Vary the initial owner and construct balanced counterexamples. High training scores may reflect an anti-copy shortcut; counterfactuals help identify the intended rule.

### R42 - Binary relation graphs and sparse anchors

**Experimental result.** Compare shared and separate representations, anchor reversal, relation closure, and multiple seeds. Absolute orientation can propagate along shared paths while relative relations are preserved; the conclusion is limited to the given hypothesis class.

### R43 - Character identity and access to relational structure

**Experimental result.** Learn matching from letter-equality supervision, then connect it to shared relational structure. Coverage of the base alphabet supports new combinations; candidates and hard assignments are supplied, and edge deletion also changes the training amount.

### R44 - Leakage-free graph and program construction

**Exploratory construction.** Generate neutral, reversed, and program-structured tasks, checking categories and priors. Candidate yield and type balance remain limited; prototypes do not establish general reasoning ability.

## Conditional value of experience

### R45 - Structural density and representation recovery

**Conditional result.** Compare structural densities, residualized conditions, and random controls. Recovery of local structural readouts did not guarantee broad gains and must not be conflated with capacity.

### R46 - Tail training on shared-source clusters

**Conditional result.** Compare true, shuffled, repeated, and ordinary experience. Outperforming mismatched controls does not mean outperforming all references; model measurements are incomplete.

### R47 - Views, breadth, and repetition under a fixed budget

**Experimental result.** Replace training experience under a common budget and retain multiple late-stage models. Late-stage gains depend on experience type and architecture; adding distinct sentences is not equivalent to adding documents.

### R48 - Register substitution and data-value prediction

**Experimental result.** Hold added content fixed, vary the displaced register, and compare word distributions and task scores. Opportunity costs differ with the material displaced; results from different measurement settings cannot directly validate the same prediction.

### R49 - Selecting explicit relational text

**Exploratory construction.** Compare explicit relational text with random quality controls and filter natural state text. Qualified candidates are limited; short-run results cannot be treated as a mature large-scale training stream.

## Supervision targets and sources

### R50 - Deletion of source-absent and copied targets

**Experimental result.** Keep inputs fixed and nearly match the number of deleted targets, using source-pair-disjoint and document-disjoint splits. Local source-pair results hold, but intervals under stricter document separation cross zero; rankings on the full task set differ.

### R51 - Correct, wrong, and missing sources

**Experimental result.** Hold target tokens fixed and vary source conditions. Source effects can exist without added supervision and cannot directly be treated as the cause of final-model improvement.

### R52 - Coverage, order, and local context

**Methods and diagnostics.** Distinguish source presence, prefix placement, gaps, and relative text positions within windows. Comparisons with incompletely verified model-loading identity illustrate context geometry only; they do not confirm specific mechanism effects.

## Functional mechanisms and learning signals

### R53 - Ordered composition and dynamic binding

**Experimental result.** Compare dynamic and static relations, local and final supervision, and examine readouts. Information decodable from representations is not necessarily a computation the model actually performs.

### R54 - Contextual hidden states and interventions

**Experimental result.** Compare consistent prefixes, mismatches, entity substitutions, and activation replacements. Topic information can be present even when state-dependent execution fails; behavioral and functional tests are both needed.

### R55 - State updates in microworlds

**Methods and diagnostics.** Control states, actions, contradictions, answer cues, and overwrites. These provide diagnostic tools, not automatic evidence that the same selector has formed in natural text.

### R56 - Gradient conflict and learning-signal estimation

**Conditional result.** Use comparisons with identical materials, across pairs, and with held-out objects. Local correlations have not become stable prospective predictors of learning gains.

### R57 - Contrastive prediction and auxiliary readouts

**Conditional result.** Compare same-source, cross-source, and hard-negative conditions. Fitting or saturating an auxiliary objective is insufficient evidence that new capability has formed.

### R58 - Layer-wise readouts and position editing

**Conditional result.** Use bilinear and ridge-regression readouts, rotated-null controls, and local edits. Local fit does not guarantee held-out transfer; limited editing examples do not support broad mechanistic conclusions.

### R59 - Score changes under pair swaps

**Methods and diagnostics.** Check matching conditions including categories, shuffling, and length swaps. Algebraic cancellation in scores does not mean that the model actually used the correspondence.

## Measurement, transfer, and attribution

### R60 - Shared-parameter initialization

**Methods and diagnostics.** Explicitly copy shared tensors and separately measure architectures' absolute capability and the relative effects of data. The same random seed does not guarantee identical initial values; after correction, evidence for a large interaction was substantially weakened.

### R61 - Macro averages and item-level changes

**Methods and diagnostics.** Compute gains, losses, net changes, and complementary items separately. Macro averages and net changes in the number of correct answers can have opposite signs; one aggregate score cannot fully describe capability changes.

### R62 - Transfer conditions and representation geometry

**Conditional result.** Compare seed conditions, principal components, and random controls. Correlations and radial growth have not established capability conservation or a universal law of transfer.

### R63 - Complete evaluation and model interfaces

**Methods and diagnostics.** Verify the loading implementation, AoA trajectories, and commonsense-task and Reading scoring. Missing measurements, zero scores, and publicly displayed values are treated separately; complete model identity determines comparability.

## Systems and computation

### R64 - Word boundaries, packing, and visible budgets

**Methods and diagnostics.** Check pair integrity, truncation, and actual presentation in continuous windows. Accounted word counts must correspond to visible content; prototype and final tokenizer implementations must not be mixed.

### R65 - Microbatches and gradient equivalence

**Methods and diagnostics.** Weight by target count and test gradient checkpointing and deterministic gradients. These checks validate limited implementation conditions, not exact identity of complete stochastic training trajectories.

### R66 - Model reconstruction and cost breakdown

**Methods and diagnostics.** Fix the identities of weights, configuration, and code, separating training from search costs. Same-seed reconstruction, replication across seeds, and compute savings are distinct conclusions.

## Core research: extensions and empirical tests

### R67 - Selectivity in relation learning

**Experimental result.** Compare source advantages from verbatim repetition and aligned reformulation across three seeds. The effects on compact reformulation targets differ in direction and do not imply that repetition is universally harmful.

### R68 - Prediction windows with matched materials

**Experimental result.** Split repetition pairs and reformulation pairs into separate windows in their respective conditions. The large source effects were substantially attenuated; content is matched between the same-window and split-window versions of each pairing condition.

### R69 - Transfer conditions for natural reformulations

**Experimental result.** Group natural-restatement targets by whether their exact tokenizer ID appears in the source. Aligned restatement improves source use for recurring target tokens; nonrecurring targets show no equally stable improvement over the reference. This distinction is lexical, not a definition of semantic novelty.

### R70 - Familiar performance and functional access with unseen symbols

**Experimental result.** Compare ordinary continuation, statically weighted supervision, and interleaved supervision. Recovery of familiar behavior does not guarantee restored functional reach for unseen symbols.

### R71 - Erasing and relocating functional signals

**Experimental result.** Center, zero, and rotate attribute-position signals, and test refitted directions. The results support the functional role of a specific signal; unsuccessful recovery using linear directions does not rule out all distributed recoding.

### R72 - Dense inputs and sparse supervision

**Modeling method.** Reuse existing text while controlling masking candidates and focused targets separately. At equal exposure, complete evaluations for both seeds outperformed ordinary continuation; the comparison does not isolate masking as a single factor.

### R73 - Preservation inputs and effective strength

**Experimental result.** Measure preservation gradients at fixed shared target positions and compare ordinary and dense inputs. With the same coefficient, gradient magnitudes still differ by a factor of approximately 12; the full comparison also changes the set of target positions.

### R74 - Principle-guided complete model

**Modeling method.** Recheck ordinary continuation, relation learning, and the complete preservation scheme from the same parent model. The complete scheme improved the nine-task Overall score for both continuation seeds; additional exposure and compute are reported separately.
