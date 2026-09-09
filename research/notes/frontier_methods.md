# Representation, Objectives, and Frontier Consolidation

## Research Questions

These studies ask how a fixed language-learning budget should be allocated among
source diversity, alternative expressions, prediction targets, representational
capacity, and later consolidation. Two questions must remain separate: which
configuration gives a stronger measured endpoint, and which controlled contrast
identifies a reusable learning mechanism. Stage I develops and verifies the
first-generation endpoint. Stage II tests explanations, including unsuccessful
controls and later corrections; its probes are not additional released models.

## Compact Views and Budget Reinvestment

Compact rewriting was studied through matched source/repeat/view contrasts, not
as an isolated claim that shorter text is better. A common core contained 10,094
source passages. Compact views released 69,575 words relative to the near-length
core budget; reinvestment admitted another 2,061 passage pairs. The resulting
12,155 pairs covered 4,529 documents and contained 261,803 source words plus
161,708 rewrite words. Nine neutral words completed the 423,520-word changed
block inside a 10,000,000-word pool. Ten ordered passes supplied the nominal
100,000,000-word exposure stream.

The final builder removes intact non-generated base rows, preserves the inherited
generated-pair block and all remaining coherent base rows, and packs each selected
source immediately before its view. An earlier overlay that shuffled the base
into word streams was superseded. The realized source-key order, packet packing,
common filler, and pass permutations are part of the experiment. Charged words,
visible BPE tokens, masked targets, and sequence positions are different budgets.

Compact views were compared with extra independent sources, source repetition,
broken adjacency, repacked recurrence and source-attested extraction. These comparisons do not
identify a universal compression effect. They change different combinations of
information, supervision, local context, and displaced base material.

## Representation and Prediction Targets

Tokenizer provenance changed the admissible comparison coordinate. Early results
using the inherited larger-corpus tokenizer cannot serve as compliant small-data
endpoints. The final parent uses a 16,384-token byte-level BPE tokenizer fitted
on the selected 10M-word pool. Historical tokenizer results remain informative
only within their stated coordinate.

Additional experiments tested larger vocabularies and depth, support-gated
compositional residuals, exposure-faithful chunking, whole-word versus token-level
objectives, word-mean loss, innovation-biased masking, relational pivots, auxiliary
source/view objectives, and optimizer or schedule changes. Their importance is
not limited to successful scores. Lower training loss, more visible targets, a
nonzero auxiliary gradient, or successful local retrieval did not reliably imply
broader downstream improvement.

Target-count-weighted microbatch reduction is an implementation result: averaging
microbatch means is generally not the same objective when their numbers of masked
targets differ. Deterministic reduction checks and short trajectory checks do not
establish full-run equality under changed dropout, numerical order, or batching.

## The Parent and Frozen-Anchor Continuation

The primary model is an eight-layer, width-480 DeBERTa masked language model with
128-dimensional post-layer bottleneck residuals, trained at residual scale 1.75.
Its 35,463,008 parameters are jointly trained; the original backbone is not an
independently preserved function during this phase. The selected checkpoint is
at 82,012,495 charged words, not exactly 82 million. Its complete nine-component
Overall is 41.942481167385985. The checkpoint was saved after update 2,074;
continuation begins after 530,944 stream rows.

The continuation freezes this complete parent and attaches another zero-output
bottleneck after each primary residual. The effective layer order is base layer,
primary residual, then dedicated continuation residual. Only the new 995,584
parameters are trained, giving 36,458,592 total parameters. Coherent replay uses
masked-language-model cross-entropy and deterministic enabled-versus-disabled
branch KL on the same legal suffix. It consumes 3,992,800 additional words in
101 updates, reaching 86,005,295 words.

A crucial correction separates this trainer from its predecessor: the earlier
neutrality-only mode did not perform main-stream MLM acquisition. Its auxiliary
aligned/shuffled conditions therefore differ from coherent replay in training
objective.

The matched coherent and span-broken replay arms score 44.10643 and 43.12143 on
the seven-component aggregate. This is a bounded structural contrast, not a
four-arm causal comparison with ordinary continuation and auxiliary shuffled
training, whose objectives or optimizer states differ. Disabling the added
branch recovers the frozen parent, but this construction does not guarantee
preservation when the branch is enabled.

The released first-generation frontier applies residual scale 0.75 to the same
learned continuation weights. The historical evaluation recorded 42.1210247099666; the released, adapter-aware common evaluation is 42.023967991315104. The latter is the reference in the final comparison table. Changing
the scale changes the function, not the training seed. The selected checkpoint
and scale are posterior choices, not benchmark-independent selection rules.
The aggregate gain coexists with a negative net balance of discrete correct
items. A separate 80M-to-84M continuation did not reproduce the proposed broad
preservation-plus-acquisition pattern: coherent replay scored 43.67000 versus
44.12214 for ordinary continuation. Endpoint utility therefore remains distinct
from a general consolidation principle.

## Mechanism Tests and Measurement

Natural-data mechanism studies separate source-absent target supervision from
copied-target supervision, and measure source-pair or document-disjoint target
loss. Improvements on these local readouts do not by themselves establish causal
mediation of downstream gains. Source absence is a token-level operational
definition, not proof of semantic novelty. Extractive coverage and fluent-bridge
construction also require separate fidelity, packing, and supervision controls.

Architecture comparisons require more than the same random seed. The
disentangled-position ablation was revised to copy all common initial tensors;
the strong negative interaction from the unmatched initialization did not survive
that repair. A correction from a fraction to percentage-point units also changes the
interpreted magnitude of an effect. Task-weighted aggregates,
item-weighted changes, and per-family effects answer different questions.

Dose and register-replacement studies further make the displaced material part
of the treatment. Admitting the same new block while removing different base
registers can reverse simple quality or residual-loss predictions. The proposed
combination of distinct support, remaining learnable structure, displaced value,
and architecture-dependent transfer is a working explanation, not a fitted or
independently identified predictive law. Some replicate checkpoints lack
complete benchmark measurements.

## Graphs, Memory, and Interfaces

Graph-conditioned generation exposed weak edge-changing transformations and
strong query priors. Candidate quality and negative controls did not establish
successful large-scale graph learning.

The early tiny entity-memory recombination interpretation is withdrawn: its
answers followed action outcomes regardless of the queried entity. High scores
for shared and independent memory therefore do not establish compositional
binding. A corrected corpus and revised trainer do not establish a successful
corrected-model experiment. Separate supplied-address studies tested functional
writing and reading, while raw-role and raw-name failures limited access to those
operations from text. The [withdrawal and addressing distinction](../scientific_guide.md#memory-and-addressing) concerns different experimental tasks.

Controlled relation-graph experiments provide a narrower positive result:
sparse absolute anchors can orient held relations through shared computation and
comparison edges. Changing the anchor sign changes the direction of held-state
predictions. Learned finite-alphabet equality can supply part of the binding
interface. These experiments still supply candidates, equality supervision,
routing structure, and restricted relation families; they do not demonstrate
unrestricted language understanding. Incompletely fitted untied controls,
saturated-probability margin bookkeeping, and unequal row/update counts after
edge deletion limit stronger necessity claims.

## Methods and Evidence

| Scientific question | Method | Evidence and boundary |
| --- | --- | --- |
| What does compression release for other examples? | [Compact views and reinvestment](../../methods/compact_views.md) | [Word and pair budget](../../results/compact_budget.csv); counts alone do not establish semantic improvement |
| Can an increment preserve and extend a fixed model? | [Residual learning](../../methods/residual_learning.md) | [Late controls](../../results/late_consolidation_controls.csv), [scale sweep](../../results/private_scale_sweep.csv); enabled-function retention is empirical |
| Which target supervision changes prediction? | [Target-channel comparison](#mechanism-tests-and-measurement) | [Source-disjoint losses](../../results/source_disjoint_target_loss.csv), [100M target deletion](../../results/target_deletion_100m.csv); distinct populations and outcomes |
| Did architecture or initialization change the data effect? | [Common-tensor control](#mechanism-tests-and-measurement) | [Initialization identity](../../experiments/archive/frontier_consolidation/data/commoncopy_integrity_final/commoncopy_integrity_47bb9cf8.json), [four-condition comparison](../../experiments/archive/frontier_consolidation/data/commoncopy_architecture_interaction_four_cell_bootstrap/architecture_interaction_four_cell_bootstrap.json) |
| How does displaced register affect data value? | [Register and dose comparisons](#mechanism-tests-and-measurement) | [Second-seed register contrasts](../../experiments/archive/relation_learning/data/frontier_consolidation_readout/register_seed_contrasts.csv); no universal predictor is identified |
| When do sparse anchors orient unseen relations? | [Shared relation computation](#graphs-memory-and-interfaces) | [Multiseed graph and affine readouts](../../experiments/archive/representation_and_objectives/data/multiseed_gauge_affine_full/multiseed_gauge_and_affine_summary.json); supplied candidates and restricted relations bound transfer |

Complete Overall, seven-family aggregates, item changes and target losses answer
different questions. Their distinctions separate useful endpoints from causal
explanations of learning.
