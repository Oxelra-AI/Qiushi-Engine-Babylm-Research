# Relations, Functional Reach, and Selective Preservation

## Scientific Questions

The mechanism studies ask what finite language-model experience actually teaches.
Does repetition improve general prediction, or a particular source-to-target
routine? Does a representation remain usable for unpracticed symbols after a
broader objective is introduced? Can new source-responsive behavior be acquired
without unnecessarily changing the model's established predictions?

Two complementary experimental families address these questions. **Relation learning** varies
source correspondence, target form, and same-window availability in natural text.
**Functional learning** uses controlled binding tasks to intervene on credit,
internal signals, and transfer to held symbols. Stage III applies the resulting
design constraints to an existing small language model. The studies do not make
the synthetic task equivalent to natural language, or a mechanism probe equivalent
to a full benchmark evaluation.

## Source Relation, Target Form, and Window

Exact recurrence, corresponding restatement, shuffled correspondence, and separated
source/companion exposure are different learning experiences even when their word
budgets are similar. The split controls preserve source and companion text
multisets while removing their co-occurrence in the same training row. Their
attenuation of the recurrence and restatement effects supports a contribution
from practiced local relations, not exposure counts alone.

The readout separates true source (T), unrelated source (U), and neutral context
(N). Source benefit is measured against N, alongside absolute target loss and the
U-versus-N change. This matters because an apparent improvement in source benefit
can result from a worse comparison context rather than better prediction under T.
Target classes must also remain separate: a word recurring in the source and a
source-absent substitution are not interchangeable tests of restatement.

In the designed compact-text family, corresponding restatement improved
source-conditioned prediction for nonoverlapping targets, while exact recurrence
impaired it. The final report's neutral-anchor source-advantage contrasts are
+0.7970 and -0.8138 nats, respectively; other probe populations
use different target sets and do not estimate the same contrast. Exact-recurrence liability
extended more broadly across registers and, in one tested setting, to a causal
language-model objective. Positive restatement transfer was narrower and depended
on the practiced target relation. Neither observation warrants a universal claim
about all duplication or all paraphrase augmentation.

Two corrections are essential. First, the half-rewrite control also trained
compact targets after neutral filling. A hash-mixed versus half-rewrite contrast
therefore did not cleanly isolate cooperation between exact and rewritten examples:
its compact-target difference was driven by N/U movement, not improved T loss.
Second, unrelated-neighbor vulnerability separated correspondence from
noncorrespondence most clearly in the near-register probe. It was not a general
cross-register measure of trust.

Exposure audits further narrowed the interpretation of natural probes. A
nominal holdout was not sufficient: newly generated material and inherited streams
could contain the same reference passages. Overlap measurements led to corrected
complement analyses. Generalization claims based on overlapping passages are
withdrawn; those examples do not become independent tests through a different
label.

## Functional Reach and Activation Interventions

Early source-trigger tasks did not establish reusable binding. A supposedly
unpaired comparator enforced a different target and thus taught anti-identity,
while direct attention-edge blocking left indirect information paths available.
Same-token-bag rebinding probes showed that late source-conditioned interactions
could coexist with deteriorating held-out prediction. Those findings motivated a
query-first permutation-orbit task in which the answer must follow the queried
entity's current assignment.

In that controlled task, answer-focused preparation produced a transferable
selection computation. Broad continuation could retain or recover familiar-symbol
performance while losing transfer to held query symbols. Static, relation-aligned
answer credit with restrained non-answer pressure preserved much of the benefit
of interleaving. Misaligned answer targets did not. The coefficient 1/17 was an
algebraic matching coefficient for the tested layout, not a universal optimum or
a scaling law.

A query-match component at attribute positions supplied an intervention target.
Donor-query patching redirected answers, whereas an orthogonal-component control
did not produce the same redirection. This is sufficiency evidence. Subsequent
clean-run centering, zeroing, and rotation of the component showed that ordinary
answers depended on its slot-specific pattern. Cross-model transplantation and
trajectory measurements distinguished weakened signal formation from reduced
downstream sensitivity; they did not establish that all damage occurred at only
one location. Directions fitted on held-symbol states did not recover the lost
reach. These results concern a small, engineered binding task and two principal
transfer seeds, not a demonstrated universal natural-language circuit.

## Failed Bridges and Scientific Corrections

Natural-language state-update and binding constructions exposed several alternative
solutions: candidate presence, descriptor matching, position, recency, operation
count, and formatting. Balanced batches, mirrored assignments, frame variation,
entity-position checks, and counterfactual controls are scientifically important
even when their initial headline result was withdrawn. Learnable literal
assignments within a constructed family did not establish general state tracking
on unrelated benchmark tasks.

Compaction also required semantic checks. Shorter text could lose negation,
modality, causal qualifications, or entity/number relations. The construction
therefore combines semantic template constraints, admission rules, matched
recurrence and auxiliary-support allocations, and negative controls. Compression
ratio alone is not evidence of a better learning experience.

The corrected bridge trainer normalized losses over a complete word-paced update
and used row-keyed ordinary masking across arms. Earlier averaging of microbatch
means and differing corruption streams changed the experiment being compared.
Those older conditions use different loss weighting or masking and are not
equivalent implementations of the corrected comparison.

## Dense Masks, Sparse Targets, and Ordinary-State Preservation

The final acquisition policy keeps the first/source view visible in packed
restatement rows, densely masks detected second-view content groups, and supervises
a sparse deterministic subset. Sparse labels follow the original sampling policy;
dense masking is capped and always includes the labeled positions. Non-pair rows
retain ordinary whole-word masking. Comparing sparse-mask/sparse-label, dense-mask/
sparse-label, and dense-mask/dense-label policies separates changes to available
clues from expansion of supervised targets more carefully than a single dense
versus ordinary comparison.

Training updates the existing dedicated adapter rather than replacing the model:
995,584 trainable parameters within a 36,458,592-parameter masked language model.
The principal continuation uses 80 word-paced updates, 512-token sequences,
learning-rate schedule offset 101 of 455, peak learning rate 0.00005, and ordinary
mask probability 0.15. Focus probability is 0.35 with at most 16 sampled label
groups per row and focus-loss weight 0.15. Construction rules and random-number streams
determine which examples and target positions participate in each update.

The preservation policy adds KL(teacher || student) from the frozen starting model
on ordinary-mask versions of the same packed pair rows. Both preservation
forwards use evaluation mode, and RNG restoration prevents teacher loading or the
auxiliary branch from shifting later acquisition randomness. The source evidence
remains present: this is ordinary-state preservation, not evidence-absent
distillation. Earlier coefficient comparisons also changed RNG/dropout behavior
and are superseded. A zero-preservation replay checks equivalence with the
acquisition implementation.

Acquisition changed evidence dependence asymmetrically: correct-source losses
improved slightly, while wrong-source and absent-source conditions often worsened
more. Preservation reduced ordinary-state drift while retaining much of the
acquisition. On 1,003 fixed common-support tokens from familiar training rows,
dense-state CE gains were about 0.757 for acquisition and 0.682 for ordinary-state
preservation, versus 0.159 when preservation was applied under dense corruption.
These are state-specific fit measurements, not held-out transfer scores.

**Constraint strength remains a confound.** At the same coefficient, dense-state
preservation had about 11.835 times the adapter-gradient norm on common support;
the full branches also used different target supports. The dense-preservation
result therefore changes rendering, support, and effective constraint strength.
It cannot uniquely establish a geometry-only mechanism. Matched rollback explains
much of the repair, and no control isolated teacher information from the extra
ordinary-mask student presentations. Whole-adapter scaling and rollback of only
the new update are distinct controls. A lexical probe referenced to an older
checkpoint measures a different contrast from recovery to the immediate teacher.

## Full Evaluation and Boundaries

The adapter-aware evaluation repaired a loader path that otherwise used a stock
encoder without the intended adapters. Historical and repaired evaluations
therefore concern different loaded functions. Full results require all measured components;
partial screens, export compatibility tests, and missing AoA trajectories cannot
substitute for them. Measured AoA values of zero are distinguished from unavailable
measurements and accompanied by raw-fit diagnostics.

The completed fixed-coordinate aggregate ladder is:

| Policy | Training seed 62064 | Training seed 62065 |
| --- | ---: | ---: |
| Starting reference | 42.02397 | 42.02397 |
| Matched ordinary continuation | 42.09261 | 42.11592 |
| Dense-mask, sparse-label acquisition | 42.20254 | 42.17887 |
| Acquisition with ordinary-state preservation | 42.24641 | 42.23173 |

These are policy contrasts on one inherited model lineage. The preservation
increment over acquisition is small, about 0.044/0.053 aggregate points. The
complete policy exceeds ordinary continuation by about 0.154/0.116, with Entity
gains of 1.24/1.36 points but remaining BLiMP/Supplement costs. Downstream
fine-tuning variation is material; the ordinary second-seed model has the highest
SuperGLUE value in this ladder. A paired-item interval for the first-seed direct
preservation increment crosses zero. GlobalPIQA improvements are concentrated in
a handful of fixed items and do not establish broad commonsense advancement.

The supported conclusion is a bounded acquisition-retention improvement, informed
by relation structure, credit, and effective input. It is not universal state
tracking, uniformly improved context use, a uniquely identified teacher effect,
or replication across independently trained trunks.

## Methods and Evidence

| Scientific question | Method | Evidence |
| --- | --- | --- |
| Which source-to-target relation is practiced? | [Relation learning](../../methods/relation_learning.md) | [Neutral-anchor contrasts](../../results/relation_context_use.csv), [original three-seed aggregate](../../experiments/archive/relation_learning/data/original_threeseed_neutral_anchor/rewrite_TUN_across_seed_contrasts.csv) |
| Can familiar behavior coexist with lost held-symbol use? | [Functional access](../../methods/functional_access.md) | [Behavior and intervention table](../../results/interface_reach.csv) |
| Does dense input require dense supervision? | [Acquisition design](../../methods/principle_guided_training.md) | [Complete policy comparison](../../results/training_strategy_comparison.csv) |
| How strongly does preservation constrain new learning? | [Preservation analysis](#dense-masks-sparse-targets-and-ordinary-state-preservation) | [State readouts](../../results/state_preservation_readouts.csv), [matched-position gradients](../../results/preservation_gradient_diagnostic.csv) |

The [mechanism report](../../reports/zh/chapters/04_principles.tex) and [model-comparison report](../../reports/zh/chapters/05_guided_improvement.tex) distinguish the controlled-task evidence from the complete-model result.
