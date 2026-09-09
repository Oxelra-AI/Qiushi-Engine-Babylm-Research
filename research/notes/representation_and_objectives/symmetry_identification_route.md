# symmetry identification route — Symmetry-identification route after compact-branch closure

## Scientific Motivation exists

The post-SOTA goal is not another BabyLM score: it is a general data-efficient learning principle for how limited data and limited compute create reusable, compositional, generalizable, and transferable knowledge.

The compact-view branch now has a strong negative constraint. earlier analysis showed that official Entity view-minus-breadth (V-B) is a real score surface, but the paired affected/unaffected binding table does not show conserved state-record formation:

- official Entity V-B late macro mean: +2.994 pp overall, +3.376 pp zero-operation, +2.918 pp nonzero-operation;
- paired binding V-B late mean both-correct: -1.389 pp;
- not-both→both: +2.257 pp, but both→not-both: +3.646 pp;
- dominant flow: unaffected-only→affected-only +20.312 pp, plus both→affected-only +2.257 pp.

Thus the Entity aggregate advantage should not be read as source-conditioned state binding. It is churn around a decision surface, with changed-state gains and stable-state erosion. Unless already-running frontier_consolidation scorer files materially change broad ex-Entity V-B or paired conservation, compact-view mechanism work should remain closed.

## Surviving scientific object

The strongest surviving controlled result is not compact rewriting. It is a symmetry/identifiability law:

> When a new relation component is learned only through internally consistent relations among its own members, its absolute role orientation remains ambiguous up to a permutation or sign. Sparse reliable evidence that links the new component to already grounded role coordinates identifies that orientation. Data-efficient transfer occurs when this orientation propagates through reusable argument-slot interfaces, not when the learner merely fits held-held consistency or aggregate accuracy.

Prior evidence:

- role coordinate anchor and state probe factorized signed-role model: true anchors make held-seen and held-held transfer perfect; shuffled anchors preserve held-held consistency but fail mixed held-seen orientation.
- Steps257–260 slot reuse controls: reusable fillers and reliable aligned sparse evidence improve held transfer; anti-aligned evidence moves behavior in the opposite direction; raw familiarity/random cooccurrence is insufficient.
- earlier analysis pretrained bridge: aligned evidence orients familiar event-role and some familiar ranking interfaces; state transfer to held contexts remains weak.
- Steps265–273 temporal bridge: supplied addresses and exact role strings can compose, but prior/revised paraphrases fail; lexical matching explains exact role success. This blocks supplied tags, exact role-string mapping, and directly addressed records as evidence for natural semantic role induction.

## Literature grounding used in symmetry identification route

The natural test should use established transfer-of-possession role structure rather than arbitrary labels.

- VerbNet transfer semantics defines possession transfer with pre-state Source has Theme, Recipient lacks Theme, transfer event, post-state Recipient has Theme, Source lacks Theme, and causal linkage. It distinguishes source-initiated give-like classes and recipient-initiated steal/obtain-like classes, exactly the kind of role-coordinate inversion needed here. Source: `Knowledge/objects/papers/VerbNet-Representations-Subevent-Semantics-for-Transfer-Verbs--a5ba6a3b6ab2--0553b2454969/object.md`, citation `\cite{brown2019verbnet}`.
- FrameNet possession-transfer overview lists Giving (Donor, Recipient, Theme), Receiving (Donor, Recipient, Theme), Borrowing/Lending, Commerce buy/sell/pay, Sending, Transfer, and Exchange frames; it emphasizes old/new possessor roles and explicit recipient-like elements. Source: `Knowledge/objects/papers/Verbs-of-transfer-of-possession-in-FrameNet--77486f837d0a--cefbd69d2948/object.md`, citation `\cite{dimitrova2024verbs}`.
- FrameNet valence work gives concrete dative alternation examples for `give.v`: Donor.NP.Ext + Recipient.NP.Obj + Theme.NP.Dep (“He gives local charities money”) versus Donor.NP.Ext + Theme.NP.Obj + Recipient.PP[to].Dep (“He gives money to local charities”). Source: `Knowledge/objects/papers/an-API-to-Query-Valence-Patterns-in-FrameNet--1fa883daa26f--a670621d43e9/object.md`, citation `\cite{kabbachapi}`.
- A recent BabyLM dative controlled-rearing paper shows that direct and indirect evidence both affect dative preferences: balanced/removal conditions weaken but do not erase short-first effects; swapped direct evidence neutralizes/reverses direct preference components; global length manipulation moves the induced preference. This supports treating sparse orientation evidence and broader linguistic regularities separately. Source: `Knowledge/objects/papers/Both-Direct-and-Indirect-Evidence-Contribute-to-Dative-Alternation-Prefe--e651233c5660--5d760a653813/object.md`, citation `\cite{yao2025both}`.

## symmetry identification route file-only substrate

I materialized an auditable natural transfer-of-possession substrate in:

- script: `experiments/archive/representation_and_objectives/scripts/symmetry_identification_substrate.py`
- summary: `research/documents/representation_and_objectives/data/symmetry_identification_substrate/symmetry_substrate_summary.md`
- audit json: `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate/symmetry_substrate_audit.json`
- arm/eval JSONL files under `experiments/archive/representation_and_objectives/data/symmetry_identification_substrate`

The construction is deliberately CPU/file-only. It does not train, load a model, use GPU, run official evaluation, upload, or touch a leaderboard.

### Relation inventory

Seen coordinate:

- `give_to`: source-subject, recipient is participant 2.
- `receive_from`: recipient-subject, recipient is participant 1.

Held component:

- `cede_to`, `bequeath_to`: source-subject held relations; participant 2 receives.
- `acquire_from`, `inherit_from`: recipient-subject held relations; participant 1 receives.

The held component contains both orientations, so it can support internal comparisons while retaining a global swap ambiguity.

### Training arms

Common seen-coordinate rows: 192 labeled rows, true fraction 0.500.

Arm rows from the audit:

| arm | supervised rows | unsup rows | breaks held-seen orientation? | interpretation |
|---|---:|---:|---|---|
| exposure_only | 0 | 64 | false | raw held-event text only; no absolute labels |
| heldheld_only | 64 | 64 | false | internal held component coherence; invariant under global held swap |
| aligned_state_bridge | 80 | 64 | true | held component plus sparse correct natural state consequences for one held relation |
| inverted_state_bridge | 80 | 64 | true | same but changed-state labels flipped; should reverse latent orientation |
| decoupled_state_bridge | 80 | 64 | false | same text fertility without causal orientation; should not identify the component |
| mixed_event_bridge | 68 | 64 | true | sparse comparison between one held relation and seen transfer coordinate |

### Evaluation suites

| suite | rows | role of the suite |
|---|---:|---|
| held_held_coherence | 48 | tests whether a coherent held component is learnable; not positive evidence by itself |
| mixed_held_seen_orientation | 96 | primary symmetry-identification readout; labels change under global held swap |
| paired_state_conservation | 192 | changed and unaffected possession queries after held events |
| cross_frame_state_readout | 96 | held relation tested in different paraphrase/syntax from training |
| name_permutation_counterfactual | 64 | counterfactual names/argument order; blocks identity and position shortcuts |

The audit verifies train/eval name sets disjoint, train/eval changed-object sets disjoint, static objects disjoint from changed objects after repair, and position shortcuts at 0.500 on state evaluations. Held-held train labels are invariant under a global swap of held possessor roles; mixed held-seen labels are not; aligned/inverted bridges break the symmetry; decoupled bridge does not.

## What would count as the mechanism in the next low-cost learned run

This is not a classifier-accuracy task. The next experiment must preserve and report the symmetry structure.

The mechanism becomes stronger if the following pattern appears under a small pretrained encoder/probe or short adapter run from an identical calibrated base:

1. `heldheld_only` learns high held-held coherence but remains near random or systematically ambiguous on mixed held-seen orientation.
2. `aligned_state_bridge` or `mixed_event_bridge` orients unbridged held relations (`bequeath_to`, `acquire_from`, `inherit_from`) on mixed held-seen tests, not only the directly bridged `cede_to`.
3. `inverted_state_bridge` rotates/reverses the mixed held-seen confusion matrix in the predicted direction, rather than merely lowering confidence.
4. `decoupled_state_bridge` fails to produce the same orientation despite matched consequence text and row count.
5. Paired state readout moves pairs toward both-correct, especially unaffected-only→both-correct, without both-correct→affected-only erosion.
6. Cross-frame and name-permutation suites preserve the orientation, excluding exact lexical/surface and identity shortcuts.

The route weakens or must be rebuilt if:

- aligned, inverted, and decoupled bridges all improve equally: generic calibration or text regularization;
- only `cede_to` improves: per-lexeme grounding, not component orientation;
- held-held coherence never rises: no internally reusable component was learned, so orientation is not the bottleneck;
- mixed held-seen improves but both-correct conservation does not: orientation and protected state storage are separate mechanisms;
- affected accuracy rises while unaffected accuracy falls: the same update-prior failure as compact Entity V-B;
- exact phrase or role-string matching explains success.

## Minimal next execution

Before any H100 or BabyLM-scale continuation, run a small low-cost learned pilot on the symmetry identification route files:

- use an already available small pretrained encoder such as DeBERTa/RoBERTa from existing environment if available, or a tiny BiGRU/Transformer if model loading is the bottleneck;
- clone one identical calibrated reader/classification head across arms, or use minimal-pair pseudolikelihood, so prephase alignment design and bias analysis's generic head-calibration artifact cannot explain arm differences;
- train/evaluate only the six arms above with 2–3 seeds and small row counts;
- report separate confusion matrices for held-held, mixed held-seen, paired state states (`both_correct`, `affected_only`, `unaffected_only`, `neither`), cross-frame, and name-permutation suites;
- stop immediately if held-held does not fit or if decoupled/inverted controls do not separate from aligned.

This pilot should be treated as a symmetry-identification experiment. It should not be promoted as a general principle unless it shows component-level orientation, predicted inversion, decoupled separation, cross-frame persistence, and paired conservation together.
